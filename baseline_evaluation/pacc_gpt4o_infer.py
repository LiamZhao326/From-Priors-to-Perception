import os
import json
import time
import base64
import cv2
import numpy as np
from tqdm import tqdm
from openai import OpenAI


API_KEY = "your_api_key"

VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"

def extract_frames_to_openai_format(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    content_list = []
    content_list.append({
        "type": "text", 
        "text": "The following are 16 key frames sampled from a video in chronological order with their precise timestamps. Please analyze them sequentially to answer the question.\n"
    })

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue

        timestamp = i / fps
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        content_list.append({"type": "text", "text": f"[Frame Timestamp: {timestamp:.3f}s]"})
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "low"  
            }
        })
        
    cap.release()
    return content_list

def main():
    results = []
    processed_videos = set()
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            results = json.load(f)
            processed_videos = {item["video"] for item in results}

    print(f"📖 读取测试集: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"总计 {len(test_samples)} 个测试用例，发现已存盘 {len(processed_videos)} 个。")

    for idx, sample in enumerate(tqdm(test_samples, desc="GPT-4o API 推理进度")):
        video_rel_path = sample['video'][0]
        
        if video_rel_path in processed_videos:
            continue

        video_path = os.path.join(VIDEO_ROOT, video_rel_path)
        human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "").replace("<video>", "").strip()
        ground_truth = sample["conversations"][1]["value"]

        content_parts = extract_frames_to_openai_format(video_path, num_frames=16)
        if not content_parts:
            print(f"\n⚠️ 无法读取视频 {video_path}, 跳过。")
            continue

        content_parts.append({"type": "text", "text": f"\n\nQuestion: {human_prompt}"})

        wait_time = 5
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": content_parts}
                    ],
                    max_tokens=2048,
                    temperature=0.0
                )
                
                response_text = response.choices[0].message.content.strip()

                results.append({
                    "video": video_rel_path,
                    "prompt": human_prompt,
                    "ground_truth": ground_truth,
                    "model_output": response_text
                })
                processed_videos.add(video_rel_path)
                
                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                    
                break 
                
            except Exception as e:
                error_msg = str(e)
                print(f"\n -> 遭遇异常 ({error_msg[:60]})，{wait_time} 秒后重试...")
                time.sleep(wait_time)
                wait_time *= 2

if __name__ == "__main__":
    main()