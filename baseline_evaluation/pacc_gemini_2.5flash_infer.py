import os
import json
import time
import base64
import cv2
import numpy as np
from tqdm import tqdm
from google import genai
from google.genai import types


client = genai.Client(
    api_key="your_api_key"
)

VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"

def extract_frames_with_timestamps(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    extracted_parts = []
    
    extracted_parts.append(types.Part(text="The following are 16 key frames sampled from a video in chronological order with their precise timestamps. Please analyze them sequentially to answer the question.\n"))

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue

        timestamp = i / fps
        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()

        extracted_parts.append(types.Part(text=f"\n[Frame Timestamp: {timestamp:.3f}s]"))
        extracted_parts.append(types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=base64.b64encode(img_bytes).decode('utf-8')
            )
        ))
        
    cap.release()
    return extracted_parts

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

    for idx, sample in enumerate(tqdm(test_samples, desc="Gemini API 推理进度")):
        video_rel_path = sample['video'][0]
        
        if video_rel_path in processed_videos:
            continue

        video_path = os.path.join(VIDEO_ROOT, video_rel_path)
        human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "").replace("<video>", "").strip()
        ground_truth = sample["conversations"][1]["value"]

        video_parts = extract_frames_with_timestamps(video_path, num_frames=16)
        if not video_parts:
            print(f"\n⚠️ 无法读取视频 {video_path}, 跳过。")
            continue

        final_parts = video_parts + [types.Part(text=f"\n\nQuestion: {human_prompt}")]

        wait_time = 10
        for attempt in range(8):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=[types.Content(parts=final_parts)]
                )
                
                response_text = response.text.strip()

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

                retry_keywords = ['503', '429', 'SSL', 'EOF', 'Connection', 'Timeout', 'timed out', 'disconnected', 'quota']
                
                if any(keyword in error_msg for keyword in retry_keywords):
                    short_error = error_msg.split(']')[0] + "]" if "]" in error_msg else error_msg[:60]
                    print(f"\n -> 遭遇网络波动或 API 限流 ({short_error})，{wait_time} 秒后自动重试...")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, 120)
                    print(f"\n -> 发生非网络严重异常，跳过此条: {error_msg}")
                    break

        time.sleep(4)

if __name__ == "__main__":
    main()