import os
import glob
import json
import time
import base64
import cv2
import numpy as np
from google import genai
from google.genai import types

client = genai.Client(
    api_key="your_api_key",
    http_options={'api_version': 'v1alpha'}
)

def extract_frames_with_timestamps(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    extracted_parts = []
    
    extracted_parts.append(types.Part(text="""
[SYSTEM] The following inputs are 16 key frames sampled from a continuous video. 
Each frame is labeled with its precise timestamp. 
You must treat them as a temporal sequence to analyze physical causality.
    
### Role & Core Function
You are an expert in **Computer Vision (CV)**, specializing in video semantic understanding and strict visual fact extraction.

**Core Guidelines:**
* **Visual Evidence Only:** All analysis must be strictly based on the visible pixel content of the video. Do NOT hallucinate objects or actions not present.
* **No Subjectivity:** Strictly describe *what* is happening. Do NOT infer *why* (physics laws, gravity, momentum), *intent*, or *emotions*.

### Input Context
* **Context: Input Video:** The video file to be analyzed.

### Task Execution
**Visual Fact Anchoring**
* **Basis:** The Input Video.
* **Task(Caption):** Generate a strictly objective visual description. Accurately describe the factual phenomena occurring in the video. Focus ONLY on **Subjects**, **Actions**, and **Environment**. 
    * *Constraint:* Prohibition on explaining "why it happens," "what it implies," or summarizing the overarching event (e.g., say "the glass shatters into pieces," do NOT say "the glass breaks because of the impact").

### Output Format
*Return a single valid JSON object. Do NOT use markdown code blocks (like ```json).*

{
  "visual_fact_caption": "String. Strict objective description of actions and objects."
}
    """))

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

def process_folder(folder_path, output_json_path):
    results = []
    processed_videos = set()

    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            processed_videos = {item["video_path"] for item in results}

    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))


    print(f"总计 {len(video_files)} 个视频，发现已存盘 {len(processed_videos)} 个。")

    for idx, video_path in enumerate(video_files, 1):
        if video_path in processed_videos:
            print(f"[{idx}/{len(video_files)}] 跳过已处理: {os.path.basename(video_path)}")
            continue
            
        print(f"[{idx}/{len(video_files)}] 正在处理: {os.path.basename(video_path)}")
        video_parts = extract_frames_with_timestamps(video_path, num_frames=16)
        
        wait_time = 10
        for attempt in range(8):
            try:
                response = client.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents=[types.Content(parts=video_parts)],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                response_data = json.loads(response.text)

                if isinstance(response_data, list) and len(response_data) > 0:
                    response_data = response_data[0]

                video_data = {"video_path": video_path} 
                video_data.update(response_data)        
                
                results.append(video_data)
                processed_videos.add(video_path)
                
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                    
                print("  -> 成功提取并存盘")
                break
                
            except Exception as e:
                error_msg = str(e)
                
                retry_keywords = ['503', '429', 'SSL', 'EOF', 'Connection', 'Timeout', 'timed out', 'disconnected']
                
                if any(keyword in error_msg for keyword in retry_keywords):
                    short_error = error_msg.split(']')[0] + "]" if "]" in error_msg else error_msg[:50]
                    print(f"  -> 遭遇网络断流或接口限流 ({short_error})，{wait_time} 秒后自动重试...")
                    
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, 120)
                else:
                    print(f"  -> 发生非网络严重异常，跳过此条: {error_msg}")
                    break

        time.sleep(6.5)

# --- 运行配置 ---
target_folder = r"path/to/your/target/folder"
output_file = r"path/to/your/out_put_file.json"

process_folder(target_folder, output_file)