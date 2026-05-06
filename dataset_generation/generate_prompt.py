import os
import glob
import json
import time
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
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue

        timestamp = i / fps
        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()

        extracted_parts.append(types.Part.from_text(text=f"\n[Frame Timestamp: {timestamp:.3f}s]"))
        extracted_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        
    cap.release()
    return extracted_parts

def process_video_prompts(input_folder, output_json_path):
    results = []
    processed_videos = set()

    if os.path.exists(output_json_path):
        with open(output_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            processed_videos = {item["video_path"] for item in results}

    json_files = glob.glob(os.path.join(input_folder, "*llmcaption.json"))
    tasks = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item.get("generation_method") == "AI_Generation":
                    tasks.append(item)

    print(f"扫描完毕：共找到 {len(json_files)} 个数据文件，提取到 {len(tasks)} 个 AI_Generation 任务，发现已存盘 {len(processed_videos)} 个。")

    for idx, item in enumerate(tasks, 1):
        video_path = item["video_path"]
        
        if video_path in processed_videos:
            print(f"[{idx}/{len(tasks)}] 跳过已处理: {os.path.basename(video_path)}")
            continue
            
        print(f"[{idx}/{len(tasks)}] 正在处理: {os.path.basename(video_path)}")
        
        video_parts = extract_frames_with_timestamps(video_path, num_frames=16)
        
        system_prompt = f"""
[SYSTEM] The following inputs are 16 key frames sampled from a continuous video. 
Each frame is labeled with its precise timestamp. 
You must treat them as a temporal sequence to analyze physical causality.

### Dynamic Inputs
* **Visual Frames:** [Provided above as image sequence]
* **Camera & Style:** {item.get('visual_style_and_camera', '')}
* **Original Caption:** {item.get('visual_fact_caption', '')}
* **Target Fallacy Scenario:** {item.get('target_fallacy_scenario', '')}

### Core Task & Directives
Based on the `Visual Frames`, `Camera & Style`, and `Original Caption` of the source video, you must write a highly precise English prompt for a video generation model to guide it in generating the given `Target Fallacy Scenario`, creating a modified new version of the video.

* **Establish Baseline & Execute Replacement:** The prompt must use the `Original Caption` as the realistic baseline for the environment and subjects, but you must **completely replace** the logical, natural outcome of the original video with the counter-intuitive or absurd events described in the `Target Fallacy Scenario`.
* **Absolute Forcing Principle:** The prompt must force the video model to prioritize executing the `Target Fallacy Scenario`. You are **strictly prohibited** from attempting to "correct" or rationalize these anti-physical phenomena. You must command these impossible events directly as objective visual facts.

### Core Prompting Rules
* **Structural Front-loading:** Always place the camera movement, visual style, or perspective (e.g., "High quality, Low angle shot") at the very beginning of the prompt to set the visual tone.
* **Strong Explicit Negation & Contrast:** You MUST use strong negative words (e.g., capitalized "NOT", "completely misses", "without") or extreme contrasting modifiers (e.g., "tiny pebble" vs. "towering geyser") to definitively break common sense and semantic priors.
* **Temporal Slicing Expression:** For sudden state changes or interventions, use clear temporal anchors to define the sequence of events (e.g., "Upon impact", "Phase 1...", "Trigger:", "Phase 2...").
* **Summary Qualitative Label:** Conclude the entire prompt with an extremely brief declarative sentence that definitively labels the physical anomaly (e.g., "A clean miss.", "The material behaves like rubber.").

### Few-Shot Examples
[Example 1] A bowling ball rolls fast towards the single white pin. However, it misses the target completely. The ball rolls past the pin on the right side without making any contact. The pin remains standing perfectly still. A clean miss.

[Example 2] A pink vase falls and hits the rusty floor. Upon impact, the object exhibits soft body physics. It drastically deforms, squashes, and compresses, then bounces back with elasticity. The material behaves like rubber. High-speed camera, slow motion, detailed texture of the deformation.

[Example 3] High quality, realistic footage. Dynamic tracking shot matching the original camera movement. Phase 1: The original dog is walking/running on the grass. It maintains its original posture and movement rhythm. Trigger: Mid-stride, a surreal transformation occurs. Phase 2: The animal smoothly morphs into a ginger tabby cat. The cat occupies the exact same position and path as the dog. The cat continues the walking motion seamlessly. Environment Constraint: The grass texture, lighting, and background scenery remain identical to the original footage. Only the animal species changes.

[Example 4] High quality, realistic footage starting from the reference image. The metal knife blade presses down firmly into the orange mandarin. Upon cutting pressure, the orange does NOT split into two halves. Instead, it instantly fractures and crumbles into multiple (more than 6) small, separate orange segments and irregular peel pieces that scatter slightly across the textured glass cutting board. Juice sprays slightly. The knife continues downwards to the board.

[Example 5] Low angle shot. A tiny pebble drops vertically into the center of the calm lake. The moment it touches the surface, the water erupts violently. A massive, towering geyser of water shoots fifty meters high into the air immediately upon contact. The impact site generates a giant explosion of white foam and heavy waves radiating outwards. The scene demonstrates extreme kinetic amplification.

[Example 6] Low angle shot. A massive grey boulder falls from the sky and hits the water surface. The water surface remains completely undisturbed and flat.

### Output Constraint
* Return a single valid JSON object. Do NOT use markdown code blocks (like ```json).
* The JSON must contain exactly one key: "generated_video_prompt".

{{
  "generated_video_prompt": "String. The final constructed English prompt."
}}
"""
        
        request_parts = []
        request_parts.extend(video_parts)
        request_parts.append(types.Part.from_text(text=system_prompt))

        wait_time = 10
        for attempt in range(8):
            try:
                response = client.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents=[types.Content(parts=request_parts)],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                
                response_data = json.loads(response.text)
                final_prompt = response_data.get("generated_video_prompt", "")
                
                new_item = {
                    "video_path": video_path,
                    "generated_video_prompt": final_prompt
                }
                
                results.append(new_item)
                processed_videos.add(video_path)
                
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                    
                print(f"  -> 成功生成: {final_prompt[:60]}...") 
                break 
                
            except Exception as e:
                error_msg = str(e)
                if '503' in error_msg or '429' in error_msg:
                    print(f"  -> 遭遇服务器限流，{wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, 120)
                else:
                    print(f"  -> 发生异常，跳过此条: {error_msg}")
                    break
            
        time.sleep(6.5)

if __name__ == "__main__":
    target_folder = r"E:\workspace\ML\DeepLearning\PACC\all data"
    output_file = r"E:\workspace\ML\DeepLearning\PACC\all data\Kling_Prompts_Output.json"
    
    process_video_prompts(target_folder, output_file)