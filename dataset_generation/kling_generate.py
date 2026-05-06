import os
import cv2
import json
import time
import base64
import requests
import jwt

# ================= 配置区 =================
AK = ""
SK = ""
INPUT_JSON = "path/to/your/input/prompt.json"
OUTPUT_JSON = "./pacc_results.json"

def get_jwt_token():
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": AK,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5
    }
    return jwt.encode(payload, SK, headers=headers)

def extract_first_frame_base64(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"[-] 无法读取视频帧: {video_path}")
        return None
        
    orig_h, orig_w = frame.shape[:2]
    orig_ratio = orig_w / orig_h
    
    standard_resolutions = [
        (1280, 720),  # 16:9
        (720, 1280),  # 9:16
        (720, 720),   # 1:1
        (960, 720),   # 4:3
        (720, 960),   # 3:4
    ]
    
    closest_res = min(standard_resolutions, key=lambda res: abs((res[0]/res[1]) - orig_ratio))
    target_w, target_h = closest_res
    
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2
    cropped_frame = resized_frame[y_offset:y_offset+target_h, x_offset:x_offset+target_w]
    
    # 5. 压缩为 JPEG 并转 Base64
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    _, buffer = cv2.imencode('.jpg', cropped_frame, encode_param)
    
    return base64.b64encode(buffer).decode('utf-8')

def submit_task(image_b64, prompt):
    url = "https://api-beijing.klingai.com/v1/videos/image2video"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_jwt_token()}"
    }

    payload = {
        "model_name": "kling-v3",
        "image": image_b64,
        "prompt": prompt,
        "duration": "5",
        "mode": "pro",
        "sound": "off"
    }
    
    res = requests.post(url, headers=headers, json=payload).json()
    if res.get('code') == 0:
        return res['data']['task_id']
    else:
        print(f"[-] 任务提交失败: {res}")
        return None

def poll_task(task_id):
    """轮询下载"""
    url = f"https://api-beijing.klingai.com/v1/videos/image2video/{task_id}"
    while True:
        headers = {"Authorization": f"Bearer {get_jwt_token()}"}
        res = requests.get(url, headers=headers).json()
        status = res['data']['task_status']
        
        if status == "succeed":
            return res['data']['task_result']['videos'][0]['url']
        elif status in ["failed", "killed"]:
            print(f"[-] 任务生成失败: {res['data'].get('task_status_msg')}")
            return None
            
        print(f"[*] 任务 {task_id} 处理中... 10秒后重试")
        time.sleep(10)

def main():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
    results = []
    processed_files = set()
    
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            results = json.load(f)
            for r in results:
                vid_path = r.get('video_path', '')
                if vid_path:
                    processed_files.add(os.path.basename(vid_path.replace('\\', '/')))

    for i, task in enumerate(tasks):
        print(f"\n================ 处理进度: {i+1}/{len(tasks)} ================")
        
        path_parts = task['video_path'].replace('\\', '/').split('/')
        category_dir = path_parts[-2]
        filename = path_parts[-1]
        
        if filename in processed_files:
            print(f"[*] 该视频已成功生成并记录，直接跳过: {filename}")
            continue

        local_vid_path = os.path.join(category_dir, filename)
        output_dir = f"{category_dir}_negative"
        save_path = os.path.join(output_dir, filename)
        prompt = task['generated_video_prompt']
        
        print(f"[*] 目标视频: {local_vid_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        image_b64 = extract_first_frame_base64(local_vid_path)
        if not image_b64:
            continue
            
        task_id = None
        while not task_id:
            task_id = submit_task(image_b64, prompt)
            if not task_id:
                print(f"[*] 接口繁忙或并发受限，原地等待 30 秒后重试...")
                time.sleep(30)
                
        print(f"[*] 成功提交，Task ID: {task_id}")
        
        video_url = poll_task(task_id)
        if video_url:
            print(f"[+] 云端生成完成！正在下载实体视频...")
            video_bytes = requests.get(video_url).content
            
            with open(save_path, 'wb') as v_file:
                v_file.write(video_bytes)
            print(f"[+] 实体视频已存入: {save_path}")
            
            task['kling_result_url'] = video_url
            task['local_neg_video_path'] = save_path
            results.append(task)
            
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

            processed_files.add(filename)

    print(f"\n所有 {len(tasks)} 个任务流水线处理完毕！")

if __name__ == "__main__":
    main()