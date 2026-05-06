import os
import cv2
import json
import numpy as np
import torch
import shutil
import subprocess
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor

# ================= 配置区 =================
INPUT_TASKS_JSON = "path/to/your/input_tasks_json.json"
SAM2_CHECKPOINT = "path/to/your/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
PROPAINTER_DIR = "ProPainter"
TMP_DIR = Path("/tmp/pacc_render_tmp")

def ensure_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def extract_frames(vid_path, out_dir):
    cap = cv2.VideoCapture(vid_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return frames

def save_video(frames, out_path, fps):
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

def apply_static_freezing(frames, start_f, end_f, frozen_rgb, frozen_mask):
    out_frames = frames.copy()
    alpha = (frozen_mask > 0).astype(np.float32)[..., None]
    for i in range(start_f, end_f + 1):
        if 0 <= i < len(out_frames):
            out_frames[i] = (frozen_rgb * alpha + out_frames[i] * (1 - alpha)).astype(np.uint8)
    return out_frames

def apply_affine_tracking(frames, mask_f, start_f, end_f, frozen_rgb, frozen_mask):
    out_frames = frames.copy()
    gray_ref = cv2.cvtColor(frames[mask_f], cv2.COLOR_BGR2GRAY)
    
    p_init = cv2.goodFeaturesToTrack(gray_ref, maxCorners=200, qualityLevel=0.01, minDistance=5, mask=frozen_mask)
    if p_init is None:
        print("[!] 警告: 未检测到足够特征点，退化为静态冻结")
        return apply_static_freezing(frames, start_f, end_f, frozen_rgb, frozen_mask)
        
    h, w = frames[0].shape[:2]
    
    p_start = p_init.copy()
    p_prev = p_init.copy()
    gray_prev = gray_ref.copy()
    
    for i in range(mask_f + 1, end_f + 1):
        if i >= len(frames): break
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        p_curr, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, p_prev, None)
        
        valid = st.flatten() == 1
        if not np.any(valid): 
            print(f"  [!] 第 {i} 帧特征点全部丢失，后续退化为静态")
            break
        
        matrix, _ = cv2.estimateAffinePartial2D(p_start[valid], p_curr[valid])
        if matrix is not None:
            w_rgb = cv2.warpAffine(frozen_rgb, matrix, (w, h))
            w_mask = cv2.warpAffine(frozen_mask, matrix, (w, h))
            alpha = (w_mask > 0).astype(np.float32)[..., None]
            out_frames[i] = (w_rgb * alpha + out_frames[i] * (1 - alpha)).astype(np.uint8)
            
        p_prev = p_curr[valid].reshape(-1, 1, 2)
        p_start = p_start[valid].reshape(-1, 1, 2)
        gray_prev = gray_curr

    p_start = p_init.copy()
    p_prev = p_init.copy()
    gray_prev = gray_ref.copy()
    
    for i in range(mask_f - 1, start_f - 1, -1):
        if i < 0: break
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        p_curr, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, p_prev, None)
        
        valid = st.flatten() == 1
        if not np.any(valid): 
            break
        
        matrix, _ = cv2.estimateAffinePartial2D(p_start[valid], p_curr[valid])
        if matrix is not None:
            w_rgb = cv2.warpAffine(frozen_rgb, matrix, (w, h))
            w_mask = cv2.warpAffine(frozen_mask, matrix, (w, h))
            alpha = (w_mask > 0).astype(np.float32)[..., None]
            out_frames[i] = (w_rgb * alpha + out_frames[i] * (1 - alpha)).astype(np.uint8)
            
        p_prev = p_curr[valid].reshape(-1, 1, 2)
        p_start = p_start[valid].reshape(-1, 1, 2)
        gray_prev = gray_curr
        
    return out_frames

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True

    print("加载 SAM 2 视频模型中...")
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)

    with open(INPUT_TASKS_JSON, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    for idx, task in enumerate(tasks):
        vid_path = Path(task['video_path'])
        if not vid_path.exists():
            continue
            
        print(f"\n[{idx+1}/{len(tasks)}] 正在处理: {vid_path.name}")
        
        parent_name = vid_path.parent.name
        neg_parent_name = parent_name + "_negative"
        out_vid_path = vid_path.parent.with_name(neg_parent_name) / vid_path.name

        if out_vid_path.exists():
            print(f"  [√] 视频已存在，直接跳过: {out_vid_path.name}")
            continue

        fallacy_type = task['fallacy_type']
        track_mode = task.get('tracking_mode', 'static')
        use_raw_bbox = task.get('use_raw_bbox', False)
        start_f, end_f = task['start_frame'], task['end_frame']
        mask_f = task['mask_frame']
        
        bbox = task.get('bbox')
        if not bbox: continue
        bbox_arr = np.array(bbox) if isinstance(bbox[0], list) else np.array([bbox])

        frames_dir = ensure_dir(TMP_DIR / "frames")
        masks_dir = ensure_dir(TMP_DIR / "masks")
        pp_out_dir = ensure_dir(TMP_DIR / "propainter_out")

        frames = extract_frames(str(vid_path), frames_dir)
        h, w = frames[0].shape[:2]
        num_frames = len(frames)
        
        if mask_f == -1:
            mask_f = start_f

        masks_seq = np.zeros((num_frames, h, w), dtype=np.uint8)
        
        if use_raw_bbox:
            for i in range(start_f, end_f + 1):
                if 0 <= i < num_frames:
                    for b in bbox_arr:
                        x1, y1, x2, y2 = map(int, b)
                        masks_seq[i, y1:y2, x1:x2] = 255
        else:
            state = predictor.init_state(video_path=str(frames_dir))
            for i, b in enumerate(bbox_arr):
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=mask_f, obj_id=i, box=b
                )
            for out_f_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                if start_f <= out_f_idx <= end_f:
                    merged = (out_mask_logits.cpu().numpy() > 0.0).any(axis=0).squeeze()
                    masks_seq[out_f_idx] = (merged * 255).astype(np.uint8)

            del state
            torch.cuda.empty_cache()

        for i, m in enumerate(masks_seq):
            cv2.imwrite(str(masks_dir / f"{i:05d}.jpg"), m)

        fps = cv2.VideoCapture(str(vid_path)).get(cv2.CAP_PROP_FPS)
        final_frames = frames.copy()

        needs_erasure = fallacy_type == "Erasure" or track_mode == "freezing & erasure"
        
        if needs_erasure:
            print("  -> 执行 ProPainter 擦除重绘...")
            
            pp_w, pp_h = w, h
            if max(w, h) > 1280:
                scale = 1280.0 / max(w, h)
                pp_w, pp_h = int(w * scale), int(h * scale)
                
            pp_w, pp_h = pp_w - (pp_w % 8), pp_h - (pp_h % 8) 
            
            cmd = [
                "python", f"{PROPAINTER_DIR}/inference_propainter.py",
                "--video", str(frames_dir),
                "--mask", str(masks_dir),
                "--output", str(pp_out_dir),
                "--fp16",                  
                "--subvideo_length", "40", 
                "--width", str(pp_w),      
                "--height", str(pp_h)      
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            
            pp_mp4s = list(pp_out_dir.rglob("*.mp4"))
            if pp_mp4s:
                pp_cap = cv2.VideoCapture(str(pp_mp4s[0]))
                clean_frames = []
                while True:
                    ret, fr = pp_cap.read()
                    if not ret: break
                    
                    if fr.shape[:2] != (h, w):
                        fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_CUBIC)
                        
                    clean_frames.append(fr)
                pp_cap.release()
                final_frames = clean_frames
            else:
                print("  [!] ProPainter 生成失败，跳过擦除步骤。")

        if fallacy_type == "Freezing" or track_mode == "freezing & erasure":
            print(f"  -> 执行冻结覆盖 (模式: {track_mode})...")
            frozen_rgb = frames[mask_f].copy()
            frozen_mask = masks_seq[mask_f].copy()

            if track_mode == "affine":
                final_frames = apply_affine_tracking(final_frames, mask_f, start_f, end_f, frozen_rgb, frozen_mask)
            else:
                final_frames = apply_static_freezing(final_frames, start_f, end_f, frozen_rgb, frozen_mask)

        print(f"  -> 保存至: {out_vid_path}")
        save_video(final_frames, out_vid_path, fps)

if __name__ == "__main__":
    main()