import os
import cv2
import json
import numpy as np
import torch
import textwrap
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

INPUT_TASKS_JSON = "path/to/your/input_tasks_json.json"
OUTPUT_JSON = "path/to/your/pacc_manual_tasks.json"
SAM2_CHECKPOINT = "path/to/your/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

drawing = False
ix, iy = -1, -1
bbox = None
points = []
labels = []
current_mask = None
mask_frame = -1
action_cmd = None
btn_rects = {}

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, bbox, points, labels, action_cmd, btn_rects

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_btn = False
        for name, (x1, y1, x2, y2) in btn_rects.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                action_cmd = name
                clicked_btn = True
                break
        
        if clicked_btn:
            return

        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            if abs(x - ix) > 10 and abs(y - iy) > 10:
                bbox = [min(ix, x), min(iy, y), max(ix, x), max(iy, y)]
            else:
                points.append([x, y])
                labels.append(1)
                
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)

def apply_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def dummy_callback(val):
    pass

def draw_bottom_ui(img):
    global btn_rects
    h, w = img.shape[:2]
    
    ui_scale = max(w / 1280.0, 0.4) 
    
    pad = int(10 * ui_scale)
    bh = int(45 * ui_scale)
    base_font_scale = 0.8 * ui_scale
    base_thick = max(1, int(2 * ui_scale))
    
    row1 = [
        ("prev_vid", "Prev Vid", (80, 80, 80)),
        ("next_vid", "Next Vid", (80, 80, 80)),
        ("prev", "< Frame", (150, 150, 150)),
        ("next", "Frame >", (150, 150, 150)),
        ("set_start", "Set START", (0, 128, 128)),
        ("set_end", "Set END", (0, 128, 128))
    ]
    row2 = [
        ("sam2", "Run SAM2", (0, 165, 255)),
        ("clear", "Clear Marks", (0, 0, 255)),
        ("erasure", "SAVE: ERASURE", (0, 200, 0)),
        ("freezing", "SAVE: FREEZING", (200, 0, 0))
    ]
    
    btn_rects.clear()
    
    bw1 = (w - (len(row1) + 1) * pad) // len(row1)
    y1_start = h - 2 * bh - 2 * pad
    for i, (name, text, color) in enumerate(row1):
        x1 = pad + i * (bw1 + pad)
        x2 = x1 + bw1
        y1 = y1_start
        y2 = y1_start + bh
        btn_rects[name] = (x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), max(1, int(2*ui_scale)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, base_font_scale, base_thick)[0]
        tx = x1 + (bw1 - text_size[0]) // 2
        ty = y1 + (bh + text_size[1]) // 2
        cv2.putText(img, text, (tx, ty), font, base_font_scale, (255, 255, 255), base_thick)

    bw2 = (w - (len(row2) + 1) * pad) // len(row2)
    y2_start = h - bh - pad
    for i, (name, text, color) in enumerate(row2):
        x1 = pad + i * (bw2 + pad)
        x2 = x1 + bw2
        y1 = y2_start
        y2 = y2_start + bh
        btn_rects[name] = (x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), max(1, int(2*ui_scale)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, base_font_scale, base_thick)[0]
        tx = x1 + (bw2 - text_size[0]) // 2
        ty = y1 + (bh + text_size[1]) // 2
        cv2.putText(img, text, (tx, ty), font, base_font_scale, (255, 255, 255), base_thick)

def main():
    global bbox, points, labels, current_mask, action_cmd, mask_frame

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True

    print("加载 SAM 2 模型中...")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    tasks_results = []
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            tasks_results = json.load(f)

    if not os.path.exists(INPUT_TASKS_JSON):
        print(f"[-] 输入文件 {INPUT_TASKS_JSON} 不存在，请检查路径！")
        return
        
    with open(INPUT_TASKS_JSON, 'r', encoding='utf-8') as f:
        input_tasks = json.load(f)

    cv2.namedWindow("PACC Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("PACC Annotator", mouse_callback)

    video_idx = 0
    while video_idx < len(input_tasks):
        task = input_tasks[video_idx]
        vid_path = task.get('video_path')
        scenario_text = task.get('target_fallacy_scenario', 'No scenario provided')

        if not vid_path or not os.path.exists(vid_path):
            video_idx += 1
            continue

        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        cv2.createTrackbar("Progress", "PACC Annotator", 0, total_frames - 1, dummy_callback)
        expected_trackbar_pos = 0

        bbox = None; points = []; labels = []; current_mask = None
        mask_frame = -1
        predictor_set = False
        need_redraw = True 
        fallacy_start = 0
        fallacy_end = total_frames - 1
        
        inner_action = None 
        force_seek = True

        while True:
            if need_redraw:
                ret = False
                new_frame = None

                if not force_seek:
                    ret, new_frame = cap.read()

                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, new_frame = cap.read()

                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        for _ in range(frame_idx):
                            r, _ = cap.read()
                            if not r:
                                break 
                        ret, new_frame = cap.read()

                if not ret:
                    if frame_idx == 0:
                        inner_action = "next_vid"
                        break
                    
                    frame_idx = max(0, frame_idx - 1)
                    cv2.setTrackbarPos("Progress", "PACC Annotator", frame_idx)
                    expected_trackbar_pos = cv2.getTrackbarPos("Progress", "PACC Annotator")
                    force_seek = True
                    continue

                frame = new_frame
                need_redraw = False
                force_seek = False 
            
            display_frame = frame.copy()

            if bbox:
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            for pt, lbl in zip(points, labels):
                color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
                cv2.circle(display_frame, tuple(pt), 5, color, -1)
            if current_mask is not None:
                display_frame = apply_mask(display_frame, current_mask)

            frame_h, frame_w = display_frame.shape[:2]
            ui_scale = max(frame_w / 1280.0, 0.4)
            
            text_scale = 0.6 * ui_scale
            text_thick = max(1, int(2 * ui_scale))
            line_step = int(40 * ui_scale)
            
            wrap_width = int(240 / ui_scale)
            wrapped_text = textwrap.wrap(f"Target: {scenario_text}", width=wrap_width)
            
            hud_height = int(140 * ui_scale) + len(wrapped_text) * line_step
            
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_w, hud_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

            status_text = f"Video: {video_idx+1}/{len(input_tasks)} | Frame: {frame_idx}/{total_frames-1} | Range: [{fallacy_start} -> {fallacy_end}]"
            
            y_pos_1 = int(45 * ui_scale)
            y_pos_2 = int(95 * ui_scale)
            
            cv2.putText(display_frame, status_text, (15, y_pos_1), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thick)
            cv2.putText(display_frame, "[ NAVIGATE ] Use Bottom Buttons for FULL Control", (15, y_pos_2), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thick)

            y_offset = int(145 * ui_scale)
            for line in wrapped_text:
                cv2.putText(display_frame, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 165, 255), text_thick + 1)
                y_offset += line_step

            draw_bottom_ui(display_frame)
            cv2.imshow("PACC Annotator", display_frame)
            
            key = cv2.waitKeyEx(30)
            cmd = action_cmd
            action_cmd = None

            trackbar_pos = cv2.getTrackbarPos("Progress", "PACC Annotator")
            if trackbar_pos != expected_trackbar_pos:
                frame_idx = trackbar_pos
                need_redraw = True
                force_seek = True
                predictor_set = False
                expected_trackbar_pos = trackbar_pos

            if key == -1 and cmd is None:
                continue

            if cmd == "prev_vid":
                inner_action = "prev_vid"
                break
            elif cmd == "next_vid":
                inner_action = "next_vid"
                break

            if key in (65361, 81, 2424832, ord('a'), ord('A')) or cmd == "prev": 
                frame_idx = max(frame_idx - 1, 0)
                cv2.setTrackbarPos("Progress", "PACC Annotator", frame_idx)
                expected_trackbar_pos = cv2.getTrackbarPos("Progress", "PACC Annotator")
                need_redraw = True
                force_seek = True
                predictor_set = False
                
            elif key in (65363, 83, 2555904, ord('d'), ord('D')) or cmd == "next": 
                frame_idx += 1
                if frame_idx >= total_frames:
                    total_frames = frame_idx + 1
                    cv2.setTrackbarMax("Progress", "PACC Annotator", total_frames - 1)
                cv2.setTrackbarPos("Progress", "PACC Annotator", frame_idx)
                expected_trackbar_pos = cv2.getTrackbarPos("Progress", "PACC Annotator")
                need_redraw = True
                force_seek = False
                predictor_set = False
                
            elif key == ord('s'):
                frame_idx += 10
                if frame_idx >= total_frames:
                    total_frames = frame_idx + 1
                    cv2.setTrackbarMax("Progress", "PACC Annotator", total_frames - 1)
                cv2.setTrackbarPos("Progress", "PACC Annotator", frame_idx)
                expected_trackbar_pos = cv2.getTrackbarPos("Progress", "PACC Annotator")
                need_redraw = True
                force_seek = True
                predictor_set = False
                
            elif key == ord('w'):
                frame_idx = max(frame_idx - 10, 0)
                cv2.setTrackbarPos("Progress", "PACC Annotator", frame_idx)
                expected_trackbar_pos = cv2.getTrackbarPos("Progress", "PACC Annotator")
                need_redraw = True
                force_seek = True
                predictor_set = False
                
            elif key == ord('c') or cmd == "clear":
                bbox = None; points = []; labels = []; current_mask = None; mask_frame = -1
            
            elif cmd == "set_start":
                fallacy_start = frame_idx
            elif cmd == "set_end":
                fallacy_end = frame_idx
            
            elif key == ord(' ') or cmd == "sam2": 
                if not predictor_set:
                    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    predictor_set = True
                input_box = np.array(bbox) if bbox else None
                input_point = np.array(points) if points else None
                input_label = np.array(labels) if labels else None
                if input_box is not None or input_point is not None:
                    masks, _, _ = predictor.predict(
                        point_coords=input_point, point_labels=input_label,
                        box=input_box, multimask_output=False
                    )
                    current_mask = masks[0]
                    mask_frame = frame_idx
            
            elif key == ord('1') or cmd == "erasure":
                fallacy_type = "Erasure"
                inner_action = "save"
                break
            elif key == ord('2') or cmd == "freezing":
                fallacy_type = "Freezing"
                inner_action = "save"
                break
            elif key == 27:
                inner_action = "quit"
                break

        if inner_action == "save" and (current_mask is not None or bbox is not None):
            new_record = {
                "video_path": vid_path,
                "fallacy_type": fallacy_type,
                "mask_frame": mask_frame,
                "start_frame": fallacy_start,
                "end_frame": fallacy_end,
                "bbox": bbox,
                "points": points,
                "labels": labels
            }

            if os.path.exists(OUTPUT_JSON):
                try:
                    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                        tasks_results = json.load(f)
                except Exception as e:
                    print(f"[-] 警告：读取外部修改的 JSON 失败 ({e})，将使用内存版本兜底。")

            existing_idx = -1
            for i, item in enumerate(tasks_results):
                if item["video_path"] == vid_path:
                    existing_idx = i
                    break
            
            if existing_idx != -1:
                tasks_results[existing_idx].update(new_record)
                print(f"[*] 已局部更新: {os.path.basename(vid_path)} -> {fallacy_type} | 区间: [{fallacy_start}, {fallacy_end}]")
            else:
                tasks_results.append(new_record)
                print(f"[*] 已保存新记录: {os.path.basename(vid_path)} -> {fallacy_type} | 区间: [{fallacy_start}, {fallacy_end}]")

            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(tasks_results, f, indent=4, ensure_ascii=False)
            
            video_idx += 1
            
        elif inner_action == "next_vid":
            video_idx += 1
            
        elif inner_action == "prev_vid":
            video_idx = max(0, video_idx - 1)
            
        elif inner_action == "quit":
            cap.release()
            break
        
        else:
            video_idx += 1

        cap.release()
        
    cv2.destroyAllWindows()
    print("所有任务处理完毕或手动退出！")

if __name__ == "__main__":
    main()