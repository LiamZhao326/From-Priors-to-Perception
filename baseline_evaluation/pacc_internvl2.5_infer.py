import os
import sys
import time

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if cvd:
    devices = [d.strip() for d in cvd.split(",")]
    if local_rank < len(devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
# ========================================================

import torch
import json
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from decord import VideoReader, cpu
from accelerate import Accelerator
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

CODE_PATH = 'path/to/internvl2.5-main'
sys.path.append(CODE_PATH)
os.chdir(CODE_PATH)

VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"

MODEL_PATH = os.path.join(CODE_PATH, "InternVL2_5-8B")
MAX_NUM_FRAMES = 16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=16):
    start, end = bound if bound else (-100000, 100000)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=16):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        
    return torch.cat(pixel_values_list), num_patches_list

def main():
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print("⏳ 加载 Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)

    if accelerator.is_main_process:
        print(f"⏳ 加载 InternVL2-8B (启动多卡 accelerate 并行, 当前进程卡号: {accelerator.local_process_index}) ...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map={"": accelerator.local_process_index}
    ).eval()

    if accelerator.is_main_process:
        print(f"\n📖 读取测试集: {TEST_DATA_PATH}")
        
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line.strip()) for line in f if line.strip()]
    
    local_results = []
    generation_config = dict(max_new_tokens=2048, do_sample=False)
    
    system_message = 'You are a multimodal video analysis assistant, required to answer user questions based on temporal video clips.'

    with accelerator.split_between_processes(test_samples) as local_samples:
        for sample in tqdm(local_samples, desc="推理进度 (以 Rank 0 为准)", disable=not accelerator.is_local_main_process):
            video_rel_path = sample['video'][0]
            video_path = os.path.join(VIDEO_ROOT, video_rel_path)
            
            human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "").replace("<video>", "").strip()
            ground_truth = sample["conversations"][1]["value"]

            try:
                pixel_values, num_patches_list = load_video(video_path, max_num=1, num_segments=MAX_NUM_FRAMES)
            except Exception as e:
                print(f"⚠️ 无法读取视频 {video_path}, 跳过。原因: {e}")
                continue

            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
            question = video_prefix + human_prompt

            with torch.inference_mode():
                answer = model.chat(
                    tokenizer, 
                    pixel_values, 
                    question, 
                    generation_config, 
                    system_message=system_message,
                    num_patches_list=num_patches_list, 
                    history=None
                )
            
            local_results.append({
                "video": video_rel_path,
                "prompt": human_prompt,
                "ground_truth": ground_truth,
                "model_output": answer.strip()
            })

            del pixel_values
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    local_output_path = f"{OUTPUT_JSON_PATH}_rank_{accelerator.local_process_index}.json"
    with open(local_output_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, ensure_ascii=False, indent=4)
    
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n⏳ 所有显卡推理完毕，正在合并结果...")
        all_results = []
        
        for i in range(accelerator.num_processes):
            rank_file = f"{OUTPUT_JSON_PATH}_rank_{i}.json"
            try:
                with open(rank_file, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))
                os.remove(rank_file)
            except Exception as e:
                print(f"⚠️ 读取 Rank {i} 的文件时出错: {e}")

        print(f"💾 合并完成！总共生成 {len(all_results)} 条结果。")
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print("✅ 完美保存成功！")

if __name__ == "__main__":
    main()