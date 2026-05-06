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
from decord import VideoReader, cpu
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"

# 挂载 Flash-VStream 代码路径
CODE_PATH = 'path/to/flash-vstream-main'
sys.path.append(CODE_PATH)
os.chdir(CODE_PATH)

# 模型权重路径 (根据你提供的示例提取)
MODEL_PATH = "path/to/flash-vstream-model"
MAX_NUM_FRAMES = 16  # 严格对齐 16 帧

try:
    from flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from flash_vstream.conversation import conv_templates, SeparatorStyle
    from flash_vstream.model.builder import load_pretrained_model
    from flash_vstream.utils import disable_torch_init
    from flash_vstream.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
except ImportError as e:
    print(f"🚨 导入失败: {e}。请确认 CODE_PATH 是否指向 Flash-VStream 项目根目录。")
    sys.exit(1)


def get_16_frames(video_path, num_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def main():
    if local_rank == 0:
        print("⏳ 加载 Flash-VStream-7B (启动多卡硬隔离并行) ...")

    disable_torch_init()

    device = "cuda"
    model_name = get_model_name_from_path(MODEL_PATH)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH, 
        model_base=None, 
        model_name=model_name,
        load_8bit=False, 
        load_4bit=False,
        device=device
    )

    model.use_video_streaming_mode = True
    model.eval()

    if local_rank == 0:
        print(f"\n📖 读取测试集: {TEST_DATA_PATH}")
        
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line.strip()) for line in f if line.strip()]
    
    local_results = []

    chunk_size = len(test_samples) // world_size
    start_idx = local_rank * chunk_size
    end_idx = start_idx + chunk_size if local_rank < world_size - 1 else len(test_samples)
    local_samples = test_samples[start_idx:end_idx]

    for sample in tqdm(local_samples, desc=f"Rank {local_rank} 推理进度"):
        video_rel_path = sample['video'][0]
        video_path = os.path.join(VIDEO_ROOT, video_rel_path)
        
        human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "").replace("<video>", "").strip()
        ground_truth = sample["conversations"][1]["value"]

        try:
            video_frames = get_16_frames(video_path, num_frames=MAX_NUM_FRAMES)
        except Exception as e:
            print(f"⚠️ 无法读取视频 {video_path}, 跳过。原因: {e}")
            continue

        image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(device, dtype=torch.float16)

        model.video_embedding_memory = [] 
        with torch.inference_mode():
            model.embed_video_streaming(image_tensor.unsqueeze(0))

        conv = conv_templates["vicuna_v1"].copy()
        prompt_text = DEFAULT_IMAGE_TOKEN + '\n' + human_prompt
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=2048,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        response = tokenizer.decode(output_ids[0, input_token_len:], skip_special_tokens=True).strip()
        
        local_results.append({
            "video": video_rel_path,
            "prompt": human_prompt,
            "ground_truth": ground_truth,
            "model_output": response
        })

        del image_tensor, input_ids, output_ids
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    local_output_path = f"{OUTPUT_JSON_PATH}_rank_{local_rank}.json"
    with open(local_output_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, ensure_ascii=False, indent=4)
    
    if local_rank == 0:
        print("\n⏳ 等待其他进程完成推理并合并结果...")
        expected_files = [f"{OUTPUT_JSON_PATH}_rank_{i}.json" for i in range(world_size)]
        
        while not all(os.path.exists(f) for f in expected_files):
            time.sleep(2)
            
        all_results = []
        for f_path in expected_files:
            with open(f_path, "r", encoding="utf-8") as f:
                all_results.extend(json.load(f))
            os.remove(f_path)
            
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print("✅ 完美保存成功！")

if __name__ == "__main__":
    main()