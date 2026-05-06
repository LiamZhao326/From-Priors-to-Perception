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

import torch
import json
import warnings
from functools import partial
from tqdm import tqdm

warnings.filterwarnings("ignore")

VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"

CODE_PATH = "path/to/your/VideoLLaMA2-main"
MODEL_PATH = os.path.join(CODE_PATH, "DAMO-NLP-SG/VideoLLaMA2-7B-16F")
MAX_NUM_FRAMES = 16

sys.path.insert(0, CODE_PATH)

from videollama2.model import load_pretrained_model
from videollama2.mm_utils import process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollama2.constants import NUM_FRAMES, DEFAULT_VIDEO_TOKEN
from videollama2.utils import disable_torch_init

def model_init(model_path, device="cuda"):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path = model_path,
        model_base = None,
        model_name = model_name,
        torch_dtype = torch.float16,
        use_flash_attn = True,
        device_map={"": device}
    )

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    video_processor = partial(process_video, processor=processor, aspect_ratio=None, num_frames=MAX_NUM_FRAMES)
   
    return model, video_processor, tokenizer

def main():
    if local_rank == 0:
        print("⏳ 加载 VideoLLaMA2-7B-16F (统一 float16 精度) ...")

    disable_torch_init()
    device = "cuda"
    
    model, video_processor, tokenizer = model_init(MODEL_PATH, device)
    
    model.half()
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

    sys_msg = (
        "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe.\n<</SYS>>"
    )

    for sample in tqdm(local_samples, desc=f"Rank {local_rank} 推理进度"):
        video_rel_path = sample['video'][0]
        video_path = os.path.join(VIDEO_ROOT, video_rel_path)
        
        human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "").replace("<video>", "").strip()
        ground_truth = sample["conversations"][1]["value"]

        try:
            tensor = video_processor(video_path).to(device, dtype=torch.float16)
        except Exception as e:
            print(f"⚠️ 无法读取视频 {video_path}, 跳过。原因: {e}")
            continue
        
        video_tensor = [(tensor, 'video')]

        qa_text = DEFAULT_VIDEO_TOKEN + '\n' + human_prompt
        conv = [
            {'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': qa_text}
        ]

        prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        
        input_ids = tokenizer_multimodal_token(prompt, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().to(device)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device)

        keywords = [tokenizer.eos_token]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=video_tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=2048,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
        local_results.append({
            "video": video_rel_path,
            "prompt": human_prompt,
            "ground_truth": ground_truth,
            "model_output": response
        })

        del tensor, video_tensor, input_ids, attention_masks, output_ids
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
        print("✅ 完美保存成功！下班！")

if __name__ == "__main__":
    main()