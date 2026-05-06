import torch
import json
import os
import sys
from tqdm import tqdm
import warnings
from accelerate import Accelerator

warnings.filterwarnings("ignore")

VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"

CODE_PATH = "path/to/your/Video-LLaVA-main"
MODEL_PATH = "path/to/your/Video-LLaVA-model"

# 挂载官方库
sys.path.append(CODE_PATH)
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def main():
    accelerator = Accelerator()
    gpu_id = accelerator.local_process_index

    if accelerator.is_main_process:
        print("⏳ 加载 Video-LLaVA 7B (启动多进程数据并行) ...")

    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_PATH)

    tokenizer, model, processor, _ = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=True,  
        device=f'cuda:{gpu_id}',
        cache_dir='cache_dir'
    )
    
    video_processor = processor['video']
    model.eval()

    if accelerator.is_main_process:
        print(f"\n📖 读取测试集: {TEST_DATA_PATH}")
        
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line.strip()) for line in f if line.strip()]

    local_results = []
    
    with accelerator.split_between_processes(test_samples) as local_samples:
        for sample in tqdm(local_samples, desc=f"Rank {gpu_id} 推理进度", disable=not accelerator.is_local_main_process):
            video_rel_path = sample['video'][0]
            video_path = os.path.join(VIDEO_ROOT, video_rel_path)

            human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "").replace("<video>", "").strip()
            ground_truth = sample["conversations"][1]["value"]

            video_tensor_raw = video_processor(video_path, return_tensors='pt')['pixel_values']
            if isinstance(video_tensor_raw, list):
                tensor = [v.to(model.device, dtype=torch.float16) for v in video_tensor_raw]
            else:
                tensor = video_tensor_raw.to(model.device, dtype=torch.float16)

            num_frames = model.get_video_tower().config.num_frames
            fn_tokens = ' '.join([DEFAULT_IMAGE_TOKEN] * num_frames)
            inp = fn_tokens + '\n' + human_prompt

            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=2048,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            response = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            if response.endswith(stop_str):
                response = response[:-len(stop_str)].strip()

            local_results.append({
                "video": video_rel_path,
                "prompt": human_prompt,
                "ground_truth": ground_truth,
                "model_output": response
            })

            del tensor, video_tensor_raw, input_ids, output_ids
            torch.cuda.empty_cache()

    local_output_path = f"{OUTPUT_JSON_PATH}_rank_{gpu_id}.json"
    with open(local_output_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, ensure_ascii=False, indent=4)
    
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n⏳ 所有显卡推理完毕，正在合并结果...")
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
        
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
        print(f"正在保存最终版至 {OUTPUT_JSON_PATH} ...")
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print("✅ 完美保存成功！")

if __name__ == "__main__":
    main()