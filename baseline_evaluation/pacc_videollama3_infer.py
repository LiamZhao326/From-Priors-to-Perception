import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import warnings
from accelerate import Accelerator
from tqdm import tqdm

warnings.filterwarnings("ignore")

VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"
MODEL_PATH = "path/to/your/VideoLLaMA3-7B"
def main():
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print("⏳ 加载 Processor ...")
        
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    if accelerator.is_main_process:
        print("⏳ 加载基座模型 (必须与训练时完全一致的 4-bit nf4 量化) ...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=[
            "vision_encoder",   
            "mm_projector"      
        ]
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
        device_map={"": accelerator.local_process_index} 
    )

    model.eval()

    if accelerator.is_main_process:
        print(f"\n📖 读取测试集: {TEST_DATA_PATH}")
        
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line.strip()) for line in f if line.strip()]
    
    local_results = []

    with accelerator.split_between_processes(test_samples) as local_samples:
        for sample in tqdm(local_samples, desc="推理进度 (以 Rank 0 为准)", disable=not accelerator.is_local_main_process):
            video_rel_path = sample['video'][0]
            video_path = os.path.join(VIDEO_ROOT, video_rel_path)
            
            human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "")
            ground_truth = sample["conversations"][1]["value"]

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": {"video_path": video_path, "nframes": 16}},
                        {"type": "text", "text": human_prompt}
                    ]
                }
            ]

            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True, 
                return_tensors="pt"
            )

            inputs = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    inputs[k] = v.to(torch.bfloat16)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512)

            response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            local_results.append({
                "video": video_rel_path,
                "prompt": human_prompt,
                "ground_truth": ground_truth,
                "model_output": response.strip()
            })

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
        print(f"正在保存最终版至 {OUTPUT_JSON_PATH} ...")
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print("✅ 完美保存成功！")

if __name__ == "__main__":
    main()