import torch
import json
import os
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"
MODEL_PATH = "path/to/Qwen3_5-VL-35B-A3B-Instruct"
def main():
    print("⏳ 加载 Qwen3-VL Processor ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"⏳ 加载 Qwen3-VL 30B MoE 基座模型 (启用 device_map='auto' 跨卡切分) ...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto", 
        trust_remote_code=True 
    )
    model.eval()

    print(f"\n📖 读取测试集: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line.strip()) for line in f if line.strip()]
    
    all_results = []
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    for sample in tqdm(test_samples, desc="推理进度"):
        video_rel_path = sample['video'][0]
        video_path = os.path.join(VIDEO_ROOT, video_rel_path)
        
        human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "")
        ground_truth = sample["conversations"][1]["value"]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 640 * 360, "nframes": 16, 
                    },
                    {"type": "text", "text": human_prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True 
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.95
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        all_results.append({
            "video": video_rel_path,
            "prompt": human_prompt,
            "ground_truth": ground_truth,
            "model_output": response.strip()
        })

        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 完美保存成功！总共生成 {len(all_results)} 条结果。")

if __name__ == "__main__":
    main()