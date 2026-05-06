import torch
import json
import os
import sys
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from accelerate import Accelerator
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"
MAX_NUM_FRAMES = 16 

# 挂载你的 MiniCPM 环境路径
CODE_PATH = 'path/to/your/MiniCPM-o-main'
sys.path.append(CODE_PATH)
os.chdir(CODE_PATH)

MODEL_PATH = 'MiniCPM-V-2_6'

def encode_video(video_path, max_num_frames=16):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = max(round(vr.get_avg_fps() / 1), 1)  
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

def main():
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print("⏳ 加载 Tokenizer ...")
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if accelerator.is_main_process:
        print("⏳ 加载 MiniCPM-V-2_6 (启动多卡 accelerate 并行) ...")

    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True,
        attn_implementation='flash_attention_2', 
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.local_process_index}
    )
    model.eval()

    if accelerator.is_main_process:
        print(f"\n📖 读取测试集: {TEST_DATA_PATH}")
        
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line.strip()) for line in f if line.strip()]
    
    local_results = []
    
    params = {
        "use_image_id": False,
        "max_slice_nums": 2
    }

    with accelerator.split_between_processes(test_samples) as local_samples:
        for sample in tqdm(local_samples, desc="推理进度 (以 Rank 0 为准)", disable=not accelerator.is_local_main_process):
            video_rel_path = sample['video'][0]
            video_path = os.path.join(VIDEO_ROOT, video_rel_path)
            
            human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "").replace("<video>", "").strip()
            ground_truth = sample["conversations"][1]["value"]

            try:
                frames = encode_video(video_path, max_num_frames=MAX_NUM_FRAMES)
            except Exception as e:
                print(f"⚠️ 无法读取视频 {video_path}, 跳过。原因: {e}")
                continue

            msgs = [{'role': 'user', 'content': frames + [human_prompt]}]
            current_max_inp_length = int(4096 + len(frames) * 199)

            with torch.inference_mode():
                answer = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    max_inp_length=current_max_inp_length,
                    **params
                )
            
            local_results.append({
                "video": video_rel_path,
                "prompt": human_prompt,
                "ground_truth": ground_truth,
                "model_output": answer.strip()
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