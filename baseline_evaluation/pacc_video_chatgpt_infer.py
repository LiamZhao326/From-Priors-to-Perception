import torch
import json
import os
import sys
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

VIDEO_ROOT = "path/to/your/video/root"
TEST_DATA_PATH = "path/to/your/test_dataset.jsonl" 
OUTPUT_JSON_PATH = "path/to/results.json"

CODE_PATH = "path/to/your/Video-ChatGPT-main"
MODEL_NAME = os.path.join(CODE_PATH, "model_weights/LLaVA-7B-Lightening-v1-2")
PROJECTION_PATH = os.path.join(CODE_PATH, "model_weights/video_chatgpt-7B.bin")

sys.path.append(CODE_PATH)
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from video_chatgpt.constants import DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_VIDEO_TOKEN

def get_spatio_temporal_features_torch(features):
    t, s, c = features.shape
    temporal_tokens = torch.mean(features, dim=1)
    
    padding_size = 100 - t
    if padding_size > 0:
        temporal_tokens = torch.cat((temporal_tokens, torch.zeros(padding_size, c, device=features.device)), dim=0)
        
    spatial_tokens = torch.mean(features, dim=0)
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()
    return concat_tokens

def main():
    print("⏳ 加载 Video-ChatGPT 7B (暴力对齐单卡版) ...")
    disable_torch_init()
    
    original_torch_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['map_location'] = 'cpu'
        return original_torch_load(*args, **kwargs)
    torch.load = safe_load

    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(MODEL_NAME, PROJECTION_PATH)
    
    torch.load = original_torch_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 正在强制将所有组件对齐至: {device}")
    
    model.to(device).half()
    vision_tower.to(device).half()
    
    model.eval()
    vision_tower.eval()

    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN

    print(f"\n📖 读取测试集: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line.strip()) for line in f if line.strip()]

    all_results = []
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    
    for sample in tqdm(test_samples, desc="推理进度"):
        video_rel_path = sample['video'][0]
        video_path = os.path.join(VIDEO_ROOT, video_rel_path)

        human_prompt = sample["conversations"][0]["value"].replace("<video>\n", "").replace("<video>", "").strip()
        ground_truth = sample["conversations"][1]["value"]

        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            num_samples = min(100, total_frames)
            indices = np.linspace(0, total_frames - 1, num_samples).astype(int)
            frames = vr.get_batch(indices).asnumpy()
            
            # 处理器
            image_tensor = image_processor.preprocess(list(frames), return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.half().to(device) # 确保输入也在同一张卡上
            
        except Exception as e:
            print(f"⚠️ 无法读取视频 {video_path}, 跳过。原因: {e}")
            continue
        
        chunk_size_f = 16  
        frame_features_list = []
        
        with torch.no_grad():
            for i in range(0, image_tensor.shape[0], chunk_size_f):
                chunk = image_tensor[i : i + chunk_size_f]
                chunk_outs = vision_tower(chunk, output_hidden_states=True)
                chunk_hidden = chunk_outs.hidden_states[-2][:, 1:] 
                frame_features_list.append(chunk_hidden)
        
        frame_features = torch.cat(frame_features_list, dim=0)
        video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

        conv_mode = "video-chatgpt_v1"
        state = conv_templates[conv_mode].copy()
        state.append_message(state.roles[0], human_prompt + '\n' + DEFAULT_VIDEO_TOKEN)
        state.append_message(state.roles[1], None)
        prompt = state.get_prompt()
        
        prompt = prompt.replace(DEFAULT_VIDEO_TOKEN, replace_token, 1)
        
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).to(device)
        
        stop_str = state.sep if state.sep_style != SeparatorStyle.TWO else state.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=2048,
                stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        response = outputs.strip()
        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()

        all_results.append({
            "video": video_rel_path,
            "prompt": human_prompt,
            "ground_truth": ground_truth,
            "model_output": response
        })

        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        del image_tensor, frame_features, video_spatio_temporal_features, input_ids, output_ids
        torch.cuda.empty_cache()

    print(f"\n✅ 推理完成！结果已保存至: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()