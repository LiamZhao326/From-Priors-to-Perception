import os
import glob
import json
import time
from tqdm import tqdm
from google import genai
from google.genai import types

client = genai.Client(
    api_key="your_api_key",
    http_options={'api_version': 'v1alpha'}
)

# ======= 2. 路径配置 =======
RESULTS_DIR = r"path/to/results"

# 修复 1：补齐完整的 System Prompt，带上格式限制和示例
SYSTEM_PROMPT = """You are a rigorous academic evaluator for physical causality and reasoning. Please evaluate the "Model Output" against the "Ground Truth" based on the provided "Question".

[Evaluation Criteria]
Compare the Model Output with the Ground Truth. You must strictly decouple the evaluation of the final verdict (Real/Forged) from the evaluation of the reasoning process. Output a strict JSON object with the following three fields.

**CRITICAL RULE: The `score` MUST strictly align with the `accuracy`:**
- If `accuracy` = 0, the `score` MUST be 1 or 2.
- If `accuracy` = 1, the `score` MUST be 3, 4, or 5.

1. "reasoning": (String) A step-by-step analysis comparing the entities, state transitions, and causal logic. Point out any correct deductions, omissions, or hallucinations in the Model Output.
2. "accuracy": (Integer) 0 or 1. STRICTLY evaluate ONLY the final binary verdict. Does the Model Output correctly conclude whether the video is "real" or "forged" as stated in the Ground Truth? (1 if the final verdict matches, 0 if it contradicts or fails to answer). DO NOT let flawed reasoning or incorrect attribution lower the accuracy to 0. If the model guessed the right verdict for the wrong reasons, the accuracy MUST still be 1.
3. "score": (Integer) 1 to 5. The reasoning quality score, strictly constrained by the `accuracy` value:
   - 1: (Requires accuracy=0) Wrong verdict, and severe hallucinations or completely irrelevant analysis.
   - 2: (Requires accuracy=0) Wrong verdict, but correctly observed some relevant entities or actions.
   - 3: (Requires accuracy=1) Correct verdict, BUT the causal reasoning/attribution is completely wrong, missing, or hallucinated (i.e., "Right for the wrong reasons").
   - 4: (Requires accuracy=1) Correct verdict, and the reasoning basically matches the Ground Truth, with only minor detail flaws or verbosity.
   - 5: (Requires accuracy=1) Correct verdict, and perfect alignment with the Ground Truth in physical causal reasoning.

[Output Requirements]
Output ONLY a valid JSON object. Do not include any additional explanatory text, and do not use Markdown formatting blocks such as ```json.

[Example Output]
{
  "reasoning": "The Model Output correctly identified the video as 'forged', matching the Ground Truth verdict. However, it hallucinated the reason by claiming the pull-tab remained rigid under pressure, completely missing the actual temporal fallacy (causal reversal) stated in the Ground Truth. Because the final verdict is correct, accuracy is 1, but due to entirely flawed reasoning, the score is penalized to 3.",
  "accuracy": 1,
  "score": 3
}
"""

def evaluate_single_file(input_path, output_path):
    print(f"\n⏳ 正在评测文件: {os.path.basename(input_path)}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        target_samples = json.load(f)

    results = []
    processed_videos = set()

    # 断点续传机制
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            processed_videos = {item["video"] for item in results}
            
    print(f"该模型共 {len(target_samples)} 条数据，已评测 {len(processed_videos)} 条。")

    for sample in tqdm(target_samples, desc=f"评估进度"):
        video_id = sample['video']
        
        if video_id in processed_videos:
            continue

        eval_content = f"""
[Inputs]
- Question: {sample['prompt']}
- Ground Truth: {sample['ground_truth']}
- Model Output: {sample['model_output']}
"""
        
        wait_time = 2
        for attempt in range(8):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[types.Content(parts=[types.Part(text=eval_content)])],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT, 
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                
                eval_dict = json.loads(response.text.strip())
                
                evaluated_sample = sample.copy()
                evaluated_sample["evaluation"] = eval_dict
                
                results.append(evaluated_sample)
                processed_videos.add(video_id)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                    
                break 
                
            except Exception as e:
                error_msg = str(e)
                retry_keywords = ['503', '429', 'SSL', 'EOF', 'Connection', 'Timeout', 'timed out', 'disconnected', 'quota']
                
                if any(keyword in error_msg for keyword in retry_keywords):
                    short_error = error_msg.split(']')[0] + "]" if "]" in error_msg else error_msg[:60]
                    print(f"\n -> 网络/限流异常 ({short_error})，{wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, 60)
                else:
                    print(f"\n -> 严重异常或JSON解析失败，跳过: {error_msg}")
                    break
                    
        time.sleep(0.5)

def main():
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*_results.json"))
    
    for input_file in all_files:
        if input_file.endswith("_evaluated.json"):
            continue
            
        output_file = input_file.replace("_results.json", "_evaluated.json")
        evaluate_single_file(input_file, output_file)
        
    print("\n✅ 所有模型的评测全部完成！下班！")

if __name__ == "__main__":
    main()