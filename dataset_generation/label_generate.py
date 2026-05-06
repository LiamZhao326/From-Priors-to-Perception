import os
import glob
import json
import time
from google import genai
from google.genai import types

API_KEY = "your_api_key"
MODEL_NAME = "gemini-3-pro-preview"

# 文件夹与数据源配置
POS_FOLDER = r"path/to/your/positive/videos"
NEG_FOLDER = r"path/to/your/negative/videos"
METADATA_JSON_PATH = r"path/to/your/metadata.json"
OUTPUT_JSON_PATH = r"path/to/your/output"

PROMPT_TEMPLATE = """
### Input Information
`Original Visual Caption`: {visual_fact_caption}
`Target Relative Scenario`: {target_fallacy_scenario}
`Trap Category`: {selected_typical_scenario}

================================

### Task Objective and Rules
You are an expert data annotator specializing in visual logic datasets. Based on the [Input Information] above, construct a pair of structured Three-Step CoT (Chain of Thought) data (Original Sample vs. Relative Sample) for training Video-LLMs to resist "semantic prior hallucinations."

Note: BOTH samples are on equal footing. They are both physically plausible.

**1. Observation**
- Rule: Generate a strictly objective visual description. Accurately describe the factual phenomena, focusing ONLY on **Subjects**, **Actions**, **Environment**, and precise **Spatial Relationships/Contact**.
- Original Sample Observation: Base this entirely on the `Original Visual Caption`.
- Relative Sample Observation: Use the `Original Visual Caption` as the background setting, and seamlessly integrate the modified actions from the `Target Relative Scenario`. Explicitly describe the interaction details (e.g., the level of force applied, the angle, or the lack of expected reaction) exactly as provided in the text.
- Taboo: Absolutely NO hallucinations, and NO causal speculations or semantic expectations.

**2. Attribution**
- Rule: Based on the content of the "[Observation]", strictly use the `PACC Category Dictionary` below to explain the physical logic.
- Original Sample Attribution: Precisely point out how the presence OR absence of the expected consequences logically and correctly aligns with the specific interaction facts described in the original caption (e.g., sufficient force vs. insufficient force). 
- Relative Sample Attribution: Similarly, explain how the modified scenario's outcome strictly aligns with the explicit factual descriptions provided in the `Target Relative Scenario`. For both attributions, emphasize that the explicitly described objective facts dictate the physical outcome, overriding any semantic priors.

**3. Verdict**
- Original Sample Verdict: Clearly summarize in one sentence that the video is a visually authentic recording that perfectly conforms to physical logic.
- Relative Sample Verdict: Clearly summarize in one sentence that the video is a visually authentic recording that perfectly conforms to physical logic.

================================

### PACC Category Dictionary
* **Core Principle:**
Violation of "Consistency between Statistical Priors and Visual Facts." These are NOT physical fallacies, but "visually authentic" adversarial traps designed to penalize models that hallucinate expected consequences based on semantic scripts (e.g., "hitting implies breaking") while blatantly ignoring strict visual evidence that the required physical threshold (force, angle) was never met.

* **Specific Adversarial Type: Consequence Arrest**
* **Definition:** A physical interaction or contact actually occurs, but due to insufficient force, inadequate angle, or unexpected resistance, the semantically expected causal outcome (e.g., shattering, spilling, moving) rightfully fails to manifest.
* **Typical Scenarios:**
    1.  **Insufficient Impact:** An object physically strikes another, but the applied force is too weak to cause the expected structural damage or displacement (e.g., a hammer strikes a pane of glass, but due to low impact force, the glass remains completely intact without any cracks).
    2.  **Threshold Failure:** An action intended to change an object's state or position is performed, but it fails to cross the critical physical threshold required for the change (e.g., a hand tilts a cup of water, but the tilt angle is too shallow, so the water does not pour out; a person pushes a heavy box, but it remains completely stationary).

================================

### Output Requirements
Strictly return the output in JSON format, containing a nested structure for both the original and relative samples. The `sft_response` must combine the previous three steps into a coherent and complete paragraph:
{
  "original_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "...",
    "sft_response": "**Observation**:...\n **Attribution**:...\n **Verdict**:..."
  },
  "relative_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "...",
    "sft_response": "**Observation**:...\n **Attribution**:...\n **Verdict**:..."
  }
}
"""

def build_metadata_index(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    index = {}
    for item in data:
        filename = os.path.basename(item['video_path'].replace('\\', '/'))
        index[filename] = item
    return index

def main():
    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})

    print("正在加载元数据...")
    metadata_index = build_metadata_index(METADATA_JSON_PATH)

    pos_files = {os.path.basename(p) for p in glob.glob(os.path.join(POS_FOLDER, "*.mp4"))}
    neg_files = {os.path.basename(p) for p in glob.glob(os.path.join(NEG_FOLDER, "*.mp4"))}
    matched_files = list(pos_files.intersection(neg_files))
    
    print(f"找到正样本 {len(pos_files)} 个，负样本 {len(neg_files)} 个。")
    print(f"成功匹配同名文件对：{len(matched_files)} 个。")

    results = []
    processed_files = set()
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            results = json.load(f)
            processed_files = {item['filename'] for item in results}

    print(f"读取到已存盘进度：{len(processed_files)} 个，本次实际需处理：{len(matched_files) - len(processed_files)} 个。")

    for idx, filename in enumerate(matched_files, 1):
        if filename in processed_files:
            print(f"[{idx}/{len(matched_files)}] 跳过已处理: {filename}")
            continue
            
        if filename not in metadata_index:
            print(f"[{idx}/{len(matched_files)}] 警告: 在 JSON 元数据中找不到 {filename}，跳过。")
            continue
            
        print(f"[{idx}/{len(matched_files)}] 正在生成标签: {filename}")

        meta = metadata_index[filename]
        caption = meta.get("visual_fact_caption", "")
        target = meta.get("target_fallacy_scenario", "")
        category = meta.get("selected_typical_scenario", "")

        prompt = PROMPT_TEMPLATE.replace("{visual_fact_caption}", caption)
        prompt = prompt.replace("{target_fallacy_scenario}", target)
        prompt = prompt.replace("{selected_typical_scenario}", category)

        wait_time = 10
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                
                response_data = json.loads(response.text)

                final_entry = {
                    "filename": filename,
                    "pos_video_path": os.path.join(POS_FOLDER, filename),
                    "neg_video_path": os.path.join(NEG_FOLDER, filename),
                    "original_metadata": meta,
                    "llm_generated_labels": response_data
                }
                
                results.append(final_entry)

                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                    
                print("  -> 生成成功并存盘")
                break 
                
            except Exception as e:
                error_msg = str(e)
                if '503' in error_msg or '429' in error_msg:
                    print(f"  -> 接口拥挤，{wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, 60)
                else:
                    print(f"  -> 非网络异常，跳过此条: {error_msg}")
                    break
                    
        time.sleep(2)

if __name__ == "__main__":
    main()