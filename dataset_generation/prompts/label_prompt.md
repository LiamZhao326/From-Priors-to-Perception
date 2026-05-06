### Input Information
`Visual Fact Caption`: {Insert visual_fact_caption here}
`Target Fallacy Scenario`: {Insert target_fallacy_scenario here}
`Fallacy Category`: {Insert selected_typical_scenario here}

================================

### Task Objective and Rules
You are an expert data annotator specializing in physical logic datasets. Based on the [Input Information] above, construct a pair of structured Three-Step CoT (Chain of Thought) data (Physically Plausible vs. Physically Implausible) for training Video-LLMs in a single pass. You must strictly follow the three steps below to generate the logical deduction:

**1. Observation**
- Rule: Generate a strictly objective visual description. Accurately describe the factual phenomena occurring in the video. Focus ONLY on **Subjects**, **Actions**, and **Environment**.
- Positive Sample Observation: Base this entirely on the `Visual Fact Caption`. Do not add any hallucinated details.
- Negative Sample Observation: Use the `Visual Fact Caption` as the background setting, and seamlessly integrate the abnormal actions from the `Target Fallacy Scenario` into a coherent, objective description.
- Taboo: Absolutely NO hallucinations, and NO causal speculations or inferences.

**2. Attribution**
- Rule: Based on the content of the "[Observation]", strictly use the `PACC Category Dictionary` below to explain whether the phenomenon adheres to physical laws.
- Positive Sample Attribution: Explain which fundamental physical laws the `Visual Fact Caption` perfectly adheres to.
- Negative Sample Attribution: Precisely point out which physical law or causal logic the `Target Fallacy Scenario` violates (expand this based on the provided `Fallacy Category`).

**3. Verdict**
- Positive Sample Verdict: Clearly summarize in one sentence that the video is real and perfectly conforms to real-world physical logic.
- Negative Sample Verdict: Clearly summarize in one sentence that the video is forged and contains physical fallacies.

================================

### PACC Category Dictionary
... (Insert definitions here)

================================

### Output Requirements
Strictly return the output in JSON format, containing a nested structure for both positive and negative samples. The `sft_response` must combine the previous three steps into a coherent and complete paragraph:
{
  "positive_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "...",
    "sft_response": "**Observation**：...\n **Attribution**：...\n **Verdict**：..."
  },
  "negative_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "...",
    "sft_response": "**Observation**：...\n **Attribution**：...\n **Verdict**：..."
  }
}