### Input Information
`Visual Fact Caption`: {Insert visual_fact_caption here}
`Verified Positive O-A-V Annotation`: {Insert verified_positive_oav_annotation here}
`Target Fallacy Scenario`: {Insert target_fallacy_scenario here}
`Fallacy Category`: {Insert selected_typical_scenario here}
`Category Definition`: {Insert category_definition here}

================================

### Task Objective and Rules
You are an expert data annotator specializing in physical logic datasets. Based on the [Input Information] above, construct a pair of structured Observation--Attribution--Verdict annotations (Physically Plausible vs. Physically Implausible) for training Video-LLMs in a single pass. You must strictly follow the three steps below:

**1. Observation**
- Rule: Generate a strictly objective visual description. Accurately describe the factual phenomena occurring in the video. Focus ONLY on **Subjects**, **Actions**, and **Environment**.
- Positive Sample Observation: Preserve the verified positive Observation and ensure that it remains fully supported by the `Visual Fact Caption`. Do not add any hallucinated details.
- Negative Sample Observation: Use the verified positive description and the `Visual Fact Caption` as the background setting, and integrate the abnormal actions from the `Target Fallacy Scenario` into a coherent, objective description.
- Taboo: Absolutely NO hallucinations, and NO causal speculations or inferences.

**2. Attribution**
- Rule: Based on the content of the "[Observation]", strictly use the `PriorPair Category Dictionary` below to explain whether the phenomenon adheres to physical laws.
- Positive Sample Attribution: Preserve the verified positive Attribution and state only the relevant physical constraints supported by the visible events.
- Negative Sample Attribution: Precisely point out which physical law or causal logic the `Target Fallacy Scenario` violates (expand this based on the provided `Fallacy Category`).

**3. Verdict**
- Positive Sample Verdict: Use exactly `The video is physically valid.`
- Negative Sample Verdict: Use exactly `The video is physically invalid.`

================================

### PriorPair Category Dictionary
{Insert category_definition here}

================================

### Output Requirements
Strictly return the output in JSON format, containing a nested structure for both positive and negative samples. The `sft_response` must exactly concatenate the preceding `observation`, `causal_attribution`, and `verdict` fields in the required order. Do not paraphrase, shorten, expand, or add any other text.
{
  "positive_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "The video is physically valid.",
    "sft_response": "**Observation**: ...\n\n**Attribution**: ...\n\n**Verdict**: The video is physically valid."
  },
  "negative_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "The video is physically invalid.",
    "sft_response": "**Observation**: ...\n\n**Attribution**: ...\n\n**Verdict**: The video is physically invalid."
  }
}
