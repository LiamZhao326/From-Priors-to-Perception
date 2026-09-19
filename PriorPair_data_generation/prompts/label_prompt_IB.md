### Input Information
`Visual Fact Caption`: {visual_fact_caption}
`Verified Positive O-A-V Annotation`: {verified_positive_oav_annotation}

================================

### Task Objective and Rules
You are an expert data annotator specializing in physical logic datasets. Based on the [Input Information] above, construct a pair of structured Observation--Attribution--Verdict annotations (Physically Plausible vs. Physically Implausible) for training Video-LLMs in a single pass.

Note: The negative sample in this category is generated programmatically by identifying the causal dividing point in the video and swapping the chronological order of the "Cause" chunk and the "Effect" chunk (Cause-Effect Swapping). You must strictly follow the three steps below:

**1. Observation**
- Rule: Generate a strictly objective visual description. Focus ONLY on **Subjects**, **Actions**, and **Environment**.
- Positive Sample Observation: Preserve the verified positive Observation and ensure that it remains fully supported by the `Visual Fact Caption`. Describe the events unfolding in their normal logical chronological order (Cause -> Effect).
- Negative Sample Observation: You must SIMULATE the visual effect of "Cause-Effect Swapping." Based on the `Visual Fact Caption`, identify the physical cause and its resulting effect. Describe a disjointed sequence where the macroscopic result (effect) happens first, and then the scene abruptly cuts to the physical action (cause) that was supposed to trigger it happening afterward. Maintain a strictly objective tone.
- Taboo: Absolutely NO hallucinations of new objects or interactions not present in the original caption.

**2. Attribution**
- Rule: Based on the content of the "[Observation]", strictly use the `PriorPair Category Dictionary` below to explain whether the phenomenon adheres to physical laws.
- Positive Sample Attribution: Preserve the verified positive Attribution and explain that the physical cause logically and chronologically precedes the effect.
- Negative Sample Attribution: Precisely point out that presenting the macroscopic effect before its physical cause violates the fundamental arrow of time and macroscopic causality, explicitly identifying it as a Causal Reversal.

**3. Verdict**
- Positive Sample Verdict: Use exactly `The video is physically valid.`
- Negative Sample Verdict: Use exactly `The video is physically invalid.`

================================

### PriorPair Category Dictionary
* **Core Principle:** Violation of "Linearity and Causality of Time." Events must unfold in a continuous chronological sequence where the physical cause always precedes the macroscopic effect.
* **Specific Fallacy Type: Causal Reversal**
* **Definition:** The logical chronological order of a causal event is swapped. Both the cause and the effect may play out normally in isolation, but the macroscopic effect is shown occurring *before* the physical action that causes it (e.g., first seeing a glass shatter on the floor, and then abruptly cutting to someone throwing the intact glass).

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
