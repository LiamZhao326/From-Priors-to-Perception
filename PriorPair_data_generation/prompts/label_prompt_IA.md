### Input Information
`Visual Fact Caption`: {Insert visual_fact_caption here}
`Verified Positive O-A-V Annotation`: {Insert verified_positive_oav_annotation here}

================================

### Task Objective and Rules
You are an expert data annotator specializing in physical logic datasets. Based on the [Input Information] above, construct a pair of structured Observation--Attribution--Verdict annotations (Physically Plausible vs. Physically Implausible) for training Video-LLMs in a single pass.

Note: The negative sample in this category is generated programmatically by chunking the original video frames into 3-5 groups and scrambling their chronological order (Frame Shuffle). You must strictly follow the three steps below:

**1. Observation**
- Rule: Generate a strictly objective visual description. Focus ONLY on **Subjects**, **Actions**, and **Environment**.
- Positive Sample Observation: Preserve the verified positive Observation and ensure that it remains fully supported by the `Visual Fact Caption`.
- Negative Sample Observation: You must SIMULATE the visual effect of "Frame Shuffling." Based on the `Visual Fact Caption`, describe a disjointed, non-linear sequence of events where the continuous action is broken. Describe the objects suddenly jumping back and forth in their state or position (e.g., an action starts, suddenly resets to the beginning, then jumps to the end, then back to the middle). Maintain a strictly objective tone.
- Taboo: Absolutely NO hallucinations of new objects or interactions not present in the original caption.

**2. Attribution**
- Rule: Based on the content of the "[Observation]", strictly use the `PriorPair Category Dictionary` below to explain whether the phenomenon adheres to physical laws.
- Positive Sample Attribution: Preserve the verified positive Attribution and explain that the sequence is consistent with the linearity of time and local temporal continuity of physical actions.
- Negative Sample Attribution: Precisely point out that the disjointed jumping of states violates the linearity of time and local or micro-temporal continuity of motion, explicitly identifying it as a Coherence Violation.

**3. Verdict**
- Positive Sample Verdict: Use exactly `The video is physically valid.`
- Negative Sample Verdict: Use exactly `The video is physically invalid.`

================================

### PriorPair Category Dictionary
* **Core Principle:** Violation of "Linearity and Causality of Time." Events must unfold in a continuous, irreversible chronological sequence unless physically reversed.
* **Specific Fallacy Type: Coherence Violation**
* **Definition:** The continuous progression of an action is broken at a micro-temporal level. The chronological order of frames is scrambled, resulting in objects or subjects abruptly teleporting between different stages of a single continuous action without fluid transition.

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
