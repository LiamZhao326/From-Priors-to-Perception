### Input Information
`Visual Fact Caption`: {visual_fact_caption}
`Verified Positive O-A-V Annotation`: {verified_positive_oav_annotation}

================================

### Task Objective and Rules
You are an expert data annotator specializing in physical logic datasets. Based on the [Input Information] above, construct a pair of structured Observation--Attribution--Verdict annotations (Physically Plausible vs. Physically Implausible) for training Video-LLMs in a single pass.

Note: The negative sample in this category is generated programmatically by completely reversing the chronological order of the original video frames (Continuous Rewind / Time Reversal). You must strictly follow the three steps below:

**1. Observation**
- Rule: Generate a strictly objective visual description. Focus ONLY on **Subjects**, **Actions**, and **Environment**.
- Positive Sample Observation: Preserve the verified positive Observation and ensure that it remains fully supported by the `Visual Fact Caption`. Describe the physical event progressing normally.
- Negative Sample Observation: You must SIMULATE the visual effect of "Continuous Rewind." Based on the `Visual Fact Caption`, describe the exact sequence of events happening completely in reverse. Describe the visually continuous but physically impossible phenomena, such as objects falling upwards (anti-gravity), dispersed liquids gathering and flowing upwards into containers, or shattered fragments spontaneously reassembling into a pristine whole (anti-entropy). Maintain a strictly objective tone.
- Taboo: Absolutely NO hallucinations of new objects or interactions not present in the original caption.

**2. Attribution**
- Rule: Based on the content of the "[Observation]", strictly use the `PriorPair Category Dictionary` below to explain whether the phenomenon adheres to physical laws.
- Positive Sample Attribution: Preserve the verified positive Attribution and explain how the visible sequence is consistent with the relevant mechanical or thermodynamic constraints.
- Negative Sample Attribution: Precisely identify the reversed motion or state evolution that lacks the visible force, mechanism, or energy input required to produce it, such as fragments reassembling, spilled liquid gathering into a container, or an unsupported object moving upward. Classify the resulting anti-gravity or anti-entropy phenomenon as a Dynamic Violation.

**3. Verdict**
- Positive Sample Verdict: Use exactly `The video is physically valid.`
- Negative Sample Verdict: Use exactly `The video is physically invalid.`

================================

### PriorPair Category Dictionary
* **Core Principle:** Violation of "Causal Consistency of Physical Interactions and Thermodynamics." The motion, trajectory, and entropy of objects must strictly adhere to classical mechanics and the irreversible arrow of time.
* **Specific Fallacy Type: Dynamic Violation (Anti-Gravity / Anti-Entropy)**
* **Definition:** A continuous physical process is displayed in exact reverse, producing motion or state evolution that lacks the force, mechanism, or energy input required under the visible conditions. Examples include unsupported upward motion, shattered fragments spontaneously reassembling, or dispersed liquid gathering and returning to a container without external manipulation.

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
