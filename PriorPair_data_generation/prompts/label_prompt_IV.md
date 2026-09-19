## IV-A: Near-Miss Annotation Prompt

### Input Information
`Original Visual Caption`: {visual_fact_caption}
`Verified Original O-A-V Annotation`: {verified_original_oav_annotation}
`Target Relative Scenario`: {target_fallacy_scenario}
`Original Interaction Judgement`: {interaction_judgement}
`Target Event`: {target_event}
`Scenario Type`: {selected_typical_scenario}

================================

### Task Objective and Rules
You are an expert data annotator specializing in visual event verification datasets. Based on the [Input Information] above, construct a pair of structured Three-Step labels (Positive Sample vs. Negative Sample) for training Video-LLMs to resist semantic-prior hallucinations.

Both samples are on equal footing and both are physically plausible. The task is NOT to judge whether either video conforms to physical laws. The task is to determine whether the specified `Target Event` visibly occurs.

The Positive Sample must always describe the video in which the `Target Event` occurred. The Negative Sample must always describe the video in which the same `Target Event` did not occur.

Use `Original Interaction Judgement` to assign the two inputs:
- If it is `Occurred`, the `Original Visual Caption` describes the Positive Sample and the `Target Relative Scenario` describes the Negative Sample.
- If it is `Did Not Occur`, the `Original Visual Caption` describes the Negative Sample and the `Target Relative Scenario` describes the Positive Sample.

**1. Observation**
- Rule: Generate a strictly objective visual description of each video.
- Focus on visible subjects, actions, environment, trajectories, spatial relationships, contact, separation, and action completion.
- Original-video Observation: Preserve the verified original Observation and ensure that it remains fully supported by the `Original Visual Caption`.
- Counterpart Observation: Use the original scene information and the `Target Relative Scenario` to describe the visible evidence for the opposite occurrence status of the same `Target Event`.
- Positive Sample Observation: Clearly describe the visible evidence showing that the `Target Event` occurs.
- Negative Sample Observation: Clearly describe the visible evidence showing that the same `Target Event` does not occur.
- Preserve relevant scene information from the input instead of reducing the Observation to a bare verdict.
- Do not include causal explanations, intentions, semantic expectations, physical-law discussion, category names, or annotation-related meta-language.
- Do not include timestamps.
- Do not invent exact distances that cannot be reliably observed. Describe a visible gap or remaining distance without unsupported numerical measurements.
- Do not hallucinate objects, actions, contacts, trajectories, or outcomes not supplied by the input information.

**2. Attribution**
- Rule: Explain why the visible evidence in the corresponding Observation supports whether the `Target Event` occurred.
- Original-video Attribution: Preserve the verified original Attribution while keeping it focused on the specified `Target Event` and visible evidence.
- Counterpart Attribution: Derive the explanation from the `Target Relative Scenario` and the counterpart Observation without importing facts unique to the original video.
- Positive Sample Attribution: Identify the visible contact, trajectory intersection, object transfer, or completed interaction that establishes the occurrence of the `Target Event`. For transfer or action completion, identify how the relevant spatial gap closes and the required contact establishes the specified initiating interaction.
- Negative Sample Attribution: Identify the visible spatial gap, trajectory passing beside the target, or halt before the relevant contact that prevents the `Target Event`. Do not describe an arbitrary unfinished action or a missing downstream consequence after established contact as a Near-Miss.
- Base the explanation strictly on observable evidence.
- Do not discuss whether the video is physically valid or invalid.
- Do not discuss semantic priors, expected scripts, adversarial construction, dataset design, or annotation decisions.
- Do not introduce hidden causes or unobservable mechanisms.

**3. Verdict**
- Positive Sample Verdict: Use exactly `The target event occurred.`
- Negative Sample Verdict: Use exactly `The target event did not occur.`

================================

### PriorPair Category Dictionary
* **Core Principle:**
Consistency between Statistical Priors and Visual Facts. These samples test whether a model faithfully determines if a specified visual interaction actually occurs, rather than completing a statistically familiar event script from semantic expectation. Both members of the pair are physically plausible.

* **Specific Adversarial Type: Near-Miss Interaction**
* **Definition:** A potential interaction is visually approached, but a visible spatial gap or a halt before the relevant contact prevents the initiating interaction in the non-occurring video. The paired video closes the gap and establishes the same target interaction. Contact, trajectory intersection, object transfer, and action completion are judged as initiating interactions; transfer and completion must depend on closing the relevant gap and establishing contact, rather than on arbitrary action progress or a later consequence after contact.

* **Typical Scenarios:**
    1. **Trajectory Miss:** An object or person moves toward a target but passes beside it with a visible spatial gap, preventing the target contact. The paired sample shows the gap closing and the target interaction occurring.
    2. **Abrupt Halt:** An approach stops before the relevant contact while a visible spatial gap remains. The paired sample shows the approach continuing until the gap closes and the specified interaction is completed.

================================

### Output Requirements
Strictly return the output in JSON format, containing a nested structure for the Positive and Negative Samples.

The `sft_response` must exactly concatenate the preceding `observation`, `causal_attribution`, and `verdict` fields in the required order. Do not paraphrase, shorten, expand, or add any other text.

{
  "positive_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "The target event occurred.",
    "sft_response": "**Observation**: ...\n\n**Attribution**: ...\n\n**Verdict**: The target event occurred."
  },
  "negative_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "The target event did not occur.",
    "sft_response": "**Observation**: ...\n\n**Attribution**: ...\n\n**Verdict**: The target event did not occur."
  }
}


## IV-B: Consequence Arrest Annotation Prompt

### Input Information
`Original Visual Caption`: {visual_fact_caption}
`Verified Original O-A-V Annotation`: {verified_original_oav_annotation}
`Target Relative Scenario`: {target_fallacy_scenario}
`Original Consequence Judgement`: {consequence_judgement}
`Target Event`: {target_event}
`Scenario Type`: {selected_typical_scenario}

================================

### Task Objective and Rules
You are an expert data annotator specializing in visual event verification datasets. Based on the [Input Information] above, construct a pair of structured Three-Step labels (Positive Sample vs. Negative Sample) for training Video-LLMs to resist semantic-prior hallucinations.

Both samples are on equal footing and both are physically plausible. The task is NOT to judge whether either video conforms to physical laws. The task is to determine whether the specified `Target Event`, normally a consequence or observable state change, visibly occurs.

The Positive Sample must always describe the video in which the `Target Event` occurred. The Negative Sample must always describe the video in which the same `Target Event` did not occur.

Use `Original Consequence Judgement` to assign the two inputs:
- If it is `Occurred`, the `Original Visual Caption` describes the Positive Sample and the `Target Relative Scenario` describes the Negative Sample.
- If it is `Did Not Occur`, the `Original Visual Caption` describes the Negative Sample and the `Target Relative Scenario` describes the Positive Sample.

**1. Observation**
- Rule: Generate a strictly objective visual description of each video.
- Focus on visible subjects, actions, environment, contact, applied motion, deformation, displacement, material release, breakage, ignition, or other state changes relevant to the `Target Event`.
- Original-video Observation: Preserve the verified original Observation and ensure that it remains fully supported by the `Original Visual Caption`.
- Counterpart Observation: Use the original scene information and the `Target Relative Scenario` to describe the visible evidence for the opposite occurrence status of the same `Target Event`.
- Positive Sample Observation: Clearly describe the visible evidence showing that the `Target Event` occurs.
- Negative Sample Observation: Clearly describe the visible evidence showing that the same `Target Event` does not occur.
- Preserve relevant scene and action information from the input instead of reducing the Observation to a bare verdict.
- Do not include causal explanations, intentions, semantic expectations, physical-law discussion, category names, or annotation-related meta-language.
- Do not include timestamps.
- Do not hallucinate objects, actions, impacts, consequences, or state changes not supplied by the input information.

**2. Attribution**
- Rule: Explain why the visible initiating action or contact and resulting state support whether the `Target Event` occurred. Establish that the initiating action or contact has occurred in both samples; do not substitute its occurrence for the target consequence, or treat a spatial miss or pre-contact halt as an absent downstream consequence.
- Original-video Attribution: Preserve the verified original Attribution while keeping it focused on the specified `Target Event` and visible evidence.
- Counterpart Attribution: Derive the explanation from the `Target Relative Scenario` and the counterpart Observation without importing facts unique to the original video.
- Positive Sample Attribution: Explain the most reasonable direct causal relationship between the visible initiating action and the observed consequence or state change.
- Negative Sample Attribution: Explain the most reasonable ordinary reason why the visible initiating action does not produce the target consequence, such as insufficient impact, inadequate displacement, a shallow angle, limited deformation, or failure to cross the relevant threshold.
- The explanation must remain consistent with the visible action and outcome.
- Do not discuss whether the video is physically valid or invalid.
- Do not discuss semantic priors, expected scripts, adversarial construction, dataset design, or annotation decisions.
- Do not invent an unnecessary hidden mechanism when the visible action already provides a sufficient ordinary explanation.

**3. Verdict**
- Positive Sample Verdict: Use exactly `The target event occurred.`
- Negative Sample Verdict: Use exactly `The target event did not occur.`

================================

### PriorPair Category Dictionary
* **Core Principle:**
Consistency between Statistical Priors and Visual Facts. These samples test whether a model faithfully determines if a specified consequence or state change actually occurs, rather than assuming the familiar outcome of an action from semantic expectation. Both members of the pair are physically plausible.

* **Specific Adversarial Type: Consequence Arrest**
* **Definition:** An initiating action or contact has visibly occurred, but the specified consequence or state change occurs in one video and does not occur in the paired video depending on whether the relevant physical condition or threshold is reached. A preparatory pose alone is insufficient. The task is to judge the specified consequence after the initiating action or contact, unlike Near-Miss, where a spatial gap or a halt before contact prevents the initiating interaction itself.

* **Typical Scenarios:**
    1. **Insufficient Impact:** Contact or impact occurs, but its visible magnitude may be too weak to produce the target displacement, damage, collapse, or other consequence. The paired sample shows sufficient impact followed by the target consequence.
    2. **Threshold Failure:** An action is performed, but its visible angle, displacement, pressure, duration, or magnitude may not cross the threshold required for the target state change. The paired sample crosses the threshold and produces the target event.

================================

### Output Requirements
Strictly return the output in JSON format, containing a nested structure for the Positive and Negative Samples.

The `sft_response` must exactly concatenate the preceding `observation`, `causal_attribution`, and `verdict` fields in the required order. Do not paraphrase, shorten, expand, or add any other text.

{
  "positive_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "The target event occurred.",
    "sft_response": "**Observation**: ...\n\n**Attribution**: ...\n\n**Verdict**: The target event occurred."
  },
  "negative_sample": {
    "observation": "...",
    "causal_attribution": "...",
    "verdict": "The target event did not occur.",
    "sft_response": "**Observation**: ...\n\n**Attribution**: ...\n\n**Verdict**: The target event did not occur."
  }
}
