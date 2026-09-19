### Dynamic Inputs
* **Visual Frames:** [Input: 16 extracted frames from the original video]
* **Camera & Style:** [Input: Extracted visual style and camera movement]
* **Original Caption:** [Input: Strict objective description of the original action]
* **IV Category:** [Input: IV-A Near-Miss Interaction or IV-B Consequence Arrest]
* **Target Event:** [Input: Short, atomic, directly observable event shared by both videos]
* **Original Event Judgement:** [Input: Use `interaction_judgement` for IV-A or `consequence_judgement` for IV-B; the value must be exactly `Occurred` or `Did Not Occur`]
* **Target Relative Scenario:** [Input: Direct visual description of the physically plausible counterpart in which the Target Event has the opposite occurrence status]

### Core Task & Directives
Based on the `Visual Frames`, `Camera & Style`, `Original Caption`, and event information, write a precise English prompt for a video generation model to create the matched counterpart described by the `Target Relative Scenario`.

Both the original video and the generated counterpart must remain physically plausible. The generated counterpart must preserve the same `Target Event` while reversing only its occurrence status:
* If the `Original Event Judgement` is `Occurred`, the generated video must show that the Target Event does not occur.
* If the `Original Event Judgement` is `Did Not Occur`, the generated video must show that the Target Event occurs.

The final video-generation prompt must directly describe the desired visible scene and actions. Do not mention labels, categories, judgements, counterparts, modifications, source frames, semantic priors, physical validity, or dataset construction.

### Core Prompting Rules
* **Preserve the Scene:** Preserve the original subjects, object identities, environment, camera position, visual style, lighting, and temporal continuity unless a small spatial or action change is necessary to establish the opposite Target Event status.
* **Direct Event Description:** Describe the desired action and outcome as ordinary visible events. Do not use meta-language such as "the modified video," "the target event," "instead of the original," or "opposite outcome."
* **Single Event Definition:** Keep the generated action focused on the supplied Target Event. Do not replace it with an earlier prerequisite or a later unrelated consequence.
* **Physically Plausible Cause and Outcome:** Any contact, miss, halt, weak response, threshold failure, or successful consequence must follow naturally from the visible trajectory, force, angle, distance, timing, or material behavior shown in the generated video.
* **IV-A Near Miss:**
    * For a non-occurring Target Event, directly describe a trajectory passing beside the target with a visible spatial gap, or an approach that stops before the relevant contact while the gap remains.
    * For an occurring Target Event, directly describe the gap closing and the trajectory intersection, contact, transfer, or completed initiating interaction taking place. Transfer and action completion must depend on closing the relevant gap and establishing contact; do not use arbitrary unfinished actions or missing downstream consequences after contact to create a Near-Miss.
    * Do not require an object to ignore a visible collision or violate inertia merely to create a miss.
* **IV-B Consequence Arrest:**
    * For a non-occurring Target Event, visibly show the initiating action or contact actually occurring, while an ordinary cause such as a light impact, shallow angle, limited displacement, low pressure, or failure to cross a threshold prevents the specified consequence. A preparatory pose alone, a spatial miss, or a halt before the relevant initiating contact must not replace this action-to-consequence distinction.
    * For an occurring Target Event, show the initiating action reaching sufficient magnitude and the specified consequence visibly taking place.
    * Do not preserve a strong cause while demanding an impossible absence of its normal physical effect.
* **No Answer Leakage or Explanatory Text:** The prompt may describe occurrence or non-occurrence visually, but must not contain the words `Occurred`, `Did Not Occur`, `positive sample`, `negative sample`, `Near Miss`, `Consequence Arrest`, `Insufficient Impact`, or `Threshold Failure`.
* **No Unrequested Additions:** Do not introduce duplicate objects, scene cuts, new actors, supernatural forces, impossible material behavior, or unrelated actions unless explicitly required by the Target Relative Scenario.
* **Continuity:** Keep the video in one coherent shot when compatible with the source. Maintain stable identities, scale, background, and camera behavior throughout the decisive action.

### Few-Shot Examples
[Example 1: IV-A, generating a non-occurring interaction]
A golfer completes one controlled swing through the sand while the club head passes through empty space beside the stationary white golf ball. A clearly visible gap remains between the club and the ball throughout the swing. The same ball stays in its original position without moving. Realistic daylight footage with the original vertical framing and camera position.

[Example 2: IV-A, generating an occurring interaction]
A blue curling stone slides diagonally across the ice and directly contacts the upper-left red curling stone. The gap closes completely at impact, and the red stone begins moving from its original position. The other red stone remains untouched. Preserve the overhead camera and realistic curling-rink lighting.

[Example 3: IV-B, generating a non-occurring consequence]
A hand tilts the cup only slightly above the receiving container. The liquid shifts inside the cup but never reaches or crosses the rim, so no liquid leaves the cup. The hand then returns the cup upright. Preserve the original kitchen setting, camera position, and lighting.

[Example 4: IV-B, generating an occurring consequence]
A person lands barefoot on the center of the trampoline. The mat visibly depresses and then rebounds, lifting the person clearly away from the surface and upward into the air. Preserve the outdoor garden, clothing, camera angle, and natural motion.

### Output Constraint
* Return a single valid JSON object. Do NOT use markdown code blocks (like ```json).
* The JSON must contain exactly one key: `generated_video_prompt`.

{
  "generated_video_prompt": "String. The final direct English prompt for generating the physically plausible counterpart video."
}
