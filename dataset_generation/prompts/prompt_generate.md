### Dynamic Inputs
* **Visual Frames:** [Input: 16 extracted frames from the original video]
* **Camera & Style:** [Input: Extracted visual style and camera movement]
* **Original Caption:** [Input: Strict objective description of the original action]
* **Target Fallacy Scenario:** [Input: Detailed description of the target physics violation]

### Core Task & Directives
Based on the `Visual Frames`, `Camera & Style`, and `Original Caption` of the source video, you must write a highly precise English prompt for a video generation model to guide it in generating the given `Target Fallacy Scenario`, creating a modified new version of the video.

* **Establish Baseline & Execute Replacement:** The prompt must use the `Original Caption` as the realistic baseline for the environment and subjects, but you must **completely replace** the logical, natural outcome of the original video with the counter-intuitive or absurd events described in the `Target Fallacy Scenario`.
* **Absolute Forcing Principle:** The prompt must force the video model to prioritize executing the `Target Fallacy Scenario`. You are **strictly prohibited** from attempting to "correct" or rationalize these anti-physical phenomena. You must command these impossible events directly as objective visual facts.

### Core Prompting Rules
* **Structural Front-loading:** Always place the camera movement, visual style, or perspective (e.g., "High quality, Low angle shot") at the very beginning of the prompt to set the visual tone.
* **Strong Explicit Negation & Contrast:** You MUST use strong negative words (e.g., capitalized "NOT", "completely misses", "without") or extreme contrasting modifiers (e.g., "tiny pebble" vs. "towering geyser") to definitively break common sense and semantic priors.
* **Temporal Slicing Expression:** For sudden state changes or interventions, use clear temporal anchors to define the sequence of events (e.g., "Upon impact", "Phase 1...", "Trigger:", "Phase 2...").
* **Summary Qualitative Label:** Conclude the entire prompt with an extremely brief declarative sentence that definitively labels the physical anomaly (e.g., "A clean miss.", "The material behaves like rubber.").

### Few-Shot Examples
[Example 1] A bowling ball rolls fast towards the single white pin. However, it misses the target completely. The ball rolls past the pin on the right side without making any contact. The pin remains standing perfectly still. A clean miss.

[Example 2] A pink vase falls and hits the rusty floor. Upon impact, the object exhibits soft body physics. It drastically deforms, squashes, and compresses, then bounces back with elasticity. The material behaves like rubber. High-speed camera, slow motion, detailed texture of the deformation.

[Example 3] High quality, realistic footage. Dynamic tracking shot matching the original camera movement. Phase 1: The original dog is walking/running on the grass. It maintains its original posture and movement rhythm. Trigger: Mid-stride, a surreal transformation occurs. Phase 2: The animal smoothly morphs into a ginger tabby cat. The cat occupies the exact same position and path as the dog. The cat continues the walking motion seamlessly. Environment Constraint: The grass texture, lighting, and background scenery remain identical to the original footage. Only the animal species changes.

[Example 4] High quality, realistic footage starting from the reference image. The metal knife blade presses down firmly into the orange mandarin. Upon cutting pressure, the orange does NOT split into two halves. Instead, it instantly fractures and crumbles into multiple (more than 6) small, separate orange segments and irregular peel pieces that scatter slightly across the textured glass cutting board. Juice sprays slightly. The knife continues downwards to the board.

[Example 5] Low angle shot. A tiny pebble drops vertically into the center of the calm lake. The moment it touches the surface, the water erupts violently. A massive, towering geyser of water shoots fifty meters high into the air immediately upon contact. The impact site generates a giant explosion of white foam and heavy waves radiating outwards. The scene demonstrates extreme kinetic amplification.

[Example 6] Low angle shot. A massive grey boulder falls from the sky and hits the water surface. The water surface remains completely undisturbed and flat.

### Output Constraint
* Return a single valid JSON object. Do NOT use markdown code blocks (like ```json).
* The JSON must contain exactly one key: "generated_video_prompt".

{
  "generated_video_prompt": "String. The final constructed English prompt."
}