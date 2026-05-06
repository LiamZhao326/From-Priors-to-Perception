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