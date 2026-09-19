# PriorPair Task Prompts

## I–III: Physical-validity verification

```text
Task: Physical-validity verification

Determine whether the events depicted in the video are physically valid or violate real-world physical, temporal, causal, thermodynamic, or spatial constraints.

Analyze the video using the following three fields:

Observation: Objectively describe the chronological sequence of events and the decisive visual evidence explicitly present in the video.

Attribution: Based on the Observation, identify the relevant physical, temporal, causal, thermodynamic, or spatial constraints, and explain whether the observed events comply with or violate them.

Verdict: Conclude with exactly either "The video is physically valid." or "The video is physically invalid."
```

## IV: Target-event occurrence verification

```text
Task: Target-event occurrence verification

Determine whether the following target event occurs in the video.

Target event: {target_event}

Analyze the video using the following three fields:

Observation: Objectively describe the chronological visual evidence relevant to whether the target event occurs.

Attribution: Based on the Observation, explain whether the visual evidence supports that the target event occurred or did not occur. Ground the explanation in the visible evidence rather than the expected course of events.

Verdict: Conclude with exactly either "The target event occurred." or "The target event did not occur."
```

## Additional format reminder for format-sensitive baselines

For Flash-VStream, Video-ChatGPT, Video-LLaVA, VideoLLaMA2, and VideoLLaMA3 Base, the following identical reminder is appended to the applicable task prompt:

```text
Important output-format requirement: Your response must include all three requested fields—Observation, Attribution, and Verdict—in that order. Do not omit Observation or Attribution, and do not answer with only the Verdict.
```
