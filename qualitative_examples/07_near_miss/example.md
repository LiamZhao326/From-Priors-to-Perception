# Near-Miss

## Selection metadata

- Category: `IV_A_Near_Miss`
- Sample indices: positive `264`, negative `265`
- Positive video: `videos/positive.mp4`
- Negative video: `videos/negative.mp4`
- Baseline: PCA = 0, sample accuracy = 0/0, reasoning scores = 2/1, pair RAS = 1.5
- PhyAR: PCA = 1, sample accuracy = 1/1, reasoning scores = 5/5, pair RAS = 5.0
- Selection note: Clear contrast between grasping/opening the handle and stopping with a visible gap. PhyAR receives 5/5 reasoning scores.

![Pair overview](overview.jpg)

The overview places the six positive-counterpart frames first and the six negative-counterpart frames second.

## Positive counterpart

### Ground Truth

**Observation**: The hand reaches the handle, grasps it, and opens the cabinet.

**Attribution**: The objects meet at the same location with no visible gap, establishing direct contact.

**Verdict**: The target event occurred.

### Baseline output

Observation: The video shows a hand reaching towards and grasping a door handle. The hand is seen in contact with the handle, indicating an attempt to open the door. The handle is part of a door, not a cabinet, and the action is directed towards the door, not a cabinet.

Attribution: The visual evidence shows the hand making contact with the door handle, not a cabinet handle. Therefore, the target event of the hand contacting a cabinet handle did not occur.

Verdict: The target event did not occur.

### PhyAR output

**Observation**: A hand grasps and pulls a cabinet handle, opening the cabinet door.

**Attribution**: The objects meet at the same location with no visible gap, establishing direct contact.

**Verdict**: The target event occurred.

## Negative counterpart

### Ground Truth

**Observation**: A hand approaches a cabinet handle but stops before touching it, and the cabinet remains closed.

**Attribution**: A visible gap persists at the closest approach, so the objects never make contact.

**Verdict**: The target event did not occur.

### Baseline output

Observation: In the video, a hand is seen reaching towards and making contact with a cabinet handle. The hand is positioned on the handle, and the fingers are wrapped around it, indicating a firm grip.

Attribution: The visual evidence clearly shows the hand making contact with the cabinet handle. The hand's movement towards the handle and the subsequent firm grip on it confirm that the target event of the hand contacting the cabinet handle has occurred.

Verdict: The target event occurred.

### PhyAR output

**Observation**: A hand approaches a cabinet handle but stops before touching it, and the cabinet remains closed.

**Attribution**: A visible gap persists at the closest approach, so the objects never make contact.

**Verdict**: The target event did not occur.
