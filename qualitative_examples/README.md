# PriorPair Qualitative Examples

This directory contains one selected Baseline-to-PhyAR improvement pair for each of the eight PriorPair categories.

## Selection rule

The examples were selected from pairs for which the Baseline achieved PCA = 0 and PhyAR achieved PCA = 1, prioritizing higher PhyAR RAS scores and clearer reasoning improvements.

Each category directory contains:

- the positive and negative counterpart videos in `videos/`;
- an overview contact sheet with the positive row followed by the negative row;
- the exact Ground Truth, Baseline output, and PhyAR output in `example.md`.

## Selected examples

| Category | Samples | Baseline PCA | PhyAR PCA | Baseline pair RAS | PhyAR pair RAS | Note |
|---|---:|---:|---:|---:|---:|---|
| Coherence Violation | 2/3 | 0 | 1 | 1.0 | 5.0 | Close-up action with an easily visible scrambled temporal order. PhyAR receives 5/5 reasoning scores. |
| Causal Reversal | 34/35 | 0 | 1 | 1.0 | 5.0 | High-resolution mug-breaking sequence with a clear effect-before-cause reversal. PhyAR receives 5/5 reasoning scores. |
| Existence Violation | 74/75 | 0 | 1 | 2.0 | 4.0 | The negative counterpart shows liquid rising and swirling without visible input. PhyAR identifies the spontaneous rise and correctly classifies both counterparts. |
| Identity Violation | 98/99 | 0 | 1 | 1.0 | 5.0 | High-resolution transformation from a rubber duck into a live duckling. PhyAR receives 5/5 reasoning scores. |
| Dynamic Violation | 122/123 | 0 | 1 | 2.0 | 4.5 | The car visibly lifts from the track without a supporting mechanism. PhyAR identifies the unsupported airborne motion and correctly classifies both counterparts. |
| Constraint Violation | 220/221 | 0 | 1 | 1.0 | 5.0 | The magnet is moved close to the compass, but only the positive counterpart shows the expected needle rotation. PhyAR identifies the missing magnetic response. |
| Near-Miss | 264/265 | 0 | 1 | 1.5 | 5.0 | Clear contrast between grasping/opening the handle and stopping with a visible gap. PhyAR receives 5/5 reasoning scores. |
| Consequence Arrest | 320/321 | 0 | 1 | 2.0 | 4.5 | Clear close-up contrast between successful ignition and repeated lighter operation without a flame. Baseline hallucinates a flame in the negative counterpart, whereas PhyAR identifies its absence. |

The overview contact sheets provide quick visual references for the paired videos.
