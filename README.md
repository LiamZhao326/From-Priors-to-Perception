# Supplementary Material for *From Priors to Perception: Grounding Video-LLMs in Physical Reality*

This supplementary material accompanies the paper *From Priors to Perception:
Grounding Video-LLMs in Physical Reality*. It includes representative PriorPair
videos and annotations, complete Base-to-PhyAR qualitative examples, and code
for data construction, training, evaluation, and ablation studies.

## Package Contents

| Directory | Contents |
|---|---|
| `8_PriorPair_cases/` | Eight representative PriorPair video demonstrations and their complete Observation--Attribution--Verdict (O--A--V) annotations. |
| `qualitative_examples/` | One Base-to-PhyAR improvement example for each of the eight PriorPair categories, including paired videos, Ground Truth, and complete model responses. |
| `PriorPair_data_generation/` | Reference implementation of the dataset-construction pipeline, together with the final construction and annotation prompts. |
| `training/` | Dataset preparation and balanced QLoRA training scripts for the InternVL2.5-8B and VideoLLaMA3-7B PhyAR models. |
| `evaluations_and_ablations/` | Main-benchmark inference and metrics, component and stream ablations, mechanism analyses, and external generalization evaluation. |

## Representative PriorPair Cases

`8_PriorPair_cases/case.pdf` presents the complete O--A--V annotations for
eight representative PriorPair examples. The accompanying videos correspond
to the demonstrations in the PDF as follows:

| File | Category |
|---|---|
| `demo_1.mp4` | Coherence Violation |
| `demo_2.mp4` | Causal Reversal |
| `demo_3.mp4` | Existence Violation |
| `demo_4.mp4` | Identity Violation |
| `demo_5.mp4` | Dynamic Violation |
| `demo_6.mp4` | Constraint Violation |
| `demo_7.mp4` | Near-Miss |
| `demo_8.mp4` | Consequence Arrest |

## Base-to-PhyAR Qualitative Examples

`qualitative_examples/` contains the eight qualitative examples discussed in
the appendix. Each category directory provides the positive and negative
counterpart videos, an overview image, and an `example.md` file containing the
Ground Truth and the complete Baseline and PhyAR responses. The directory-level
`README.md` summarizes the selected examples, while `manifest.json` records
their sample indices and evaluation results.

## Code Organization

The code is organized according to the experimental workflow:

1. `PriorPair_data_generation/` covers scene construction, deterministic and
   generative counterpart construction, O--A--V annotation generation, and
   video alignment. Its `prompts/` directory contains the final prompts used
   throughout data construction and evaluation.
2. `training/` prepares the category-and-scenario-stratified data split and
   trains the two PhyAR backbones with paired, balanced QLoRA.
3. `evaluations_and_ablations/` provides PriorPair inference and PCA, RPB, and
   RAS computation, together with the reported ablations, paired-gradient
   analysis, Semantic Prior Dominance diagnosis, and IntPhys 2 and Video-MME
   evaluations.

Each code directory includes a README describing its contents and how to use
the code.
