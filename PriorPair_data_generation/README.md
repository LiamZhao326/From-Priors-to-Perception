# PriorPair Data Generation

This directory contains the code and final prompts for the main PriorPair
data-construction stages. The prompts in `prompts/` are used by default.

## Pipeline

1. `01_scene_construction/construct_scenarios.py` samples frames from each
   positive anchor and uses category-specific Gemini prompts to extract visual
   facts and, where required, define the target counterpart scenario and select
   the construction route.
2. `02_deterministic_generation/` provides segment shuffling, cause/effect
   swapping, continuous reversal, and localized SAM 2 + ProPainter editing.
3. `03_generative_generation/generate_video_prompts.py` converts scene
   metadata into direct video-generation prompts.
   `generate_with_kling.py` consumes those prompts and the source-video first
   frames to generate counterpart videos.
4. `04_label_generation/generate_oav_labels.py` constructs paired
   Observation--Attribution--Verdict annotations from scene metadata and
   expert-verified anchor annotations.

Generated visual facts, videos, and O--A--V annotations are reviewed and
corrected by experts before inclusion.

## Environment

Create the environment with:

```bash
conda env create -f environment/environment.yml
conda activate priorpair-data-generation
```

FFmpeg and ffprobe are required for video transformations and alignment.
Localized editing additionally requires SAM 2 and ProPainter; their setup is
described in `environment/external_dependencies.md`.

Set the API credentials required by the stages being run:

```bash
export GEMINI_API_KEY="..."
export KLING_ACCESS_KEY="..."
export KLING_SECRET_KEY="..."
```

Optional model overrides are listed in `environment/env.example`.

## Scene construction

```bash
python 01_scene_construction/construct_scenarios.py \
  --category II_A_Existence_Violation \
  --input-dir <DATA_ROOT>/II_A_Existence_Violation \
  --output <DATA_ROOT>/II_A_Existence_Violation_llmcaption.json
```

The bundled prompt directory is used by default. A different compatible
directory can be supplied with `--prompts-dir`.

## Deterministic construction

Coherence Violation divides each video into three to five segments and shuffles
their order:

```bash
python 02_deterministic_generation/shuffle_segments.py \
  --input-dir <DATA_ROOT>/I_A_Coherence_Violation \
  --output-dir <DATA_ROOT>/I_A_Coherence_Violation_negative \
  --seed 42 \
  --manifest <DATA_ROOT>/ia_shuffle_manifest.json
```

Causal Reversal exchanges cause and effect segments:

```bash
python 02_deterministic_generation/swap_causal_segments.py \
  --input-dir <DATA_ROOT>/I_B_Causal_Reversal \
  --output-dir <DATA_ROOT>/I_B_Causal_Reversal_negative \
  --annotations <DATA_ROOT>/causal_annotations.json
```

Continuous reversal is available for selected Dynamic Violation videos:

```bash
python 02_deterministic_generation/reverse_video.py \
  --selection <DATA_ROOT>/reversal_selection.json \
  --output-dir <DATA_ROOT>/III_A_Dynamic_Violation_negative
```

Localized SAM 2 propagation and ProPainter editing use expert-selected boxes
and frame ranges:

```bash
python 02_deterministic_generation/sam2_propainter_edit.py \
  --tasks <DATA_ROOT>/edit_tasks.json \
  --output-root <DATA_ROOT>/generated \
  --sam2-checkpoint <SAM2_CHECKPOINT> \
  --sam2-config <SAM2_CONFIG> \
  --propainter-dir <PROPAINTER_DIR>
```

## Generative construction

Generate direct video prompts:

```bash
python 03_generative_generation/generate_video_prompts.py \
  --metadata-dir <DATA_ROOT> \
  --output <DATA_ROOT>/video_generation_prompts.json
```

Submit the resulting prompts to Kling:

```bash
python 03_generative_generation/generate_with_kling.py \
  --prompts <DATA_ROOT>/video_generation_prompts.json \
  --output-root <DATA_ROOT>/generated \
  --results <DATA_ROOT>/kling_results.json
```

Each prompt record contains `video_path`, `category`, and
`generated_video_prompt`. The Kling script reads the first frame directly
from `video_path`.

## O--A--V annotation generation

Before annotation, organize expert-approved counterpart videos from
`<DATA_ROOT>/generated` into `<DATA_ROOT>/<category>_negative`, preserving
the corresponding anchor filenames (with `_rev` for continuously reversed
Dynamic Violation counterparts).

The verified-annotation input is a JSON object keyed by video filename. Each
value contains the expert-verified O--A--V annotation for the corresponding
anchor video. Label paths are written relative to the dataset root, with the
counterpart path targeting the aligned output directory used below.

```bash
python 04_label_generation/generate_oav_labels.py \
  --category II_A_Existence_Violation \
  --metadata <DATA_ROOT>/II_A_Existence_Violation_llmcaption.json \
  --verified-annotations <DATA_ROOT>/II_A_verified_annotations.json \
  --positive-dir <DATA_ROOT>/II_A_Existence_Violation \
  --negative-dir <DATA_ROOT>/II_A_Existence_Violation_negative \
  --output <DATA_ROOT>/II_A_Existence_Violation_labels.json
```

For continuously reversed Dynamic Violation records, add
`--type rev --negative-suffix _rev`.

## Video alignment

After label generation, align counterpart timestamps to their positive
anchors and create the `*_negative_aligned` paths recorded in the labels:

```bash
python 02_deterministic_generation/align_video.py --root <DATA_ROOT>
```

The script reads the eight category label files, creates the referenced
`*_negative_aligned` videos, and writes an alignment manifest.
