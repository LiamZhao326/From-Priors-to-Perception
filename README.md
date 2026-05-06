# PACC Supplementary Material

This supplementary material contains the following components to support the reproducibility of our paper:

## Directory Structure

```
supplementary/
├── README.md
├── 8_case_studies/                 # Eight representative case studies
├── dataset_generation/             # Scripts for PACC dataset construction
├── training/                       # Training code for PhyAR
└── baseline_evaluation/            # Evaluation scripts for all baselines
```

---

## 1. 8_case_studies

This folder contains **eight representative adversarial video pairs** from the PACC dataset, along with their complete VARC annotations.

- Each case includes:
  - A stitched video (positive + negative sample side-by-side) for easy visual comparison
  - Detailed VARC Chain-of-Thought labels (Observation → Attribution → Verdict) in PDF format

These examples cover all eight fallacy dimensions and serve as concrete demonstrations of the dataset quality and annotation standards.

---

## 2. dataset_generation

This folder contains all scripts used to construct the PACC dataset:

| Script                  | Description |
|-------------------------|-----------|
| `llm_process.py`        | LLM-assisted video annotation and target fallacy generation |
| `manual_cv_edit.py`     | Manual editing of negative samples using computer vision tools |
| `render_pipeline.py`    | Rendering pipeline for edited videos |
| `kling_generate.py`     | Prompt generation and negative sample synthesis using Kling |
| `align_video.py`        | Alignment of positive and negative video durations |
| `label_generate.py`     | Final VARC label generation |

- The `prompts/` subfolder contains all prompt templates used in the pipeline.

**Note:** All file paths in the scripts must be updated to your local environment before running.

---

## 3. training

This folder contains the training script for our proposed model **PhyAR** (Physics-Anchored Reasoner). The single script includes:

- LoRA fine-tuning on the PACC training set
- Paired data loading and gradient suppression
- VARC prompt formatting

**Note:** Please replace all paths with your actual dataset and output directories.

---

## 4. baseline_evaluation

This folder provides evaluation scripts for all baselines reported in the paper:

- `llm_evaluate.py`: Calls LLM (as judge) to generate Reasoning Alignment Score (RAS)
- `get_metrics.py`: Computes all evaluation metrics (PCA, RAS, etc.) and outputs final results

Scripts support evaluation of both proprietary models (GPT-4o, Gemini 2.5 Flash) and open-source models.

**Note:** All paths (model checkpoints, video directories, output folders) need to be configured according to your local setup.

---

## Important Notes

- All scripts require **path modification** before execution. Please update the hardcoded paths to match your actual file locations.
- Please follow the environment setup of VideoLLaMA3.