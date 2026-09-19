# PriorPair Evaluations and Ablations

This directory contains the code for the PriorPair/PhyAR evaluations and
ablations reported in the paper.

## Directory map

| Paper experiment | Code |
|---|---|
| PriorPair main benchmark (PCA and RAS) | `priorpair_benchmark/inference/`, `priorpair_benchmark/metrics/` |
| Base-to-PhyAR accuracy and RPB | `priorpair_benchmark/metrics/compute_rpb.py` |
| VARC and Paired Data Binding ablations | `component_ablation/` |
| Random counterpart binding | `component_ablation/train_internvl2_5_balanced_unpaired.py` |
| Training-stream composition ablation | `stream_composition_ablation/` |
| Paired-gradient compatibility analysis | `paired_binding_analysis/` |
| Semantic Prior Dominance diagnosis | `semantic_prior_diagnosis/` |
| IntPhys 2 generalization | `external_generalization/intphys2/` |
| Video-MME capability retention | `external_generalization/videomme/` |
| RAS repeatability | `priorpair_benchmark/metrics/compare_ras_runs.py` |

## Prompts

The final task, RAS, and Semantic Prior Dominance diagnosis prompts are
included at:

```text
prompts/task_prompts.md
prompts/ras_evaluation_prompts.md
prompts/semantic_prior_diagnosis_prompts.md
```

They define the final task prompts, hierarchical RAS rubric, and Observation
judge used by the Semantic Prior Dominance diagnosis.

## 1. PriorPair main benchmark

`priorpair_benchmark/inference/` contains one entry point for each reported Base
model and the two PhyAR backbones. `priorpair_inference_common.py` provides shared
dataset/result helpers.

Every local model must be supplied explicitly. For example:

```bash
python priorpair_benchmark/inference/priorpair_internvl2_5_infer.py \
  --model-path <INTERNVL2_5_MODEL> \
  --video-root <PRIORPAIR_ROOT> \
  --test-data-path <TEST_JSONL> \
  --output-path <OUTPUT_JSON>
```

PhyAR inference additionally requires its PEFT adapter:

```bash
python priorpair_benchmark/inference/phyar_internvl2_5_infer.py \
  --model-path <INTERNVL2_5_MODEL> \
  --adapter-path <PHYAR_ADAPTER> \
  --video-root <PRIORPAIR_ROOT> \
  --test-data-path <TEST_JSONL> \
  --output-path <OUTPUT_JSON>
```

Models with external repositories or auxiliary weights require the explicit
`--code-path`, `--vision-tower-path`, or `--projection-path` arguments shown by
their `--help` output. Video-LLaVA retains its native fixed eight-frame input;
the other reported models use 16 frames by default.

For API-based models, configure the endpoint, credentials, and model identifier:

```bash
OPENAI_API_KEY=<KEY> OPENAI_BASE_URL=<ENDPOINT> \
python priorpair_benchmark/inference/priorpair_gpt4o_infer.py \
  --model <MODEL_NAME> \
  --video-root <PRIORPAIR_ROOT> \
  --test-data-path <TEST_JSONL> \
  --output-path <OUTPUT_JSON>
```

## 2. PCA, RPB, and RAS

```bash
python priorpair_benchmark/metrics/compute_pca.py --inputs <RESULT_JSON> [<RESULT_JSON> ...]
python priorpair_benchmark/metrics/compute_rpb.py --inputs <RESULT_JSON> [<RESULT_JSON> ...]
```

RAS uses an OpenAI-compatible endpoint:

```bash
RAS_API_KEY=<KEY> OPENAI_BASE_URL=<ENDPOINT> \
python priorpair_benchmark/metrics/evaluate_ras_gemini.py \
  --inputs <RESULT_JSON> \
  --output-dir <RAS_OUTPUT_DIR> \
  --prompt-file prompts/ras_evaluation_prompts.md \
  --model <JUDGE_MODEL>
```

Repeated RAS runs can be compared at the strict pair level with:

```bash
python priorpair_benchmark/metrics/compare_ras_runs.py \
  --first-run <FIRST_EVALUATED_JSON> \
  --second-run <SECOND_EVALUATED_JSON>
```

## 3. Component ablations

The component scripts reproduce Baseline SFT, `w/o VARC`, `w/o pair binding`,
and random counterpart binding. `_internvl2_5_training_common.py` contains the
shared data processing, optimization, and adapter-saving utilities used by
these ablation entry points.

```bash
accelerate launch component_ablation/train_internvl2_5_baseline_sft.py \
  --model-path <INTERNVL2_5_MODEL> \
  --video-root <PRIORPAIR_ROOT> \
  --data-path <TRAIN_JSONL> \
  --output-dir <OUTPUT_DIR>

accelerate launch component_ablation/train_internvl2_5_component_ablation.py \
  --variant wo_varc \
  --model-path <INTERNVL2_5_MODEL> \
  --video-root <PRIORPAIR_ROOT> \
  --data-path <TRAIN_JSONL> \
  --output-dir <OUTPUT_DIR>

accelerate launch component_ablation/train_internvl2_5_component_ablation.py \
  --variant wo_pair_binding \
  --model-path <INTERNVL2_5_MODEL> \
  --video-root <PRIORPAIR_ROOT> \
  --data-path <TRAIN_JSONL> \
  --output-dir <OUTPUT_DIR>

accelerate launch component_ablation/train_internvl2_5_balanced_unpaired.py \
  --model-path <INTERNVL2_5_MODEL> \
  --video-root <PRIORPAIR_ROOT> \
  --data-path <TRAIN_JSONL> \
  --output-dir <OUTPUT_DIR>
```

Use `priorpair_benchmark/inference/phyar_internvl2_5_infer.py` with the resulting
adapter to apply the same standard O--A--V evaluation prompt to every ablation.

## 4. Training-stream composition

```bash
accelerate launch stream_composition_ablation/train_internvl2_5_stream_only.py \
  --training-stream anti-physics \
  --model-path <INTERNVL2_5_MODEL> \
  --video-root <PRIORPAIR_ROOT> \
  --data-path <TRAIN_JSONL> \
  --output-dir <OUTPUT_DIR>
```

Use `--training-stream counter-intuitive` for the second single-stream model.
The same standard InternVL2.5 inference and PCA scripts evaluate both models.

## 5. Paired Data Binding mechanism analysis

```bash
accelerate launch paired_binding_analysis/analyze_pair_binding_gradient_directions.py \
  --model-path <INTERNVL2_5_MODEL> \
  --adapter-path <PHYAR_ADAPTER> \
  --video-root <PRIORPAIR_ROOT> \
  --data-path <TRAIN_JSONL> \
  --output-dir <ANALYSIS_DIR>

python paired_binding_analysis/plot_pair_binding_gradient_cosine.py \
  --analysis-dir <ANALYSIS_DIR> \
  --paired-only
```

The analysis computes gradients at a fixed trained checkpoint and performs no
optimizer step or parameter update.

## 6. Semantic Prior Dominance diagnosis

The diagnosis compares the Base and PhyAR generated outputs on all
prior-violating negative counterparts. Observation correctness is judged with
the included stream-specific prompt; Verdict correctness is parsed
deterministically from each task's final Verdict.

```bash
RAS_API_KEY=<KEY> OPENAI_BASE_URL=<ENDPOINT> \
python semantic_prior_diagnosis/evaluate_semantic_prior_diagnosis.py \
  --base-results <BASE_RESULT_JSON> \
  --phyar-results <PHYAR_RESULT_JSON> \
  --prompt-file prompts/semantic_prior_diagnosis_prompts.md \
  --output-file <OUTPUT_JSON> \
  --model <JUDGE_MODEL>
```

## 7. IntPhys 2

Both backbones support `base` and `phyar` weights and `direct` and `varc`
prompt modes. The adapter is required only for `--weights phyar`.

```bash
python external_generalization/intphys2/intphys2_internvl2_5_infer.py \
  --data-root <INTPHYS2_MAIN_ROOT> \
  --model-path <INTERNVL2_5_MODEL> \
  --weights phyar \
  --adapter-path <PHYAR_ADAPTER> \
  --prompt-mode varc \
  --output-file <OUTPUT_JSON>

python external_generalization/intphys2/intphys2_videollama3_infer.py \
  --data-root <INTPHYS2_MAIN_ROOT> \
  --model-path <VIDEOLLAMA3_MODEL> \
  --weights base \
  --prompt-mode direct \
  --output-file <OUTPUT_JSON>
```

## 8. Video-MME

These scripts implement the visual-only setting with the native
multiple-choice prompt and no VARC prompt.

```bash
python external_generalization/videomme/videomme_internvl2_5_infer.py \
  --data-file <VIDEOMME_JSON> \
  --model-path <INTERNVL2_5_MODEL> \
  --weights phyar \
  --adapter-path <PHYAR_ADAPTER> \
  --output-file <OUTPUT_JSON>

python external_generalization/videomme/videomme_videollama3_infer.py \
  --data-file <VIDEOMME_JSON> \
  --model-path <VIDEOLLAMA3_MODEL> \
  --weights base \
  --output-file <OUTPUT_JSON>
```

Use `--smoke` for the built-in non-writing smoke tests in the external
generalization scripts.
