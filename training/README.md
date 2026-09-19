# PriorPair Training

This directory contains the data-preparation and balanced QLoRA training scripts used for InternVL2.5-8B and VideoLLaMA3-7B.

## Prepare the data

The dataset root should contain the eight category label files and their referenced positive and negative videos. The command below creates deterministic category-and-scenario-stratified train/test splits and paired SFT JSONL files.

```bash
python prepare_training_data.py \
  --dataset-root <PRIORPAIR_ROOT> \
  --prompts-path ../PriorPair_data_generation/prompts/task_prompts.md
```

Use `--dry-run` to validate the complete dataset without writing files. The generated training file is `<PRIORPAIR_ROOT>/train/train_dataset.jsonl`.

## Train InternVL2.5-8B

```bash
accelerate launch --num_processes 4 train_internvl2_5.py \
  --model-path <INTERNVL2_5_MODEL> \
  --video-root <PRIORPAIR_ROOT> \
  --data-path <PRIORPAIR_ROOT>/train/train_dataset.jsonl \
  --output-dir <OUTPUT_ROOT>/internvl2_5
```

## Train VideoLLaMA3-7B

```bash
accelerate launch --num_processes 4 train_videollama3.py \
  --model-path <VIDEOLLAMA3_MODEL> \
  --video-root <PRIORPAIR_ROOT> \
  --data-path <PRIORPAIR_ROOT>/train/train_dataset.jsonl \
  --output-dir <OUTPUT_ROOT>/videollama3
```

Both scripts train for two epochs by default while parameterizing the cosine schedule over a three-epoch horizon with 5% linear warmup. The final adapter is saved as `checkpoint_epoch_2`. The remaining defaults reproduce the reported balanced training configuration: 16 frames, 4-bit NF4 double quantization, bfloat16 computation, rank-16 LoRA with alpha 32 and dropout 0.1, 8-bit AdamW, and a learning rate of `5e-5`.
