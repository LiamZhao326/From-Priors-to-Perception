#!/usr/bin/env python3
"""Controlled InternVL2.5 component ablations for PriorPair."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)


SCRIPT_DIR = Path(__file__).resolve().parent
for import_dir in (SCRIPT_DIR,):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from _internvl2_5_training_common import (  # noqa: E402
    IGNORE_INDEX,
    IMG_CONTEXT_TOKEN,
    SYSTEM_MESSAGE,
    build_lora_target_modules,
    calculate_num_image_tokens,
    count_parameters,
    extract_answer,
    extract_prompt,
    optimizer_update,
    prepare_training_sample,
    save_adapter,
    validate_pair,
    verify_trainable_parameters,
)
from train_internvl2_5_baseline_sft import (  # noqa: E402
    BalancedIndividualDataset,
    build_baseline_prompt_and_target,
    collate_one_sample,
    group_index,
    is_counter_intuitive,
    is_negative_video_path,
    prepare_baseline_sample,
    sample_category,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

VARIANTS = ("wo_varc", "wo_pair_binding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the w/o VARC or w/o Paired Data Binding InternVL2.5 "
            "component ablation."
        )
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument(
        "--epochs-to-run",
        type=int,
        default=2,
        help="Stop after this epoch and save only that checkpoint.",
    )
    parser.add_argument(
        "--scheduler-horizon-epochs",
        type=int,
        default=3,
        help=(
            "Keep the same three-epoch cosine horizon as the existing Full "
            "PhyAR and Baseline SFT runs."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Use 4 with two GPUs for an eight-video global update.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-pairs-per-category",
        type=int,
        default=None,
        help="Smoke-only limit applied before square-root balancing.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not args.video_root.is_dir():
        raise FileNotFoundError(f"Video root not found: {args.video_root}")
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Training JSONL not found: {args.data_path}")
    if args.num_frames <= 0 or args.max_seq_length <= 0:
        raise ValueError("Frame count and maximum sequence length must be positive")
    if args.epochs_to_run <= 0:
        raise ValueError("--epochs-to-run must be positive")
    if args.scheduler_horizon_epochs < args.epochs_to_run:
        raise ValueError(
            "--scheduler-horizon-epochs cannot be shorter than --epochs-to-run"
        )
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.variant == "wo_varc" and args.gradient_accumulation_steps % 2:
        raise ValueError(
            "w/o VARC requires an even accumulation size to preserve pairs"
        )
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if (
        args.max_pairs_per_category is not None
        and args.max_pairs_per_category <= 0
    ):
        raise ValueError("--max-pairs-per-category must be positive")


def read_pairs(data_path: Path, video_root: Path) -> list[list[dict]]:
    samples = []
    with data_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {data_path}:{line_number}"
                ) from error
    if not samples or len(samples) % 2:
        raise ValueError(
            f"Expected a non-empty even number of samples, got {len(samples)}"
        )
    pairs = [
        [samples[index], samples[index + 1]]
        for index in range(0, len(samples), 2)
    ]
    for pair_index, pair in enumerate(pairs):
        validate_pair(pair, pair_index, video_root)
        for sample in pair:
            build_baseline_prompt_and_target(sample)
    return pairs


class BalancedPairDataset(Dataset):
    """Square-root-balanced, globally shuffled atomic PriorPair pairs."""

    def __init__(
        self,
        data_path: Path,
        video_root: Path,
        max_pairs_per_category: int | None,
        sampling_seed: int,
    ) -> None:
        all_pairs = read_pairs(data_path, video_root)
        category_pairs: dict[str, list[list[dict]]] = {}
        for pair in all_pairs:
            category_pairs.setdefault(sample_category(pair[0]), []).append(pair)
        if max_pairs_per_category is not None:
            category_pairs = {
                category: pairs[:max_pairs_per_category]
                for category, pairs in category_pairs.items()
            }
        self.category_pairs = category_pairs
        self.original_pair_count = sum(
            len(pairs) for pairs in category_pairs.values()
        )
        self.category_original_counts = {
            category: len(pairs)
            for category, pairs in sorted(category_pairs.items())
        }
        self.max_category_count = max(self.category_original_counts.values())
        self.category_target_pair_counts = {
            category: math.ceil(math.sqrt(count * self.max_category_count))
            for category, count in self.category_original_counts.items()
        }
        self.balanced_pair_count = sum(self.category_target_pair_counts.values())
        self.sampling_seed = sampling_seed
        self.current_epoch = -1
        self.pairs: list[list[dict]] = []
        self.set_epoch(0)

    @property
    def sampling_metadata(self) -> dict:
        return {
            "sampling_strategy": "sqrt_category_balance_then_global_pair_shuffle",
            "pair_binding": True,
            "sampling_seed": self.sampling_seed,
            "original_pair_count": self.original_pair_count,
            "original_sample_count": 2 * self.original_pair_count,
            "balanced_pairs_per_epoch": self.balanced_pair_count,
            "balanced_samples_per_epoch": 2 * self.balanced_pair_count,
            "category_original_pair_counts": self.category_original_counts,
            "category_target_pair_counts": self.category_target_pair_counts,
        }

    def set_epoch(self, epoch_index: int) -> None:
        if epoch_index < 0:
            raise ValueError("epoch_index must be non-negative")
        rng = random.Random(self.sampling_seed + epoch_index * 1_000_003)
        balanced_pairs = []
        for category in sorted(self.category_pairs):
            source_pairs = self.category_pairs[category]
            balanced_pairs.extend(source_pairs)
            remaining = self.category_target_pair_counts[category] - len(
                source_pairs
            )
            while remaining > 0:
                shuffled_cycle = list(source_pairs)
                rng.shuffle(shuffled_cycle)
                take_count = min(remaining, len(shuffled_cycle))
                balanced_pairs.extend(shuffled_cycle[:take_count])
                remaining -= take_count
        rng.shuffle(balanced_pairs)
        if len(balanced_pairs) != self.balanced_pair_count:
            raise RuntimeError("Balanced pair count mismatch")
        self.pairs = balanced_pairs
        self.current_epoch = epoch_index

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> list[dict]:
        return self.pairs[index]


def collate_one_pair(batch: list[list[dict]]) -> list[dict]:
    if len(batch) != 1:
        raise ValueError(f"Expected one pair, received {len(batch)}")
    pair = batch[0]
    if len(pair) != 2:
        raise ValueError("A pair must contain exactly two samples")
    return pair


def make_dataset(args: argparse.Namespace) -> Dataset:
    if args.variant == "wo_varc":
        return BalancedPairDataset(
            args.data_path,
            args.video_root,
            args.max_pairs_per_category,
            args.seed,
        )
    return BalancedIndividualDataset(
        args.data_path,
        args.video_root,
        args.max_pairs_per_category,
        args.seed,
    )


def flatten_dataset_samples(dataset: Dataset, variant: str) -> list[dict]:
    if variant == "wo_varc":
        return [sample for pair in dataset.pairs for sample in pair]
    return list(dataset.samples)


def prepare_variant_sample(
    sample: dict,
    args: argparse.Namespace,
    tokenizer,
    num_image_token: int,
    device: torch.device | None,
) -> tuple[dict[str, torch.Tensor], str, str]:
    if args.variant == "wo_varc":
        return prepare_baseline_sample(
            sample,
            tokenizer,
            args.video_root,
            args.num_frames,
            num_image_token,
            args.max_seq_length,
            device,
        )
    batch = prepare_training_sample(
        sample,
        tokenizer,
        args.video_root,
        args.num_frames,
        num_image_token,
        args.max_seq_length,
        device=device,
    )
    return batch, extract_prompt(sample), extract_answer(sample)


def run_preflight(args: argparse.Namespace) -> None:
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.model_max_length = args.max_seq_length
    dataset = make_dataset(args)
    samples = flatten_dataset_samples(dataset, args.variant)
    metadata = dataset.sampling_metadata
    print(
        f"Validated variant={args.variant}: "
        f"original_pairs={metadata['original_pair_count']}, "
        f"balanced_samples={metadata['balanced_samples_per_epoch']}, "
        f"pair_binding={str(metadata['pair_binding']).lower()}"
    )
    print(json.dumps(metadata, indent=2))

    representatives = {}
    for sample in samples:
        key = (
            "iv" if is_counter_intuitive(sample) else "i_iii",
            "negative"
            if is_negative_video_path(sample["video"][0])
            else "positive",
        )
        representatives.setdefault(key, sample)
    expected_keys = {
        ("i_iii", "positive"),
        ("i_iii", "negative"),
        ("iv", "positive"),
        ("iv", "negative"),
    }
    if set(representatives) != expected_keys:
        raise RuntimeError(
            f"Missing representative groups: {expected_keys - set(representatives)}"
        )

    num_image_token = calculate_num_image_tokens(config)
    for key in sorted(representatives):
        sample = representatives[key]
        batch, prompt, target = prepare_variant_sample(
            sample, args, tokenizer, num_image_token, None
        )
        print(f"\n[{key[0]} {key[1]}]")
        print(f"video: {sample['video'][0]}")
        print(f"prompt:\n{prompt}")
        print(f"target:\n{target}")
        print(
            "tokens: "
            f"total={batch['input_ids'].shape[1]}, "
            f"supervised={int((batch['labels'] != IGNORE_INDEX).sum())}, "
            f"pixel_values={tuple(batch['pixel_values'].shape)}"
        )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InternVL2.5 training")
    torch.cuda.set_device(accelerator.local_process_index)
    checkpoint_name = f"checkpoint_epoch_{args.epochs_to_run}"
    if (args.output_dir / checkpoint_name).exists():
        raise FileExistsError(
            f"Refusing to overwrite {args.output_dir / checkpoint_name}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    num_image_token = calculate_num_image_tokens(config)
    image_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    dataset = make_dataset(args)
    pair_binding = args.variant == "wo_varc"
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_one_pair if pair_binding else collate_one_sample,
        num_workers=0,
        pin_memory=True,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["vision_model", "mlp1"],
    )
    accelerator.print(
        f"Loading InternVL2.5-8B variant={args.variant} on "
        f"{accelerator.device} in 4-bit NF4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        quantization_config=quantization_config,
        device_map={"": accelerator.device},
    )
    model.img_context_token_id = image_context_token_id
    model.system_message = SYSTEM_MESSAGE
    model.config.use_cache = False
    model.language_model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model.vision_model.requires_grad_(False)

    target_modules = build_lora_target_modules(
        config.llm_config.num_hidden_layers
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable_names = verify_trainable_parameters(model)
    total_parameters, trainable_parameters = count_parameters(model)
    accelerator.print(
        f"Parameters: total={total_parameters:,}, "
        f"trainable={trainable_parameters:,} "
        f"({100 * trainable_parameters / total_parameters:.4f}%)"
    )
    accelerator.print(
        f"Trainable tensor groups={len(trainable_names)}; "
        f"LoRA target modules={len(target_modules)}"
    )

    optimizer = bnb.optim.AdamW8bit(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    items_per_rank = math.ceil(len(dataset) / accelerator.num_processes)
    micro_batches_per_rank = items_per_rank * (2 if pair_binding else 1)
    updates_per_epoch = math.ceil(
        micro_batches_per_rank / args.gradient_accumulation_steps
    )
    scheduler_total_steps = (
        updates_per_epoch * args.scheduler_horizon_epochs
    )
    expected_completed_steps = updates_per_epoch * args.epochs_to_run
    warmup_steps = int(scheduler_total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=scheduler_total_steps,
    )
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    if len(dataloader) != items_per_rank:
        raise RuntimeError(
            f"Unexpected dataloader length {len(dataloader)}; "
            f"expected {items_per_rank}"
        )
    optimizer.zero_grad(set_to_none=True)

    metadata = dataset.sampling_metadata
    accelerator.print(
        f"Ablation schedule: variant={args.variant}, "
        f"global_pairs={metadata['balanced_pairs_per_epoch']}, "
        f"global_samples={metadata['balanced_samples_per_epoch']}, "
        f"items_per_rank={items_per_rank}, "
        f"micro_batches_per_rank={micro_batches_per_rank}, "
        f"gradient_accumulation={args.gradient_accumulation_steps}, "
        f"updates_per_epoch={updates_per_epoch}, "
        f"epochs_to_run={args.epochs_to_run}, "
        f"scheduler_horizon_epochs={args.scheduler_horizon_epochs}, "
        f"scheduler_total_steps={scheduler_total_steps}, "
        f"warmup_steps={warmup_steps}, "
        f"pair_binding={str(pair_binding).lower()}"
    )
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config_record = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        } | {
            "num_processes": accelerator.num_processes,
            "items_per_rank": items_per_rank,
            "micro_batches_per_rank": micro_batches_per_rank,
            "updates_per_epoch": updates_per_epoch,
            "scheduler_total_steps": scheduler_total_steps,
            "expected_completed_steps": expected_completed_steps,
            "warmup_steps": warmup_steps,
            "varc": args.variant == "wo_pair_binding",
            "pair_binding": pair_binding,
            "saved_checkpoint": checkpoint_name,
        } | metadata
        (args.output_dir / "training_config.json").write_text(
            json.dumps(config_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "epoch_loss_metrics.jsonl").write_text(
            "", encoding="utf-8"
        )

    global_update_step = 0
    for epoch_index in range(args.epochs_to_run):
        dataset.set_epoch(epoch_index)
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch_index)
        model.train()
        accumulated_micro_batches = 0
        running_loss = 0.0
        processed_samples = 0
        group_loss_sums = [0.0, 0.0, 0.0, 0.0]
        group_sample_counts = [0, 0, 0, 0]
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch_index + 1}/{args.epochs_to_run}",
            disable=not accelerator.is_local_main_process,
        )
        for item in progress:
            samples = item if pair_binding else [item]
            item_losses = []
            for sample in samples:
                batch, _, _ = prepare_variant_sample(
                    sample,
                    args,
                    tokenizer,
                    num_image_token,
                    accelerator.device,
                )
                outputs = model(**batch)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss for {sample['video'][0]}: {loss.item()}"
                    )
                accelerator.backward(loss)
                loss_value = loss.detach().float().item()
                accumulated_micro_batches += 1
                processed_samples += 1
                running_loss += loss_value
                index = group_index(sample)
                group_loss_sums[index] += loss_value
                group_sample_counts[index] += 1
                item_losses.append(loss_value)

                if (
                    accumulated_micro_batches
                    == args.gradient_accumulation_steps
                ):
                    if optimizer_update(
                        model,
                        optimizer,
                        scheduler,
                        accelerator,
                        args.max_grad_norm,
                    ):
                        global_update_step += 1
                    accumulated_micro_batches = 0
                del batch, outputs, loss

            progress.set_postfix(
                loss=f"{sum(item_losses) / len(item_losses):.4f}",
                update=global_update_step,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        if accumulated_micro_batches:
            gradient_scale = (
                args.gradient_accumulation_steps / accumulated_micro_batches
            )
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)
            if optimizer_update(
                model,
                optimizer,
                scheduler,
                accelerator,
                args.max_grad_norm,
            ):
                global_update_step += 1

        local_stats = torch.tensor(
            [
                running_loss,
                float(processed_samples),
                *[
                    value
                    for index in range(4)
                    for value in (
                        group_loss_sums[index],
                        float(group_sample_counts[index]),
                    )
                ],
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )
        global_stats = accelerator.reduce(local_stats, reduction="sum")
        average_loss = global_stats[0].item() / global_stats[1].item()
        group_average_losses = [
            global_stats[2 + 2 * index].item()
            / global_stats[3 + 2 * index].item()
            if global_stats[3 + 2 * index].item() > 0
            else 0.0
            for index in range(4)
        ]
        group_global_counts = [
            int(global_stats[3 + 2 * index].item()) for index in range(4)
        ]
        accelerator.print(
            f"Epoch {epoch_index + 1} complete: "
            f"global_average_loss={average_loss:.6f}, "
            f"i_iii_positive_loss={group_average_losses[0]:.6f}, "
            f"i_iii_negative_loss={group_average_losses[1]:.6f}, "
            f"iv_positive_loss={group_average_losses[2]:.6f}, "
            f"iv_negative_loss={group_average_losses[3]:.6f}, "
            f"updates={global_update_step}, "
            f"lr={scheduler.get_last_lr()[0]:.6e}"
        )
        if accelerator.is_main_process:
            epoch_metrics = {
                "epoch": epoch_index + 1,
                "global_average_loss": average_loss,
                "i_iii_positive_loss": group_average_losses[0],
                "i_iii_negative_loss": group_average_losses[1],
                "iv_positive_loss": group_average_losses[2],
                "iv_negative_loss": group_average_losses[3],
                "i_iii_positive_count": group_global_counts[0],
                "i_iii_negative_count": group_global_counts[1],
                "iv_positive_count": group_global_counts[2],
                "iv_negative_count": group_global_counts[3],
                "global_update_step": global_update_step,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            with (args.output_dir / "epoch_loss_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as output_file:
                output_file.write(
                    json.dumps(epoch_metrics, ensure_ascii=False) + "\n"
                )

    if global_update_step != expected_completed_steps:
        raise RuntimeError(
            f"Optimizer step mismatch: {global_update_step} vs. "
            f"{expected_completed_steps}"
        )
    save_adapter(
        model,
        tokenizer,
        args.output_dir / checkpoint_name,
        accelerator,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.preflight_only:
        run_preflight(args)
        return
    train(args)


if __name__ == "__main__":
    main()
