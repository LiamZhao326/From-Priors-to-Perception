#!/usr/bin/env python3
"""Train InternVL2.5 on one PriorPair training stream with full PhyAR targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
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


COMPONENT_DIR = Path(__file__).resolve().parents[1] / "component_ablation"
if str(COMPONENT_DIR) not in sys.path:
    sys.path.insert(0, str(COMPONENT_DIR))

from _internvl2_5_training_common import (  # noqa: E402
    IMG_CONTEXT_TOKEN,
    SYSTEM_MESSAGE,
    build_lora_target_modules,
    calculate_num_image_tokens,
    count_parameters,
    optimizer_update,
    prepare_training_sample,
    save_adapter,
    validate_pair,
    verify_trainable_parameters,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

STREAM_CATEGORIES = {
    "anti-physics": (
        "I_A_Coherence_Violation",
        "I_B_Causal_Reversal",
        "II_A_Existence_Violation",
        "II_B_Identity_Violation",
        "III_A_Dynamic_Violation",
        "III_B_Constraint_Violation",
    ),
    "counter-intuitive": (
        "IV_A_Near_Miss",
        "IV_B_Consequence_Arrest",
    ),
}
OUTPUT_NAMES = {
    "anti-physics": "internvl2_5_anti_physics_only",
    "counter-intuitive": "internvl2_5_counter_intuitive_only",
}

# Match the Joint Full run's learning-rate trajectory, but stop after epoch 2.
EPOCHS_TO_RUN = 2
SCHEDULER_HORIZON_EPOCHS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Anti-physics-only or Counter-intuitive-only PriorPair "
            "stream-composition ablation with full VARC and pair binding."
        )
    )
    parser.add_argument(
        "--training-stream",
        choices=tuple(STREAM_CATEGORIES),
        required=True,
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help=(
            "Must be even so every positive/negative pair is consumed within "
            "one optimizer update."
        ),
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
        help=(
            "Smoke-only limit applied to every category before computing the "
            "global square-root reference maximum."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate the dataset and preprocess one pair without loading the "
            "training model or writing outputs."
        ),
    )
    parser.add_argument(
        "--balance-audit-only",
        action="store_true",
        help=(
            "Audit deterministic square-root sampling without loading the "
            "tokenizer or model and without writing outputs."
        ),
    )
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
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.gradient_accumulation_steps % 2:
        raise ValueError(
            "--gradient-accumulation-steps must be even to preserve pair binding"
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
    if args.preflight_only and args.balance_audit_only:
        raise ValueError(
            "--preflight-only and --balance-audit-only are mutually exclusive"
        )


def pair_category(pair: list[dict]) -> str:
    return (
        Path(pair[0]["video"][0])
        .parts[0]
        .replace("_negative_aligned", "")
        .replace("_negative", "")
    )


def pair_identifier(pair: list[dict]) -> tuple[str, str]:
    return pair[0]["video"][0], pair[1]["video"][0]


def read_all_category_pairs(
    data_path: Path,
    video_root: Path,
    max_pairs_per_category: int | None,
) -> dict[str, list[list[dict]]]:
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

    all_pairs = [
        [samples[index], samples[index + 1]]
        for index in range(0, len(samples), 2)
    ]
    category_pairs: dict[str, list[list[dict]]] = {}
    for pair_index, pair in enumerate(all_pairs):
        validate_pair(pair, pair_index, video_root)
        category_pairs.setdefault(pair_category(pair), []).append(pair)

    expected_categories = {
        category
        for categories in STREAM_CATEGORIES.values()
        for category in categories
    }
    if set(category_pairs) != expected_categories:
        raise ValueError(
            "Training categories do not match the expected eight categories: "
            f"missing={sorted(expected_categories - set(category_pairs))}, "
            f"unexpected={sorted(set(category_pairs) - expected_categories)}"
        )
    if max_pairs_per_category is not None:
        category_pairs = {
            category: pairs[:max_pairs_per_category]
            for category, pairs in category_pairs.items()
        }
    if any(not pairs for pairs in category_pairs.values()):
        raise ValueError("At least one category contains no pairs")
    return category_pairs


class StreamBalancedPairDataset(Dataset):
    """Atomic pairs sampled using Joint Full's global square-root targets."""

    def __init__(
        self,
        data_path: Path,
        video_root: Path,
        training_stream: str,
        max_pairs_per_category: int | None,
        sampling_seed: int,
    ) -> None:
        all_category_pairs = read_all_category_pairs(
            data_path,
            video_root,
            max_pairs_per_category,
        )
        self.full_category_original_counts = {
            category: len(pairs)
            for category, pairs in sorted(all_category_pairs.items())
        }
        self.global_reference_max = max(
            self.full_category_original_counts.values()
        )
        selected_categories = STREAM_CATEGORIES[training_stream]
        self.category_pairs = {
            category: all_category_pairs[category]
            for category in selected_categories
        }
        self.category_original_counts = {
            category: len(pairs)
            for category, pairs in self.category_pairs.items()
        }
        self.category_target_pair_counts = {
            category: math.ceil(
                math.sqrt(count * self.global_reference_max)
            )
            for category, count in self.category_original_counts.items()
        }
        self.training_stream = training_stream
        self.original_pair_count = sum(self.category_original_counts.values())
        self.balanced_pair_count = sum(
            self.category_target_pair_counts.values()
        )
        self.sampling_seed = sampling_seed
        self.current_epoch = -1
        self.pairs: list[list[dict]] = []
        self.set_epoch(0)

    @property
    def sampling_metadata(self) -> dict:
        return {
            "sampling_strategy": (
                "joint_global_sqrt_category_balance_then_stream_filter_"
                "then_global_pair_shuffle"
            ),
            "training_stream": self.training_stream,
            "selected_categories": list(self.category_pairs),
            "varc": True,
            "pair_binding": True,
            "sampling_seed": self.sampling_seed,
            "global_reference_max_pair_count": self.global_reference_max,
            "full_category_original_pair_counts": (
                self.full_category_original_counts
            ),
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
        for category in self.category_pairs:
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


def make_dataset(args: argparse.Namespace) -> StreamBalancedPairDataset:
    return StreamBalancedPairDataset(
        args.data_path,
        args.video_root,
        args.training_stream,
        args.max_pairs_per_category,
        args.seed,
    )


def sampling_fingerprint(pairs: list[list[dict]]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update("\0".join(pair_identifier(pair)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_epoch(
    dataset: StreamBalancedPairDataset,
    epoch_index: int,
) -> dict:
    dataset.set_epoch(epoch_index)
    observed_counts = Counter(pair_category(pair) for pair in dataset.pairs)
    if dict(observed_counts) != dataset.category_target_pair_counts:
        raise RuntimeError(
            f"Epoch {epoch_index + 1} category counts differ from targets: "
            f"{dict(observed_counts)}"
        )
    observed_identifiers = Counter(
        pair_identifier(pair) for pair in dataset.pairs
    )
    missing_original_pairs = []
    for pairs in dataset.category_pairs.values():
        for pair in pairs:
            identifier = pair_identifier(pair)
            if observed_identifiers[identifier] < 1:
                missing_original_pairs.append(identifier)
    if missing_original_pairs:
        raise RuntimeError(
            f"Epoch {epoch_index + 1} omitted original pairs: "
            f"{missing_original_pairs[:3]}"
        )
    return {
        "epoch": epoch_index + 1,
        "pair_count": len(dataset.pairs),
        "category_pair_counts": dict(observed_counts),
        "ordered_sampling_sha256": sampling_fingerprint(dataset.pairs),
    }


def run_balance_audit(args: argparse.Namespace) -> None:
    dataset = make_dataset(args)
    reports = [audit_epoch(dataset, epoch) for epoch in range(EPOCHS_TO_RUN)]
    replica = make_dataset(args)
    replica_reports = [
        audit_epoch(replica, epoch) for epoch in range(EPOCHS_TO_RUN)
    ]
    if reports != replica_reports:
        raise RuntimeError("Sampling is not reproducible for the configured seed")
    if reports[0]["ordered_sampling_sha256"] == reports[1][
        "ordered_sampling_sha256"
    ]:
        raise RuntimeError("Epoch 1 and epoch 2 unexpectedly have identical order")
    print(json.dumps(dataset.sampling_metadata, ensure_ascii=False, indent=2))
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print("Balance audit passed: all original pairs retained; counts and seed are reproducible.")


def run_preflight(args: argparse.Namespace) -> None:
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = make_dataset(args)
    num_image_token = calculate_num_image_tokens(config)
    print(json.dumps(dataset.sampling_metadata, ensure_ascii=False, indent=2))
    for role, sample in zip(("positive", "negative"), dataset.pairs[0]):
        batch = prepare_training_sample(
            sample,
            tokenizer,
            args.video_root,
            args.num_frames,
            num_image_token,
            args.max_seq_length,
            device=None,
        )
        print(
            f"{role}: video={sample['video'][0]}, "
            f"tokens={batch['input_ids'].shape[1]}, "
            f"pixel_values={tuple(batch['pixel_values'].shape)}"
        )


def write_run_metadata(
    args: argparse.Namespace,
    dataset: StreamBalancedPairDataset,
    accelerator: Accelerator,
    items_per_rank: int,
    micro_batches_per_rank: int,
    updates_per_epoch: int,
    scheduler_total_steps: int,
    expected_completed_steps: int,
    warmup_steps: int,
    checkpoint_name: str,
) -> None:
    config_record = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    } | {
        "backbone": "InternVL2.5-8B",
        "epochs_to_run": EPOCHS_TO_RUN,
        "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
        "num_processes": accelerator.num_processes,
        "items_per_rank": items_per_rank,
        "micro_batches_per_rank": micro_batches_per_rank,
        "updates_per_epoch": updates_per_epoch,
        "scheduler_total_steps": scheduler_total_steps,
        "expected_completed_steps": expected_completed_steps,
        "warmup_steps": warmup_steps,
        "saved_checkpoint": checkpoint_name,
    } | dataset.sampling_metadata
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "epoch_loss_metrics.jsonl").write_text(
        "",
        encoding="utf-8",
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

    checkpoint_name = f"checkpoint_epoch_{EPOCHS_TO_RUN}"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    num_image_token = calculate_num_image_tokens(config)
    image_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    dataset = make_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_one_pair,
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
        f"Loading InternVL2.5-8B stream={args.training_stream} on "
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
        model,
        use_gradient_checkpointing=True,
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
    micro_batches_per_rank = items_per_rank * 2
    updates_per_epoch = math.ceil(
        micro_batches_per_rank / args.gradient_accumulation_steps
    )
    scheduler_total_steps = updates_per_epoch * SCHEDULER_HORIZON_EPOCHS
    expected_completed_steps = updates_per_epoch * EPOCHS_TO_RUN
    warmup_steps = int(scheduler_total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=scheduler_total_steps,
    )
    model, optimizer, dataloader = accelerator.prepare(
        model,
        optimizer,
        dataloader,
    )
    if len(dataloader) != items_per_rank:
        raise RuntimeError(
            f"Unexpected dataloader length {len(dataloader)}; "
            f"expected {items_per_rank}"
        )
    optimizer.zero_grad(set_to_none=True)

    metadata = dataset.sampling_metadata
    accelerator.print(
        f"Training schedule: stream={args.training_stream}, "
        f"global_pairs={metadata['balanced_pairs_per_epoch']}, "
        f"global_samples={metadata['balanced_samples_per_epoch']}, "
        f"items_per_rank={items_per_rank}, "
        f"micro_batches_per_rank={micro_batches_per_rank}, "
        f"gradient_accumulation={args.gradient_accumulation_steps}, "
        f"updates_per_epoch={updates_per_epoch}, "
        f"epochs_to_run={EPOCHS_TO_RUN}, "
        f"scheduler_horizon_epochs={SCHEDULER_HORIZON_EPOCHS}, "
        f"scheduler_total_steps={scheduler_total_steps}, "
        f"warmup_steps={warmup_steps}, varc=true, pair_binding=true"
    )
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(
            args,
            dataset,
            accelerator,
            items_per_rank,
            micro_batches_per_rank,
            updates_per_epoch,
            scheduler_total_steps,
            expected_completed_steps,
            warmup_steps,
            checkpoint_name,
        )

    global_update_step = 0
    for epoch_index in range(EPOCHS_TO_RUN):
        dataset.set_epoch(epoch_index)
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch_index)
        model.train()
        accumulated_micro_batches = 0
        running_loss = 0.0
        processed_samples = 0
        role_loss_sums = [0.0, 0.0]
        role_sample_counts = [0, 0]
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch_index + 1}/{EPOCHS_TO_RUN}",
            disable=not accelerator.is_local_main_process,
        )
        for pair in progress:
            pair_losses = []
            for role_index, sample in enumerate(pair):
                batch = prepare_training_sample(
                    sample,
                    tokenizer,
                    args.video_root,
                    args.num_frames,
                    num_image_token,
                    args.max_seq_length,
                    device=accelerator.device,
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
                running_loss += loss_value
                processed_samples += 1
                role_loss_sums[role_index] += loss_value
                role_sample_counts[role_index] += 1
                pair_losses.append(loss_value)

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
                loss=f"{sum(pair_losses) / len(pair_losses):.4f}",
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
                role_loss_sums[0],
                float(role_sample_counts[0]),
                role_loss_sums[1],
                float(role_sample_counts[1]),
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )
        global_stats = accelerator.reduce(local_stats, reduction="sum")
        average_loss = global_stats[0].item() / global_stats[1].item()
        positive_loss = global_stats[2].item() / global_stats[3].item()
        negative_loss = global_stats[4].item() / global_stats[5].item()
        accelerator.print(
            f"Epoch {epoch_index + 1} complete: "
            f"global_average_loss={average_loss:.6f}, "
            f"positive_loss={positive_loss:.6f}, "
            f"negative_loss={negative_loss:.6f}, "
            f"updates={global_update_step}, "
            f"lr={scheduler.get_last_lr()[0]:.6e}"
        )
        if accelerator.is_main_process:
            epoch_metrics = {
                "epoch": epoch_index + 1,
                "global_average_loss": average_loss,
                "positive_loss": positive_loss,
                "negative_loss": negative_loss,
                "positive_count": int(global_stats[3].item()),
                "negative_count": int(global_stats[5].item()),
                "global_update_step": global_update_step,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            with (args.output_dir / "epoch_loss_metrics.jsonl").open(
                "a",
                encoding="utf-8",
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
    if args.balance_audit_only:
        run_balance_audit(args)
        return
    if args.preflight_only:
        run_preflight(args)
        return
    train(args)


if __name__ == "__main__":
    main()
