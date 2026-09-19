#!/usr/bin/env python3
"""Train the category-balanced, randomly paired InternVL2.5 ablation.

This control keeps one positive and one negative example from the same PriorPair
category in every pseudo-pair, but deliberately prevents true matched
counterparts from being co-batched.  It therefore isolates matched-pair
binding from local label/category balance while retaining the Full PhyAR VARC
targets and optimization recipe.
"""

from __future__ import annotations

import argparse
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


SCRIPT_DIR = Path(__file__).resolve().parent
for import_dir in (SCRIPT_DIR,):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from _internvl2_5_training_common import (  # noqa: E402
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
    verify_trainable_parameters,
)
from train_internvl2_5_baseline_sft import (  # noqa: E402
    group_index,
    is_counter_intuitive,
    is_negative_video_path,
    sample_category,
)
from train_internvl2_5_component_ablation import (  # noqa: E402
    BalancedPairDataset,
    collate_one_pair,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train InternVL2.5 with VARC and locally balanced but randomly "
            "matched positive/negative pseudo-pairs."
        )
    )
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
        help="Retain the three-epoch cosine schedule used by Full PhyAR.",
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
    if args.gradient_accumulation_steps % 2:
        raise ValueError(
            "An even accumulation size is required to keep pseudo-pairs intact"
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


def video_path(sample: dict) -> str:
    return sample["video"][0]


def canonical_pair(pair: list[dict]) -> tuple[dict, dict]:
    if len(pair) != 2:
        raise ValueError(f"Expected two samples per pair, received {len(pair)}")
    negatives = [sample for sample in pair if is_negative_video_path(video_path(sample))]
    positives = [sample for sample in pair if not is_negative_video_path(video_path(sample))]
    if len(positives) != 1 or len(negatives) != 1:
        raise ValueError(
            "Each source pair must contain exactly one positive and one negative"
        )
    if sample_category(positives[0]) != sample_category(negatives[0]):
        raise ValueError("Source pair category mismatch")
    return positives[0], negatives[0]


def source_pair_identity(pair: list[dict]) -> tuple[str, str]:
    positive, negative = canonical_pair(pair)
    return video_path(positive), video_path(negative)


def random_derangement(
    positive_entries: list[tuple[tuple[str, str], dict]],
    negative_entries: list[tuple[tuple[str, str], dict]],
    rng: random.Random,
    category: str,
) -> list[tuple[tuple[str, str], dict]]:
    """Randomly permute negatives while forbidding every true counterpart."""
    if len(positive_entries) != len(negative_entries):
        raise RuntimeError(f"Role-count mismatch in category {category}")
    if len(positive_entries) < 2:
        raise RuntimeError(
            f"At least two source pairs are required in category {category}"
        )
    for _ in range(10_000):
        candidate = list(negative_entries)
        rng.shuffle(candidate)
        if all(
            positive_identity != negative_identity
            for (positive_identity, _), (negative_identity, _) in zip(
                positive_entries, candidate
            )
        ):
            return candidate
    raise RuntimeError(
        f"Unable to construct a collision-free random derangement for {category}"
    )


class BalancedUnpairedDataset(BalancedPairDataset):
    """Square-root-balanced pseudo-pairs with category and role held fixed."""

    @property
    def sampling_metadata(self) -> dict:
        metadata = super().sampling_metadata
        metadata.update(
            {
                "variant": "balanced_unpaired",
                "sampling_strategy": (
                    "sqrt_category_balance_then_within_category_"
                    "positive_negative_derangement"
                ),
                "varc": True,
                "pair_binding": False,
                "local_role_balance": True,
                "local_category_match": True,
                "true_counterpart_collisions": 0,
            }
        )
        return metadata

    def set_epoch(self, epoch_index: int) -> None:
        super().set_epoch(epoch_index)
        true_pairs = list(self.pairs)
        rng = random.Random(
            self.sampling_seed + epoch_index * 1_000_003 + 7919
        )
        entries_by_category: dict[
            str, list[tuple[tuple[str, str], dict, dict]]
        ] = {}
        for pair in true_pairs:
            positive, negative = canonical_pair(pair)
            identity = source_pair_identity(pair)
            category = sample_category(positive)
            entries_by_category.setdefault(category, []).append(
                (identity, positive, negative)
            )

        pseudo_pairs: list[list[dict]] = []
        for category in sorted(entries_by_category):
            entries = list(entries_by_category[category])
            rng.shuffle(entries)
            positive_entries = [
                (identity, positive)
                for identity, positive, _ in entries
            ]
            negative_entries = [
                (identity, negative)
                for identity, _, negative in entries
            ]
            permuted_negatives = random_derangement(
                positive_entries,
                negative_entries,
                rng,
                category,
            )
            for (positive_identity, positive), (
                negative_identity,
                negative,
            ) in zip(positive_entries, permuted_negatives):
                if positive_identity == negative_identity:
                    raise RuntimeError("True counterpart escaped derangement audit")
                if sample_category(positive) != sample_category(negative):
                    raise RuntimeError("Pseudo-pair category mismatch")
                pseudo_pairs.append([positive, negative])

        rng.shuffle(pseudo_pairs)
        self._audit_sample_multiset(true_pairs, pseudo_pairs)
        self.pairs = pseudo_pairs
        self.current_epoch = epoch_index

    @staticmethod
    def _audit_sample_multiset(
        true_pairs: list[list[dict]], pseudo_pairs: list[list[dict]]
    ) -> None:
        if len(true_pairs) != len(pseudo_pairs):
            raise RuntimeError("Pseudo-pair count changed during derangement")
        original_roles = {
            "positive": Counter(),
            "negative": Counter(),
        }
        pseudo_roles = {
            "positive": Counter(),
            "negative": Counter(),
        }
        true_partner = {}
        for pair in true_pairs:
            positive, negative = canonical_pair(pair)
            original_roles["positive"][video_path(positive)] += 1
            original_roles["negative"][video_path(negative)] += 1
            true_partner.setdefault(video_path(positive), set()).add(
                video_path(negative)
            )
        for pair in pseudo_pairs:
            positive, negative = canonical_pair(pair)
            pseudo_roles["positive"][video_path(positive)] += 1
            pseudo_roles["negative"][video_path(negative)] += 1
            if video_path(negative) in true_partner.get(video_path(positive), set()):
                raise RuntimeError(
                    "A positive sample was paired with its true negative counterpart"
                )
        if original_roles != pseudo_roles:
            raise RuntimeError("Derangement changed the selected sample multiset")


def prepare_varc_sample(
    sample: dict,
    args: argparse.Namespace,
    tokenizer,
    num_image_token: int,
    device: torch.device | None,
) -> tuple[dict[str, torch.Tensor], str, str]:
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


def audit_dataset_epochs(dataset: BalancedUnpairedDataset) -> None:
    dataset.set_epoch(0)
    first_signature = tuple(
        (video_path(pair[0]), video_path(pair[1])) for pair in dataset.pairs
    )
    dataset.set_epoch(0)
    repeated_signature = tuple(
        (video_path(pair[0]), video_path(pair[1])) for pair in dataset.pairs
    )
    if first_signature != repeated_signature:
        raise RuntimeError("Epoch-0 pseudo-pairing is not reproducible")
    dataset.set_epoch(1)
    second_signature = tuple(
        (video_path(pair[0]), video_path(pair[1])) for pair in dataset.pairs
    )
    if first_signature == second_signature:
        raise RuntimeError("Pseudo-pairing did not change between epochs")
    dataset.set_epoch(0)


def run_preflight(args: argparse.Namespace) -> None:
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.model_max_length = args.max_seq_length
    dataset = BalancedUnpairedDataset(
        args.data_path,
        args.video_root,
        args.max_pairs_per_category,
        args.seed,
    )
    audit_dataset_epochs(dataset)
    metadata = dataset.sampling_metadata
    print(
        "Balanced-unpaired audit passed: "
        f"original_pairs={metadata['original_pair_count']}, "
        f"balanced_pseudo_pairs={metadata['balanced_pairs_per_epoch']}, "
        "true_counterpart_collisions=0, sample_multiset_preserved=yes, "
        "same_seed_reproducible=yes, cross_epoch_pairing_changes=yes"
    )
    print(json.dumps(metadata, indent=2))

    representatives = {}
    for pair in dataset.pairs:
        for sample in pair:
            key = (
                "iv" if is_counter_intuitive(sample) else "i_iii",
                "negative"
                if is_negative_video_path(video_path(sample))
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
        batch, prompt, target = prepare_varc_sample(
            sample, args, tokenizer, num_image_token, None
        )
        print(f"\n[{key[0]} {key[1]}]")
        print(f"video: {video_path(sample)}")
        print(f"prompt:\n{prompt}")
        print(f"target:\n{target}")
        print(
            "tokens: "
            f"total={batch['input_ids'].shape[1]}, "
            f"supervised={int((batch['labels'] != -100).sum())}, "
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

    dataset = BalancedUnpairedDataset(
        args.data_path,
        args.video_root,
        args.max_pairs_per_category,
        args.seed,
    )
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
        "Loading InternVL2.5-8B balanced_unpaired on "
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
    micro_batches_per_rank = items_per_rank * 2
    updates_per_epoch = math.ceil(
        micro_batches_per_rank / args.gradient_accumulation_steps
    )
    scheduler_total_steps = updates_per_epoch * args.scheduler_horizon_epochs
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
        "Ablation schedule: variant=balanced_unpaired, "
        f"global_pseudo_pairs={metadata['balanced_pairs_per_epoch']}, "
        f"global_samples={metadata['balanced_samples_per_epoch']}, "
        f"items_per_rank={items_per_rank}, "
        f"micro_batches_per_rank={micro_batches_per_rank}, "
        f"gradient_accumulation={args.gradient_accumulation_steps}, "
        f"updates_per_epoch={updates_per_epoch}, "
        f"epochs_to_run={args.epochs_to_run}, "
        f"scheduler_horizon_epochs={args.scheduler_horizon_epochs}, "
        f"scheduler_total_steps={scheduler_total_steps}, "
        f"warmup_steps={warmup_steps}, varc=true, pair_binding=false, "
        "local_role_balance=true, local_category_match=true"
    )
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config_record = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        } | {
            "variant": "balanced_unpaired",
            "num_processes": accelerator.num_processes,
            "items_per_rank": items_per_rank,
            "micro_batches_per_rank": micro_batches_per_rank,
            "updates_per_epoch": updates_per_epoch,
            "scheduler_total_steps": scheduler_total_steps,
            "expected_completed_steps": expected_completed_steps,
            "warmup_steps": warmup_steps,
            "varc": True,
            "pair_binding": False,
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
        for pseudo_pair in progress:
            item_losses = []
            for sample in pseudo_pair:
                batch, _, _ = prepare_varc_sample(
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
                        f"Non-finite loss for {video_path(sample)}: {loss.item()}"
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

                if accumulated_micro_batches == args.gradient_accumulation_steps:
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
