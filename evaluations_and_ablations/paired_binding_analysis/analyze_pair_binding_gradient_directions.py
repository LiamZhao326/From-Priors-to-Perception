#!/usr/bin/env python3
"""Compare matched and randomly bound PriorPair gradient directions.

The analysis evaluates a fixed trained InternVL2.5 LoRA adapter.  It computes
one gradient sketch per selected record, then reconstructs the pre-optimizer
gradient of each eight-record global update window under two arrangements:

1. true matched positive/negative counterparts;
2. the identical records with negatives deranged within each category.

No optimizer step is taken and no model state is modified.  Window gradients
are L2-normalized and projected together with an uncentered SVD so the origin
retains its meaning as zero update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from peft import PeftModel, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_DIR = PACKAGE_ROOT / "component_ablation"
for import_dir in (COMPONENT_DIR,):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from _internvl2_5_training_common import (  # noqa: E402
    IMG_CONTEXT_TOKEN,
    SYSTEM_MESSAGE,
    calculate_num_image_tokens,
    prepare_training_sample,
)
from train_internvl2_5_baseline_sft import sample_category  # noqa: E402
from train_internvl2_5_component_ablation import read_pairs  # noqa: E402


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/priorpair_pair_binding_gradient_analysis"),
        help="Shared temporary directory for per-rank gradient sketches.",
    )
    parser.add_argument("--num-pairs", type=int, default=64)
    parser.add_argument("--pairs-per-window", type=int, default=4)
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional single-category restriction for a smoke run.",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate selection, quotas, derangement and windows without a model.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render PDF/PNG from an existing gradient_direction_vectors.json.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.render_only:
        vector_path = args.output_dir / "gradient_direction_vectors.json"
        if not vector_path.is_file():
            raise FileNotFoundError(f"Vector JSON not found: {vector_path}")
        return
    for path, description, is_dir in (
        (args.model_path, "base model", True),
        (args.adapter_path, "LoRA adapter", True),
        (args.video_root, "video root", True),
        (args.data_path, "training JSONL", False),
    ):
        exists = path.is_dir() if is_dir else path.is_file()
        if not exists:
            raise FileNotFoundError(f"{description} not found: {path}")
    if args.num_pairs < 2:
        raise ValueError("--num-pairs must be at least two")
    if args.pairs_per_window <= 0:
        raise ValueError("--pairs-per-window must be positive")
    if args.num_pairs % args.pairs_per_window:
        raise ValueError("--num-pairs must be divisible by --pairs-per-window")
    if args.num_frames <= 0 or args.max_seq_length <= 0:
        raise ValueError("Frame count and sequence length must be positive")
    if args.sketch_dim < 2:
        raise ValueError("--sketch-dim must be at least two")


def largest_remainder_quotas(
    category_counts: dict[str, int], total: int
) -> dict[str, int]:
    """Allocate unique pairs proportional to sqrt(category size)."""
    if total > sum(category_counts.values()):
        raise ValueError("Requested more unique pairs than the dataset contains")
    weights = {category: math.sqrt(count) for category, count in category_counts.items()}
    denominator = sum(weights.values())
    ideals = {category: total * weight / denominator for category, weight in weights.items()}
    quotas = {category: max(2, math.floor(value)) for category, value in ideals.items()}
    if any(quotas[category] > category_counts[category] for category in quotas):
        raise ValueError("A category cannot supply its square-root quota")
    while sum(quotas.values()) < total:
        candidates = [
            category
            for category in quotas
            if quotas[category] < category_counts[category]
        ]
        category = max(
            candidates,
            key=lambda key: (ideals[key] - quotas[key], category_counts[key], key),
        )
        quotas[category] += 1
    while sum(quotas.values()) > total:
        candidates = [category for category in quotas if quotas[category] > 2]
        if not candidates:
            raise ValueError("Pair budget is too small to retain two per category")
        category = min(
            candidates,
            key=lambda key: (ideals[key] - quotas[key], category_counts[key], key),
        )
        quotas[category] -= 1
    return dict(sorted(quotas.items()))


def derange_pair_ids(pair_ids: list[int], rng: random.Random) -> list[int]:
    if len(pair_ids) < 2:
        raise ValueError("At least two pair IDs are required for derangement")
    for _ in range(10_000):
        candidate = list(pair_ids)
        rng.shuffle(candidate)
        if all(source != target for source, target in zip(pair_ids, candidate)):
            return candidate
    raise RuntimeError("Could not construct a collision-free derangement")


def build_analysis_plan(args: argparse.Namespace) -> tuple[list[list[dict]], dict[str, Any]]:
    pairs = read_pairs(args.data_path, args.video_root)
    category_to_ids: dict[str, list[int]] = defaultdict(list)
    for pair_id, pair in enumerate(pairs):
        category_to_ids[sample_category(pair[0])].append(pair_id)
    if args.category is not None:
        if args.category not in category_to_ids:
            raise ValueError(
                f"Unknown category {args.category!r}; available="
                f"{sorted(category_to_ids)}"
            )
        category_to_ids = defaultdict(
            list, {args.category: category_to_ids[args.category]}
        )
    category_counts = {
        category: len(pair_ids)
        for category, pair_ids in sorted(category_to_ids.items())
    }
    quotas = largest_remainder_quotas(category_counts, args.num_pairs)

    selection_rng = random.Random(args.seed)
    selected_by_category: dict[str, list[int]] = {}
    for category in sorted(category_to_ids):
        candidates = list(category_to_ids[category])
        selection_rng.shuffle(candidates)
        selected_by_category[category] = candidates[: quotas[category]]

    selected_pair_ids = [
        pair_id
        for category in sorted(selected_by_category)
        for pair_id in selected_by_category[category]
    ]
    order_rng = random.Random(args.seed + 104_729)
    order_rng.shuffle(selected_pair_ids)

    random_negative_for_positive: dict[int, int] = {}
    for category in sorted(selected_by_category):
        source_ids = list(selected_by_category[category])
        deranged = derange_pair_ids(
            source_ids,
            random.Random(args.seed + 7_919 + sum(map(ord, category))),
        )
        random_negative_for_positive.update(dict(zip(source_ids, deranged)))

    matched_windows = []
    random_windows = []
    for start in range(0, len(selected_pair_ids), args.pairs_per_window):
        positive_ids = selected_pair_ids[start : start + args.pairs_per_window]
        matched_windows.append(
            {
                "window_id": len(matched_windows),
                "positive_pair_ids": positive_ids,
                "negative_pair_ids": list(positive_ids),
            }
        )
        random_windows.append(
            {
                "window_id": len(random_windows),
                "positive_pair_ids": positive_ids,
                "negative_pair_ids": [
                    random_negative_for_positive[pair_id] for pair_id in positive_ids
                ],
            }
        )

    selected_negative_ids = sorted(random_negative_for_positive.values())
    if selected_negative_ids != sorted(selected_pair_ids):
        raise RuntimeError("Derangement changed the selected negative multiset")
    collisions = sum(
        source == target for source, target in random_negative_for_positive.items()
    )
    if collisions:
        raise RuntimeError(f"Derangement contains {collisions} true counterparts")
    for source, target in random_negative_for_positive.items():
        if sample_category(pairs[source][0]) != sample_category(pairs[target][1]):
            raise RuntimeError("Derangement crossed category boundaries")

    records = []
    for pair_id in selected_pair_ids:
        category = sample_category(pairs[pair_id][0])
        for role_index, role in enumerate(("positive", "negative")):
            sample = pairs[pair_id][role_index]
            records.append(
                {
                    "record_key": f"pair_{pair_id}:{role}",
                    "pair_id": pair_id,
                    "role": role,
                    "category": category,
                    "video": sample["video"][0],
                }
            )

    plan = {
        "seed": args.seed,
        "sampling_strategy": "unique_pairs_proportional_to_sqrt_category_size",
        "original_category_pair_counts": category_counts,
        "selected_category_pair_quotas": quotas,
        "selected_pair_ids_in_window_order": selected_pair_ids,
        "records": records,
        "random_negative_for_positive_pair_id": {
            str(key): value for key, value in sorted(random_negative_for_positive.items())
        },
        "derangement_audit": {
            "same_record_multiset": True,
            "same_category": True,
            "true_counterpart_collisions": collisions,
        },
        "matched_windows": matched_windows,
        "random_windows": random_windows,
    }
    return pairs, plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GradientCountSketch:
    """A deterministic CountSketch of a fixed ordered parameter vector."""

    def __init__(
        self,
        named_parameters: list[tuple[str, torch.nn.Parameter]],
        dimension: int,
        seed: int,
        device: torch.device,
    ) -> None:
        self.named_parameters = named_parameters
        self.dimension = dimension
        self.device = device
        self.indices_and_signs: list[tuple[torch.Tensor, torch.Tensor]] = []
        offset = 0
        for _, parameter in self.named_parameters:
            coordinates = torch.arange(
                offset,
                offset + parameter.numel(),
                dtype=torch.int64,
                device=device,
            )
            buckets = torch.remainder(
                coordinates * 1_103_515_245 + seed * 97_531 + 12_345,
                2_147_483_647,
            ).remainder(dimension)
            signs = (
                torch.remainder(
                    torch.div(
                        coordinates * 214_013 + seed * 2_531_011,
                        65_536,
                        rounding_mode="floor",
                    ),
                    2,
                )
                .mul(2)
                .sub(1)
                .to(torch.int8)
            )
            self.indices_and_signs.append((buckets, signs))
            offset += parameter.numel()
        self.total_parameters = offset

    @torch.no_grad()
    def from_current_gradients(self) -> torch.Tensor:
        sketch = torch.zeros(
            self.dimension, device=self.device, dtype=torch.float32
        )
        missing = []
        for (name, parameter), (buckets, signs) in zip(
            self.named_parameters, self.indices_and_signs
        ):
            if parameter.grad is None:
                missing.append(name)
                continue
            values = parameter.grad.detach().reshape(-1).float()
            sketch.scatter_add_(0, buckets, values * signs)
        if missing:
            raise RuntimeError(
                f"Missing gradients for {len(missing)} trainable tensors: {missing[:5]}"
            )
        if not torch.isfinite(sketch).all():
            raise FloatingPointError("Gradient sketch contains non-finite values")
        if sketch.norm().item() == 0.0:
            raise RuntimeError("Gradient sketch is identically zero")
        return sketch


def load_model_and_tokenizer(args: argparse.Namespace, accelerator: Accelerator):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    num_image_token = calculate_num_image_tokens(config)
    image_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["vision_model", "mlp1"],
    )
    accelerator.print(f"Loading fixed InternVL2.5 adapter on {accelerator.device}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        quantization_config=quantization_config,
        device_map={"": accelerator.device},
    )
    base_model.img_context_token_id = image_context_token_id
    base_model.system_message = SYSTEM_MESSAGE
    base_model.config.use_cache = False
    base_model.language_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )
    base_model.vision_model.requires_grad_(False)
    model = PeftModel.from_pretrained(
        base_model, args.adapter_path, is_trainable=True
    )
    # Hugging Face activates gradient checkpointing only in training mode.
    # Keep that memory-saving path active, but remove every Dropout source so
    # repeated gradients at this fixed checkpoint remain deterministic.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    named_parameters = sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    if not named_parameters:
        raise RuntimeError("Loaded adapter exposes no trainable parameters")
    if any("lora_" not in name for name, _ in named_parameters):
        unexpected = [name for name, _ in named_parameters if "lora_" not in name]
        raise RuntimeError(f"Unexpected trainable non-LoRA tensors: {unexpected[:5]}")
    return model, tokenizer, num_image_token, named_parameters


def compute_local_gradients(
    args: argparse.Namespace,
    accelerator: Accelerator,
    pairs: list[list[dict]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    model, tokenizer, num_image_token, named_parameters = load_model_and_tokenizer(
        args, accelerator
    )
    sketcher = GradientCountSketch(
        named_parameters, args.sketch_dim, args.seed + 31_337, accelerator.device
    )
    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable LoRA tensors={len(named_parameters)}, "
            f"parameters={sketcher.total_parameters:,}, sketch_dim={args.sketch_dim}"
        )

    assigned_records = [
        record
        for index, record in enumerate(plan["records"])
        if index % accelerator.num_processes == accelerator.process_index
    ]
    results = []
    progress = tqdm(
        assigned_records,
        desc=f"Rank {accelerator.process_index} gradients",
        disable=not accelerator.is_local_main_process,
    )
    for record in progress:
        role_index = 0 if record["role"] == "positive" else 1
        sample = pairs[record["pair_id"]][role_index]
        model.zero_grad(set_to_none=True)
        batch = prepare_training_sample(
            sample,
            tokenizer,
            args.video_root,
            args.num_frames,
            num_image_token,
            args.max_seq_length,
            device=accelerator.device,
        )
        # The model is intentionally not passed through accelerator.prepare():
        # DDP would all-reduce the per-record gradients that this analysis must
        # keep independent.  Retain the formal training run's bf16 autocast
        # explicitly instead.
        with accelerator.autocast():
            outputs = model(**batch)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss for {record['record_key']}: {loss.item()}"
            )
        loss.backward()
        sketch = sketcher.from_current_gradients()
        results.append(
            {
                **record,
                "loss": float(loss.detach().float().item()),
                "gradient_sketch_norm": float(sketch.norm().item()),
                "gradient_sketch": sketch.cpu().tolist(),
            }
        )
        progress.set_postfix(loss=f"{results[-1]['loss']:.4f}")
        del batch, outputs, loss, sketch
    model.zero_grad(set_to_none=True)
    return {
        "rank": accelerator.process_index,
        "world_size": accelerator.num_processes,
        "trainable_parameter_names": [name for name, _ in named_parameters],
        "trainable_parameter_count": sketcher.total_parameters,
        "records": results,
    }


def reconstruct_windows(
    plan: dict[str, Any], record_results: dict[str, dict[str, Any]], sketch_dim: int
) -> dict[str, list[dict[str, Any]]]:
    conditions: dict[str, list[dict[str, Any]]] = {}
    for condition, window_key in (
        ("matched", "matched_windows"),
        ("random", "random_windows"),
    ):
        condition_windows = []
        for window in plan[window_key]:
            record_keys = []
            for pair_id in window["positive_pair_ids"]:
                record_keys.append(f"pair_{pair_id}:positive")
            for pair_id in window["negative_pair_ids"]:
                record_keys.append(f"pair_{pair_id}:negative")
            sketches = np.asarray(
                [record_results[key]["gradient_sketch"] for key in record_keys],
                dtype=np.float64,
            )
            # The sign is the gradient-descent update direction.  Dividing by
            # eight matches mean-loss accumulation and does not affect direction.
            update = -sketches.mean(axis=0)
            norm = float(np.linalg.norm(update))
            if not np.isfinite(norm) or norm == 0.0:
                raise RuntimeError(
                    f"Invalid {condition} window norm for {window['window_id']}: {norm}"
                )
            normalized = update / norm
            if normalized.shape != (sketch_dim,):
                raise RuntimeError("Unexpected reconstructed sketch shape")
            condition_windows.append(
                {
                    **window,
                    "record_keys": record_keys,
                    "pre_optimizer_update_sketch": update.tolist(),
                    "pre_optimizer_update_sketch_norm": norm,
                    "normalized_update_sketch": normalized.tolist(),
                }
            )
        conditions[condition] = condition_windows
    return conditions


def add_shared_projection(conditions: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ordered = conditions["matched"] + conditions["random"]
    matrix = np.asarray(
        [window["normalized_update_sketch"] for window in ordered], dtype=np.float64
    )
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    basis = right_vectors[:2].copy()
    coordinates = matrix @ basis.T
    for component in range(2):
        pivot = int(np.argmax(np.abs(coordinates[:, component])))
        if coordinates[pivot, component] < 0:
            basis[component] *= -1
            coordinates[:, component] *= -1
    for window, coordinate in zip(ordered, coordinates):
        window["shared_uncentered_svd_coordinate"] = coordinate.tolist()
    explained_energy = singular_values**2 / np.sum(singular_values**2)
    return {
        "method": "shared_uncentered_svd_through_origin",
        "basis": basis.tolist(),
        "singular_values": singular_values.tolist(),
        "projected_energy_fraction_first_two": float(explained_energy[:2].sum()),
    }


def render_figure(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    vector_path = output_dir / "gradient_direction_vectors.json"
    with vector_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    conditions = payload["window_updates"]
    all_coordinates = np.asarray(
        [
            window["shared_uncentered_svd_coordinate"]
            for condition in ("matched", "random")
            for window in conditions[condition]
        ],
        dtype=np.float64,
    )
    limit = max(0.05, float(np.abs(all_coordinates).max()) * 1.15)
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), sharex=True, sharey=True)
    panels = (
        ("matched", "Matched counterpart binding", "#2C6EBA"),
        ("random", "Random within-category binding", "#D97706"),
    )
    for axis, (condition, title, color) in zip(axes, panels):
        axis.axhline(0.0, color="#B7B7B7", linewidth=0.7, zorder=0)
        axis.axvline(0.0, color="#B7B7B7", linewidth=0.7, zorder=0)
        for window in conditions[condition]:
            x_value, y_value = window["shared_uncentered_svd_coordinate"]
            axis.annotate(
                "",
                xy=(x_value, y_value),
                xytext=(0.0, 0.0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "alpha": 0.62,
                    "linewidth": 1.05,
                    "mutation_scale": 8,
                },
            )
        axis.scatter([0.0], [0.0], s=14, color="#222222", zorder=3)
        axis.set_title(title, fontsize=10)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Shared update axis 1", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Shared update axis 2", fontsize=9)
    figure.tight_layout(w_pad=1.4)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "gradient_direction_figure.pdf", bbox_inches="tight")
    figure.savefig(
        output_dir / "gradient_direction_figure.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> None:
    pairs, plan = build_analysis_plan(args)
    print(
        "Analysis plan: "
        f"unique_pairs={args.num_pairs}, records={2 * args.num_pairs}, "
        f"windows={args.num_pairs // args.pairs_per_window}, "
        f"pairs_per_window={args.pairs_per_window}, quotas="
        f"{plan['selected_category_pair_quotas']}, "
        "random_true_counterpart_collisions=0"
    )
    if args.preflight_only:
        return

    accelerator = Accelerator(mixed_precision="bf16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gradient analysis")
    torch.cuda.set_device(accelerator.local_process_index)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    local_payload = compute_local_gradients(args, accelerator, pairs, plan)
    rank_path = args.work_dir / f"rank_{accelerator.process_index}.json"
    rank_path.write_text(
        json.dumps(local_payload, ensure_ascii=False), encoding="utf-8"
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        rank_payloads = []
        for rank in range(accelerator.num_processes):
            path = args.work_dir / f"rank_{rank}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing rank output: {path}")
            rank_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        reference_names = rank_payloads[0]["trainable_parameter_names"]
        if any(
            payload["trainable_parameter_names"] != reference_names
            for payload in rank_payloads[1:]
        ):
            raise RuntimeError("Trainable parameter ordering differs across ranks")
        record_results = {
            record["record_key"]: record
            for payload in rank_payloads
            for record in payload["records"]
        }
        expected_keys = {record["record_key"] for record in plan["records"]}
        if set(record_results) != expected_keys:
            missing = sorted(expected_keys - set(record_results))
            extra = sorted(set(record_results) - expected_keys)
            raise RuntimeError(f"Record merge mismatch; missing={missing}, extra={extra}")
        window_updates = reconstruct_windows(plan, record_results, args.sketch_dim)
        projection = add_shared_projection(window_updates)
        adapter_file = args.adapter_path / "adapter_model.safetensors"
        output = {
            "analysis": "paired_data_binding_gradient_direction",
            "interpretation_scope": (
                "pre-optimizer mean-loss LoRA gradient directions at one fixed "
                "trained checkpoint; no optimizer step or parameter update"
            ),
            "model_path": str(args.model_path),
            "adapter_path": str(args.adapter_path),
            "adapter_sha256": sha256_file(adapter_file),
            "data_path": str(args.data_path),
            "num_frames": args.num_frames,
            "max_seq_length": args.max_seq_length,
            "sketch": {
                "type": "deterministic_countsketch",
                "dimension": args.sketch_dim,
                "seed": args.seed + 31_337,
                "full_trainable_parameter_count": rank_payloads[0][
                    "trainable_parameter_count"
                ],
                "trainable_tensor_count": len(reference_names),
                "trainable_parameter_names": reference_names,
            },
            "plan": plan,
            "per_record_gradients": [
                record_results[record["record_key"]] for record in plan["records"]
            ],
            "window_updates": window_updates,
            "projection": projection,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "gradient_direction_vectors.json"
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        accelerator.print(f"Saved gradient analysis to {output_path}")
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.render_only:
        render_figure(args.output_dir)
        print(f"Rendered gradient-direction figure in {args.output_dir}")
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_analysis(args)


if __name__ == "__main__":
    main()
