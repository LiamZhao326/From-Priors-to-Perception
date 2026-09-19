import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator
from decord import VideoReader, cpu
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

from intphys2_videollama3_infer import (
    atomic_write_json,
    append_rank_error,
    append_rank_result,
    calculate_summary,
    get_prompt,
    is_exact_prediction,
    load_rank_results,
    load_metadata,
    merge_rank_results,
    parse_prediction,
    prepare_output_paths,
    selected_adapter_path,
)


SYSTEM_MESSAGE = (
    "You are a multimodal video analysis assistant, required to answer "
    "user questions based on temporal video clips."
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate InternVL2.5-8B Base or the selected PriorPair PhyAR "
            "adapter on the IntPhys 2 Main set."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--weights",
        choices=("base", "phyar"),
        default="phyar",
        help="Evaluate either the original backbone or a PEFT adapter.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Used only when --weights=phyar.",
    )
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Defaults to 32 for Direct and 512 for VARC.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("direct", "varc"),
        default="direct",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Print a short inference check without writing result files.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Repair the difficulty fields in an existing complete result "
            "file and regenerate its summary without loading the model."
        ),
    )
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 32 if args.prompt_mode == "direct" else 512
    return args


def validate_args(args):
    metadata_path = args.data_root / "metadata.csv"
    videos_dir = args.data_root / "Videos"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {videos_dir}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    adapter_path = selected_adapter_path(args)
    if args.weights == "phyar" and adapter_path is None:
        raise ValueError("--adapter-path is required when --weights=phyar")
    if adapter_path is not None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        for file_name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter_path / file_name).is_file():
                raise FileNotFoundError(
                    f"Adapter file not found: {adapter_path / file_name}"
                )

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.smoke:
        if args.summary_only:
            raise ValueError("--smoke and --summary-only are mutually exclusive")
        if args.output_file is not None:
            raise ValueError("--smoke must not be combined with --output-file")
    elif args.output_file is None:
        raise ValueError("--output-file is required unless --smoke is used")


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(
                lambda image: image.convert("RGB")
                if image.mode != "RGB"
                else image
            ),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_frame_indices(frame_count, num_segments):
    if frame_count <= 0:
        raise ValueError("Video contains no frames")
    max_frame = frame_count - 1
    segment_size = float(max_frame) / num_segments
    return np.asarray(
        [
            int(segment_size / 2 + np.round(segment_size * index))
            for index in range(num_segments)
        ],
        dtype=np.int64,
    )


def load_video(video_path, num_frames, input_size=448):
    video_reader = VideoReader(
        str(video_path),
        ctx=cpu(0),
        num_threads=1,
    )
    frame_indices = get_frame_indices(len(video_reader), num_frames)
    frames = video_reader.get_batch(frame_indices).asnumpy()
    transform = build_transform(input_size)
    pixel_values = torch.stack(
        [transform(Image.fromarray(frame).convert("RGB")) for frame in frames]
    )
    return pixel_values, [1] * num_frames


def load_model(args, accelerator):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map={"": accelerator.device},
    ).eval()
    base_model.system_message = SYSTEM_MESSAGE

    adapter_path = selected_adapter_path(args)
    if adapter_path is not None:
        accelerator.print(f"Loading PEFT adapter: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        ).eval()
    else:
        model = base_model
    return model, tokenizer


def generate_one(row, args, model, tokenizer, device):
    video_path = args.data_root / row["file_name"]
    pixel_values, num_patches_list = load_video(video_path, args.num_frames)
    pixel_values = pixel_values.to(
        device=device,
        dtype=torch.bfloat16,
        non_blocking=True,
    )
    video_prefix = "".join(
        f"Frame{index + 1}: <image>\n" for index in range(args.num_frames)
    )
    question = video_prefix + get_prompt(args.prompt_mode)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    with torch.inference_mode():
        answer = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
        )
    response = answer.strip()
    prediction = parse_prediction(response, args.prompt_mode)
    del pixel_values
    return {
        "row_index": row["row_index"],
        "scene_index": int(row["SceneIndex"]),
        "name": row["name"],
        "file_name": row["file_name"],
        "game_name": row.get("game_name", ""),
        "condition": row["condition"],
        "environment": row.get("env", ""),
        "type": row["type"],
        "difficulty": row["normalized_difficulty"],
        "metadata_difficulty": row["Difficulty"],
        "camera": row["Camera"],
        "target": row["target"],
        "prediction": prediction,
        "correct": prediction == row["target"],
        "exact_prediction": is_exact_prediction(response, args.prompt_mode),
        "model_output": response,
    }


def finalize_existing_results(args):
    if not args.output_file.is_file():
        raise FileNotFoundError(
            f"Existing result file not found: {args.output_file}"
        )
    rows = load_metadata(args.data_root, args.max_samples)
    with args.output_file.open("r", encoding="utf-8") as stream:
        results = json.load(stream)
    if not isinstance(results, list):
        raise TypeError("Existing IntPhys2 result must be a JSON list")
    indices = [item.get("row_index") for item in results]
    if indices != list(range(len(rows))):
        raise RuntimeError(
            "Existing results do not cover every metadata row exactly once"
        )

    repaired = []
    for result, row in zip(results, rows):
        if result.get("file_name") != row["file_name"]:
            raise RuntimeError(
                f"Result/metadata mismatch at row {row['row_index']}"
            )
        item = dict(result)
        item["difficulty"] = row["normalized_difficulty"]
        item["metadata_difficulty"] = row["Difficulty"]
        repaired.append(item)

    manifest_path = Path(f"{args.output_file}.manifest.json")
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if isinstance(manifest.get("artifact_sha256"), dict):
            args._artifact_hashes = dict(manifest["artifact_sha256"])

    summary = calculate_summary(repaired, args)
    summary_path = args.output_file.with_name(
        f"{args.output_file.stem}_summary.json"
    )
    atomic_write_json(args.output_file, repaired)
    atomic_write_json(summary_path, summary)
    overall = summary["metrics"]["overall"]["All"]
    print(
        f"Repaired {len(repaired)} results in {args.output_file}; "
        f"accuracy={overall['accuracy']:.4f}%; "
        f"class_gap={summary['class_gap']:.4f}; "
        f"invalid={overall['invalid_outputs']}; summary={summary_path}"
    )


def main():
    args = parse_args()
    args.script_path = Path(__file__)
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.summary_only:
        finalize_existing_results(args)
        return

    accelerator = Accelerator()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(accelerator.local_process_index)

    rows = load_metadata(args.data_root, args.max_samples)
    accelerator.print(
        f"Loaded {len(rows)} IntPhys 2 sample(s); weights={args.weights}; "
        f"prompt={args.prompt_mode}; frames={args.num_frames}; "
        f"max_new_tokens={args.max_new_tokens}"
    )
    local_output_path = prepare_output_paths(args, accelerator)
    model, tokenizer = load_model(args, accelerator)

    if args.smoke:
        for row in rows[: min(2, len(rows))]:
            result = generate_one(
                row, args, model, tokenizer, accelerator.device
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    local_results = load_rank_results(local_output_path)
    completed_indices = {item["row_index"] for item in local_results}
    with accelerator.split_between_processes(rows) as local_rows:
        expected_local_indices = {row["row_index"] for row in local_rows}
        unexpected_indices = completed_indices - expected_local_indices
        if unexpected_indices:
            raise RuntimeError(
                "Partial rank output is incompatible with the current world "
                f"size or row split: {sorted(unexpected_indices)[:10]}"
            )
        pending_rows = [
            row for row in local_rows if row["row_index"] not in completed_indices
        ]
        progress = tqdm(
            pending_rows,
            desc=f"IntPhys 2 rank {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        )
        for row in progress:
            try:
                result = generate_one(
                    row, args, model, tokenizer, accelerator.device
                )
            except Exception as error:
                append_rank_error(local_output_path, row, error)
                accelerator.print(
                    f"IntPhys2 row {row['row_index']} failed: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            append_rank_result(local_output_path, result)
            local_results.append(result)
            progress.set_postfix(
                accuracy=(
                    f"{100.0 * sum(x['correct'] for x in local_results) / len(local_results):.2f}%"
                ),
                invalid=sum(x["prediction"] is None for x in local_results),
            )
    accelerator.wait_for_everyone()

    merge_error = None
    if accelerator.is_main_process:
        try:
            results, summary, summary_path = merge_rank_results(
                args, rows, accelerator
            )
        except Exception as error:
            merge_error = error
            print(
                "IntPhys2 run is incomplete; partial files were preserved for "
                f"resume: {type(error).__name__}: {error}"
            )
        else:
            overall = summary["metrics"]["overall"]["All"]
            class_gap = summary["class_gap"]
            class_gap_text = (
                f"{class_gap:.4f}" if class_gap is not None else "not_available"
            )
            print(
                f"Saved {len(results)} results to {args.output_file}; "
                f"accuracy={overall['accuracy']:.4f}%; "
                f"class_gap={class_gap_text}; "
                f"invalid={overall['invalid_outputs']}; summary={summary_path}"
            )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and merge_error is not None:
        raise RuntimeError(str(merge_error)) from merge_error


if __name__ == "__main__":
    main()
