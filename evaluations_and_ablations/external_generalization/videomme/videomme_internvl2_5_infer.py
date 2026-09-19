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

from videomme_videollama3_infer import (
    append_partial_error,
    append_partial_result,
    group_rows_by_video,
    is_exact_choice,
    load_rows,
    parse_choice,
    prepare_run,
    save_results,
    select_representative_smoke_rows,
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
            "Visual-only Video-MME evaluation for InternVL2.5-8B Base or "
            "the selected PriorPair PhyAR adapter."
        )
    )
    parser.add_argument("--data-file", type=Path, required=True)
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
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate three questions without writing result files.",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.data_file.is_file():
        raise FileNotFoundError(f"Video-MME data not found: {args.data_file}")
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
    if args.max_questions is not None and args.max_questions <= 0:
        raise ValueError("--max-questions must be positive")
    if args.smoke:
        if args.output_file is not None:
            raise ValueError("--smoke must not use --output-file")
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


def generate_question(
    row,
    pixel_values,
    num_patches_list,
    args,
    model,
    tokenizer,
):
    video_prefix = "".join(
        f"Frame{index + 1}: <image>\n" for index in range(args.num_frames)
    )
    question = video_prefix + row["prompt"]
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    with torch.inference_mode():
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
        ).strip()
    prediction = parse_choice(response, row["options"])
    return {
        "row_index": row["row_index"],
        "video_id": row["video_id"],
        "videoID": row["videoID"],
        "question_id": row["question_id"],
        "duration": row["duration"],
        "domain": row["domain"],
        "sub_category": row["sub_category"],
        "task_type": row["task_type"],
        "target": row["target"],
        "prediction": prediction,
        "correct": prediction == row["target"],
        "exact_choice": is_exact_choice(response),
        "model_output": response,
    }


def run_evaluation(
    args,
    rows,
    model,
    tokenizer,
    accelerator,
    partial_path=None,
    existing_results=None,
):
    results = list(existing_results or [])
    completed_indices = {item["row_index"] for item in results}
    pending_rows = [
        row for row in rows if row["row_index"] not in completed_indices
    ]
    groups = group_rows_by_video(pending_rows)
    progress = tqdm(total=len(rows), desc="Video-MME InternVL2.5")
    progress.update(len(results))
    try:
        for video_path, video_rows in groups:
            try:
                pixel_values, num_patches_list = load_video(
                    video_path, args.num_frames
                )
                pixel_values = pixel_values.to(
                    device=accelerator.device,
                    dtype=torch.bfloat16,
                    non_blocking=True,
                )
            except Exception as error:
                if partial_path is None:
                    raise
                for row in video_rows:
                    append_partial_error(partial_path, row, error)
                continue
            for row in video_rows:
                try:
                    result = generate_question(
                        row,
                        pixel_values,
                        num_patches_list,
                        args,
                        model,
                        tokenizer,
                    )
                except Exception as error:
                    if partial_path is None:
                        raise
                    append_partial_error(partial_path, row, error)
                    continue
                results.append(result)
                if partial_path is not None:
                    append_partial_result(partial_path, result)
                progress.update(1)
                progress.set_postfix(
                    accuracy=(
                        f"{100.0 * sum(x['correct'] for x in results) / len(results):.2f}%"
                    ),
                    invalid=sum(x["prediction"] is None for x in results),
                )
            del pixel_values
    finally:
        progress.close()
    return results, partial_path


def main():
    args = parse_args()
    args.script_path = Path(__file__)
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Use one process per Video-MME model run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(accelerator.local_process_index)

    rows = load_rows(args.data_file, args.max_questions)
    if args.smoke and args.max_questions is None:
        rows = select_representative_smoke_rows(rows)
    accelerator.print(
        f"Loaded {len(rows)} Video-MME question(s) from "
        f"{len(group_rows_by_video(rows))} video(s); weights={args.weights}; "
        f"visual-only; frames={args.num_frames}"
    )
    partial_path = None
    existing_results = []
    if not args.smoke:
        partial_path, existing_results = prepare_run(args, rows)
        accelerator.print(
            f"Resume state: completed={len(existing_results)}; "
            f"remaining={len(rows) - len(existing_results)}"
        )
    model, tokenizer = load_model(args, accelerator)
    results, partial_path = run_evaluation(
        args,
        rows,
        model,
        tokenizer,
        accelerator,
        partial_path=partial_path,
        existing_results=existing_results,
    )
    if args.smoke:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    summary, summary_path = save_results(args, rows, results, partial_path)
    overall = summary["metrics"]["overall"]["All"]
    print(
        f"Saved {len(results)} results to {args.output_file}; "
        f"accuracy={overall['accuracy']:.4f}%; "
        f"invalid={overall['invalid_outputs']}; summary={summary_path}"
    )


if __name__ == "__main__":
    main()
