import argparse
import hashlib
import json
import random
import re
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from decord import VideoReader, cpu
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)


PROMPT_HEADER = (
    "Select the best answer to the following multiple-choice question "
    "based on the video. Respond with only the letter (A, B, C, or D) "
    "of the correct option."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visual-only Video-MME evaluation for VideoLLaMA3-7B Base or "
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


def selected_adapter_path(args):
    return args.adapter_path if args.weights == "phyar" else None


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


def build_prompt(row):
    return "\n".join([PROMPT_HEADER, row["question"], *row["options"]])


def load_rows(data_file, max_questions=None):
    with data_file.open("r", encoding="utf-8") as stream:
        source_rows = json.load(stream)
    if not isinstance(source_rows, list):
        raise TypeError("Video-MME JSON must contain a list")
    if max_questions is not None:
        source_rows = source_rows[:max_questions]

    required = (
        "video_id",
        "duration",
        "domain",
        "sub_category",
        "videoID",
        "question_id",
        "task_type",
        "question",
        "options",
        "answer",
        "video_path",
    )
    rows = []
    for row_index, source in enumerate(source_rows):
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"Video-MME row {row_index} misses fields: {missing}")
        if len(source["options"]) != 4:
            raise ValueError(
                f"Video-MME row {row_index} does not have four options"
            )
        target = source["answer"].strip().upper()
        if target not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"Invalid target at Video-MME row {row_index}: {target}"
            )
        video_path = Path(source["video_path"])
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found at row {row_index}: {video_path}"
            )
        row = dict(source)
        row["row_index"] = row_index
        row["target"] = target
        row["video_path"] = str(video_path)
        row["prompt"] = build_prompt(row)
        rows.append(row)
    if not rows:
        raise ValueError("No Video-MME questions were loaded")
    return rows


def group_rows_by_video(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["video_path"], []).append(row)
    return list(groups.items())


def select_representative_smoke_rows(rows):
    selected = []
    used_videos = set()
    for duration in ("short", "medium", "long"):
        match = next(
            (
                row
                for row in rows
                if row["duration"].lower() == duration
                and row["video_path"] not in used_videos
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            used_videos.add(match["video_path"])
    if len(selected) != 3:
        raise RuntimeError(
            "Could not select one Video-MME smoke question for each duration"
        )
    return selected


def get_frame_indices(frame_count, num_frames):
    if frame_count <= 0:
        raise ValueError("Video contains no frames")
    max_frame = frame_count - 1
    segment_size = float(max_frame) / num_frames
    return np.asarray(
        [
            int(segment_size / 2 + np.round(segment_size * index))
            for index in range(num_frames)
        ],
        dtype=np.int64,
    )


def load_video(video_path, num_frames):
    video_reader = VideoReader(
        str(video_path),
        ctx=cpu(0),
        num_threads=1,
    )
    frame_indices = get_frame_indices(len(video_reader), num_frames)
    frames = video_reader.get_batch(frame_indices).asnumpy()
    frames = [frame for frame in frames.transpose(0, 3, 1, 2)]
    video_fps = float(video_reader.get_avg_fps())
    timestamps = [
        float(frame_index) / video_fps for frame_index in frame_indices
    ]
    return frames, timestamps


def parse_choice(response, options):
    matches = re.findall(r"(?<![A-Z])([A-D])(?![A-Z])", response.upper())
    if matches:
        return matches[0]
    normalized_response = " ".join(response.lower().split())
    for index, option in enumerate(options):
        option_text = re.sub(
            r"^[A-D][\.\)]\s*",
            "",
            option.strip(),
            flags=re.IGNORECASE,
        ).rstrip(".")
        if option_text and option_text.lower() in normalized_response:
            return "ABCD"[index]
    return None


def is_exact_choice(response):
    return response.strip().upper() in {"A", "B", "C", "D"}


def calculate_bucket(items):
    total = len(items)
    valid = sum(item["prediction"] is not None for item in items)
    correct = sum(item["correct"] for item in items)
    exact = sum(item["exact_choice"] for item in items)
    return {
        "total": total,
        "valid_outputs": valid,
        "invalid_outputs": total - valid,
        "exact_choice_outputs": exact,
        "correct": correct,
        "accuracy": round(100.0 * correct / total, 4) if total else None,
    }


def calculate_summary(results, args):
    buckets = {
        "overall": {"All": results},
        "duration": defaultdict(list),
        "domain": defaultdict(list),
        "task_type": defaultdict(list),
    }
    for item in results:
        buckets["duration"][item["duration"]].append(item)
        buckets["domain"][item["domain"]].append(item)
        buckets["task_type"][item["task_type"]].append(item)

    adapter_path = selected_adapter_path(args)
    summary = {
        "model_path": str(args.model_path.resolve()),
        "weights": args.weights,
        "adapter_path": (
            str(adapter_path.resolve()) if adapter_path is not None else None
        ),
        "data_file": str(args.data_file.resolve()),
        "evaluation": "Video-MME v1 visual-only",
        "prompt_header": PROMPT_HEADER,
        "num_frames": args.num_frames,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "artifact_sha256": artifact_hashes(args),
        "metrics": {},
    }
    for group_name, group_buckets in buckets.items():
        summary["metrics"][group_name] = {
            bucket_name: calculate_bucket(items)
            for bucket_name, items in sorted(group_buckets.items())
        }
    return summary


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(args):
    cached_hashes = getattr(args, "_artifact_hashes", None)
    if cached_hashes is not None:
        return dict(cached_hashes)
    script_path = Path(args.script_path).resolve()
    adapter_path = selected_adapter_path(args)
    hashes = {
        "data_file": sha256_file(args.data_file),
        "script": sha256_file(script_path),
        "model_config": sha256_file(args.model_path / "config.json"),
    }
    shared_script_path = Path(__file__).resolve()
    if shared_script_path != script_path:
        hashes["shared_videomme_module"] = sha256_file(shared_script_path)
    if adapter_path is not None:
        hashes["adapter_config"] = sha256_file(
            adapter_path / "adapter_config.json"
        )
        hashes["adapter_model"] = sha256_file(
            adapter_path / "adapter_model.safetensors"
        )
    args._artifact_hashes = dict(hashes)
    return hashes


def atomic_write_json(path, payload):
    temporary_path = Path(f"{path}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
    temporary_path.replace(path)


def manifest_path(output_file):
    return Path(f"{output_file}.manifest.json")


def build_manifest(args):
    adapter_path = selected_adapter_path(args)
    return {
        "benchmark": "Video-MME v1 visual-only",
        "model_path": str(args.model_path.resolve()),
        "weights": args.weights,
        "adapter_path": (
            str(adapter_path.resolve()) if adapter_path is not None else None
        ),
        "data_file": str(args.data_file.resolve()),
        "prompt_header": PROMPT_HEADER,
        "num_frames": args.num_frames,
        "max_new_tokens": args.max_new_tokens,
        "max_questions": args.max_questions,
        "seed": args.seed,
        "script_path": str(Path(args.script_path).resolve()),
        "artifact_sha256": artifact_hashes(args),
    }


def load_partial_results(partial_path, rows):
    if not partial_path.exists():
        return []
    results = []
    with partial_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Corrupt partial result {partial_path}:{line_number}: {error}"
                ) from error
    indices = [item.get("row_index") for item in results]
    if len(indices) != len(set(indices)):
        raise RuntimeError(
            f"Duplicate row_index values in partial result: {partial_path}"
        )
    row_by_index = {row["row_index"]: row for row in rows}
    for item in results:
        row = row_by_index.get(item.get("row_index"))
        if row is None or item.get("question_id") != row["question_id"]:
            raise RuntimeError(
                "Partial result does not match the current Video-MME rows: "
                f"{item.get('row_index')} / {item.get('question_id')}"
            )
    return results


def prepare_run(args, rows):
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_file.with_name(
        f"{args.output_file.stem}_summary.json"
    )
    if args.output_file.exists() or summary_path.exists():
        raise FileExistsError(
            "A completed output or summary already exists; refusing to resume "
            f"over it: {args.output_file}"
        )
    partial_path = Path(f"{args.output_file}.partial.jsonl")
    run_manifest_path = manifest_path(args.output_file)
    expected_manifest = build_manifest(args)
    if run_manifest_path.exists():
        with run_manifest_path.open("r", encoding="utf-8") as stream:
            existing_manifest = json.load(stream)
        if existing_manifest != expected_manifest:
            raise RuntimeError(
                "Existing partial results were produced with a different "
                f"configuration: {run_manifest_path}"
            )
    elif partial_path.exists():
        raise RuntimeError(
            "A legacy partial result exists without a run manifest; refusing "
            f"unsafe automatic resume: {partial_path}"
        )
    else:
        atomic_write_json(run_manifest_path, expected_manifest)
    return partial_path, load_partial_results(partial_path, rows)


def append_partial_result(partial_path, result):
    with partial_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        stream.flush()


def append_partial_error(partial_path, row, error):
    error_path = Path(f"{partial_path}.errors.jsonl")
    payload = {
        "row_index": row["row_index"],
        "question_id": row["question_id"],
        "video_path": row["video_path"],
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    with error_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def load_model(args, accelerator):
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["vision_encoder", "mm_projector"],
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
        device_map={"": accelerator.device},
    )
    adapter_path = selected_adapter_path(args)
    if adapter_path is not None:
        accelerator.print(f"Loading PEFT adapter: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )
    else:
        model = base_model
    model.eval()
    model.config.use_cache = True
    return model, processor


def generate_question(row, frames, timestamps, args, model, processor, device):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "timestamps": timestamps,
                    "num_frames": len(frames),
                },
                {"type": "text", "text": row["prompt"]},
            ],
        }
    ]
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {
        key: (
            value.to(
                device=device,
                dtype=(torch.bfloat16 if torch.is_floating_point(value) else value.dtype),
                non_blocking=True,
            )
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    generated_ids = (
        output_ids[:, input_length:]
        if output_ids.shape[1] > input_length
        else output_ids
    )
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
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
    processor,
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
    progress = tqdm(total=len(rows), desc="Video-MME VideoLLaMA3")
    progress.update(len(results))
    try:
        for video_path, video_rows in groups:
            try:
                frames, timestamps = load_video(video_path, args.num_frames)
            except Exception as error:
                if partial_path is None:
                    raise
                for row in video_rows:
                    append_partial_error(partial_path, row, error)
                continue
            if len(frames) != args.num_frames:
                raise RuntimeError(
                    f"Expected {args.num_frames} frames, got {len(frames)} "
                    f"for {video_path}"
                )
            for row in video_rows:
                try:
                    result = generate_question(
                        row,
                        frames,
                        timestamps,
                        args,
                        model,
                        processor,
                        accelerator.device,
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
            del frames, timestamps
    finally:
        progress.close()
    return results, partial_path


def save_results(args, rows, results, partial_path):
    results.sort(key=lambda item: item["row_index"])
    actual = [item["row_index"] for item in results]
    if actual != list(range(len(rows))):
        raise RuntimeError("Video-MME results do not cover every row exactly once")
    atomic_write_json(args.output_file, results)
    summary = calculate_summary(results, args)
    summary_path = args.output_file.with_name(
        f"{args.output_file.stem}_summary.json"
    )
    atomic_write_json(summary_path, summary)
    if partial_path is not None:
        partial_path.unlink()
        error_path = Path(f"{partial_path}.errors.jsonl")
        if error_path.exists():
            error_path.unlink()
        manifest_path(args.output_file).unlink()
    return summary, summary_path


def main():
    args = parse_args()
    args.script_path = Path(__file__)
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

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
    model, processor = load_model(args, accelerator)
    results, partial_path = run_evaluation(
        args,
        rows,
        model,
        processor,
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
