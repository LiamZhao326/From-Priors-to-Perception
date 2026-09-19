import argparse
import csv
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
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)


# Prompt 3 from the official IntPhys 2 MLLM evaluation code.
OFFICIAL_YES_NO_PROMPT = (
    "Analyze the physics simulations in the video. Determine if the video "
    "depicts object behavior that is consistent with Earth's physical laws. "
    "Ignore simulation quality and playback speed. CRITICAL OUTPUT "
    "INSTRUCTION: Your response must be only yes or no indicating whether "
    "the object behavior is consistent with Earth's physical laws: yes: "
    "Object behavior is consistent with Earth's physical laws. no: Object "
    "behavior is not consistent with Earth's physical laws. Please provide "
    "your analysis in the required format. Do not include any other text, "
    "explanation, context, preamble, or punctuation before or after the "
    "answer."
)

# Exact physical-validity prompt used by PriorPair for categories I--III.
VARC_PROMPT = '''Task: Physical-validity verification

Determine whether the events depicted in the video are physically valid or violate real-world physical, temporal, causal, thermodynamic, or spatial constraints.

Analyze the video using the following three fields:

Observation: Objectively describe the chronological sequence of events and the decisive visual evidence explicitly present in the video.

Attribution: Based on the Observation, identify the relevant physical, temporal, causal, thermodynamic, or spatial constraints, and explain whether the observed events comply with or violate them.

Verdict: Conclude with exactly either "The video is physically valid." or "The video is physically invalid."'''  # noqa: E501


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VideoLLaMA3-7B Base or the selected PriorPair PhyAR "
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
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 32 if args.prompt_mode == "direct" else 512
    return args


def selected_adapter_path(args):
    return args.adapter_path if args.weights == "phyar" else None


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
        if args.output_file is not None:
            raise ValueError("--smoke must not be combined with --output-file")
    elif args.output_file is None:
        raise ValueError("--output-file is required unless --smoke is used")


def load_metadata(data_root, max_samples=None):
    metadata_path = data_root / "metadata.csv"
    with metadata_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError("No IntPhys 2 metadata rows were loaded")

    required_fields = {
        "SceneIndex",
        "name",
        "file_name",
        "condition",
        "type",
        "Difficulty",
        "Camera",
    }
    missing_fields = required_fields - set(rows[0])
    if missing_fields:
        raise ValueError(
            f"IntPhys 2 metadata is missing fields: {sorted(missing_fields)}"
        )

    for row_index, row in enumerate(rows):
        video_path = data_root / row["file_name"]
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Missing video at metadata row {row_index}: {video_path}"
            )
        if "Impossible" in row["type"]:
            target = 0
        elif "Possible" in row["type"]:
            target = 1
        else:
            raise ValueError(
                f"Unknown IntPhys 2 type at row {row_index}: {row['type']!r}"
            )
        row["row_index"] = row_index
        row["target"] = target
        # The released metadata labels 172 complex-environment samples as
        # "Unknown", although the benchmark defines the Main set using only
        # Easy/Medium/Hard and characterizes complex environments as Hard.
        # Preserve the released value separately in each result record.
        row["normalized_difficulty"] = (
            "Hard" if row["Difficulty"] == "Unknown" else row["Difficulty"]
        )
    return rows


def get_prompt(prompt_mode):
    if prompt_mode == "direct":
        return OFFICIAL_YES_NO_PROMPT
    if prompt_mode == "varc":
        return VARC_PROMPT
    raise ValueError(f"Unsupported prompt mode: {prompt_mode}")


def parse_yes_no(response):
    matches = re.findall(r"\b(yes|no)\b", response.lower())
    if not matches:
        return None
    return 1 if matches[-1] == "yes" else 0


def parse_varc_verdict(response):
    verdict_matches = list(
        re.finditer(
            r"(?:\*\*)?\bverdict\b(?:\*\*)?\s*:?",
            response,
            flags=re.IGNORECASE,
        )
    )
    if verdict_matches:
        verdict_text = response[verdict_matches[-1].end() :].strip().lower()
    else:
        # A bare canonical verdict is still unambiguous. Do not otherwise
        # infer a label from Observation/Attribution wording: IntPhys2 VARC is
        # scored from the final semantic Verdict only.
        verdict_text = response.strip().lower().strip("*").strip()
        if verdict_text not in {
            "the video is physically valid.",
            "the video is physically valid",
            "the video is physically invalid.",
            "the video is physically invalid",
        }:
            return None
    if not verdict_text:
        return None

    canonical_invalid = re.findall(
        r"\bthe\s+video\s+is\s+physically\s+invalid\b",
        verdict_text,
    )
    canonical_valid = re.findall(
        r"\bthe\s+video\s+is\s+physically\s+valid\b",
        verdict_text,
    )
    if canonical_invalid or canonical_valid:
        if bool(canonical_invalid) == bool(canonical_valid):
            return None
        return 0 if canonical_invalid else 1

    invalid_patterns = (
        r"\bphysically\s+invalid\b",
        r"\bphysically\s+implausible\b",
        r"\bnot\s+physically\s+valid\b",
        r"\bnot\s+physically\s+plausible\b",
        r"\bphysically\s+impossible\b",
        r"\b(?:is|appears?)\s+forged\b",
        r"\bviolates?\b",
        r"\b(?:does|do|did)\s+not\s+compl(?:y|ies)\s+with\b",
        r"\bnot\s+consistent\s+with\b",
        r"\binconsistent\s+with\b",
    )
    valid_patterns = (
        r"\bphysically\s+valid\b",
        r"\bphysically\s+plausible\b",
        r"\bphysically\s+possible\b",
        r"\bnot\s+forged\b",
        r"\b(?:the|this)\s+video\s+is\s+real\b",
        r"\b(?:does|do|did)\s+not\s+violate\b",
        r"\bviolates?\s+no\b",
        r"\bno\s+(?:physical\s+)?violations?\b",
        r"\bcomplies?\s+with\b",
        r"\bconsistent\s+with\b",
    )
    invalid_text = re.sub(
        r"\b(?:does|do|did)\s+not\s+violate\b|"
        r"\bviolates?\s+no\b|\bno\s+(?:physical\s+)?violations?\b",
        "",
        verdict_text,
    )
    has_invalid = any(
        re.search(pattern, invalid_text) for pattern in invalid_patterns
    )
    valid_text = re.sub(r"\bnot\s+physically\s+valid\b", "", verdict_text)
    valid_text = re.sub(
        r"\bnot\s+physically\s+plausible\b",
        "",
        valid_text,
    )
    valid_text = re.sub(r"\binconsistent\s+with\b", "", valid_text)
    valid_text = re.sub(r"\bnot\s+consistent\s+with\b", "", valid_text)
    valid_text = re.sub(
        r"\b(?:does|do|did)\s+not\s+compl(?:y|ies)\s+with\b",
        "",
        valid_text,
    )
    has_valid = any(re.search(pattern, valid_text) for pattern in valid_patterns)
    if has_invalid == has_valid:
        return None
    return 0 if has_invalid else 1


def parse_prediction(response, prompt_mode):
    if prompt_mode == "direct":
        return parse_yes_no(response)
    return parse_varc_verdict(response)


def is_exact_prediction(response, prompt_mode):
    if prompt_mode == "direct":
        return response.strip().lower() in {"yes", "no"}
    verdict_match = re.search(
        r"(?is)(?:\*\*)?\bverdict\b(?:\*\*)?\s*:?\s*(.*?)\s*$",
        response,
    )
    if verdict_match is None:
        return False
    verdict = verdict_match.group(1).strip().strip("*").strip()
    return verdict in {
        "The video is physically valid.",
        "The video is physically invalid.",
    }


def calculate_bucket(items):
    total = len(items)
    valid = sum(item["prediction"] is not None for item in items)
    correct = sum(item["correct"] for item in items)
    exact = sum(item["exact_prediction"] for item in items)
    return {
        "total": total,
        "valid_outputs": valid,
        "invalid_outputs": total - valid,
        "exact_prediction_outputs": exact,
        "correct": correct,
        "accuracy": round(100.0 * correct / total, 4) if total else None,
    }


def calculate_summary(results, args):
    buckets = {
        "overall": {"All": results},
        "difficulty": defaultdict(list),
        "metadata_difficulty": defaultdict(list),
        "condition": defaultdict(list),
        "camera": defaultdict(list),
        "plausibility": defaultdict(list),
    }
    for item in results:
        buckets["difficulty"][item["difficulty"]].append(item)
        buckets["metadata_difficulty"][item["metadata_difficulty"]].append(item)
        buckets["condition"][item["condition"]].append(item)
        buckets["camera"][item["camera"]].append(item)
        label = "Possible" if item["target"] == 1 else "Impossible"
        buckets["plausibility"][label].append(item)

    adapter_path = selected_adapter_path(args)
    summary = {
        "model_path": str(args.model_path.resolve()),
        "weights": args.weights,
        "adapter_path": (
            str(adapter_path.resolve()) if adapter_path is not None else None
        ),
        "data_root": str(args.data_root.resolve()),
        "prompt_mode": args.prompt_mode,
        "prompt": get_prompt(args.prompt_mode),
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
    plausible = summary["metrics"]["plausibility"]
    if "Possible" in plausible and "Impossible" in plausible:
        possible_acc = plausible["Possible"]["accuracy"]
        impossible_acc = plausible["Impossible"]["accuracy"]
        summary["class_gap"] = round(abs(possible_acc - impossible_acc), 4)
    else:
        summary["class_gap"] = None
    return summary


def rank_output_path(output_file, process_index):
    return output_file.with_name(
        f"{output_file.stem}.rank_{process_index}.partial.jsonl"
    )


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
        "metadata_csv": sha256_file(args.data_root / "metadata.csv"),
        "script": sha256_file(script_path),
        "model_config": sha256_file(args.model_path / "config.json"),
    }
    shared_script_path = Path(__file__).resolve()
    if shared_script_path != script_path:
        hashes["shared_intphys_module"] = sha256_file(shared_script_path)
    if adapter_path is not None:
        hashes["adapter_config"] = sha256_file(
            adapter_path / "adapter_config.json"
        )
        hashes["adapter_model"] = sha256_file(
            adapter_path / "adapter_model.safetensors"
        )
    args._artifact_hashes = dict(hashes)
    return hashes


def manifest_path(output_file):
    return Path(f"{output_file}.manifest.json")


def build_manifest(args, accelerator):
    adapter_path = selected_adapter_path(args)
    return {
        "benchmark": "IntPhys2 Main",
        "num_processes": accelerator.num_processes,
        "model_path": str(args.model_path.resolve()),
        "weights": args.weights,
        "adapter_path": (
            str(adapter_path.resolve()) if adapter_path is not None else None
        ),
        "data_root": str(args.data_root.resolve()),
        "prompt_mode": args.prompt_mode,
        "prompt": get_prompt(args.prompt_mode),
        "num_frames": args.num_frames,
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "script_path": str(Path(args.script_path).resolve()),
        "artifact_sha256": artifact_hashes(args),
    }


def atomic_write_json(path, payload):
    temporary_path = Path(f"{path}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
    temporary_path.replace(path)


def prepare_output_paths(args, accelerator):
    if args.smoke:
        return None
    if accelerator.is_main_process:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    summary_path = args.output_file.with_name(
        f"{args.output_file.stem}_summary.json"
    )
    if args.output_file.exists() or summary_path.exists():
        raise FileExistsError(
            "A completed output or summary already exists; refusing to resume "
            f"over it: {args.output_file}"
        )

    run_manifest_path = manifest_path(args.output_file)
    expected_manifest = build_manifest(args, accelerator)
    if run_manifest_path.exists():
        with run_manifest_path.open("r", encoding="utf-8") as stream:
            existing_manifest = json.load(stream)
        if existing_manifest != expected_manifest:
            raise RuntimeError(
                "Existing partial results were produced with a different "
                f"configuration: {run_manifest_path}"
            )
    else:
        legacy_rank_paths = [
            rank_output_path(args.output_file, rank)
            for rank in range(accelerator.num_processes)
        ]
        if any(path.exists() for path in legacy_rank_paths):
            raise RuntimeError(
                "Partial rank output exists without a run manifest; refusing "
                "unsafe automatic resume"
            )
        if accelerator.is_main_process:
            atomic_write_json(run_manifest_path, expected_manifest)
    accelerator.wait_for_everyone()
    local_path = rank_output_path(args.output_file, accelerator.process_index)
    return local_path


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


def generate_one(row, args, model, processor, device):
    video_path = args.data_root / row["file_name"]
    prompt = get_prompt(args.prompt_mode)
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": str(video_path),
                        "nframes": args.num_frames,
                    },
                },
                {"type": "text", "text": prompt},
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
    prediction = parse_prediction(response, args.prompt_mode)
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


def load_rank_results(path):
    if not path.exists():
        return []
    results = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Corrupt partial result {path}:{line_number}: {error}"
                ) from error
    indices = [item.get("row_index") for item in results]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Duplicate row_index values in partial result: {path}")
    return results


def append_rank_result(path, result):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        stream.flush()


def append_rank_error(path, row, error):
    error_path = Path(f"{path}.errors.jsonl")
    payload = {
        "row_index": row["row_index"],
        "file_name": row["file_name"],
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    with error_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()


def merge_rank_results(args, rows, accelerator):
    all_results = []
    rank_paths = [
        rank_output_path(args.output_file, rank)
        for rank in range(accelerator.num_processes)
    ]
    for path in rank_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing rank output: {path}")
        with path.open("r", encoding="utf-8") as stream:
            all_results.extend(json.loads(line) for line in stream if line.strip())
    all_results.sort(key=lambda item: item["row_index"])
    actual_indices = [item["row_index"] for item in all_results]
    if actual_indices != list(range(len(rows))):
        raise RuntimeError(
            "Merged results do not cover every metadata row exactly once"
        )

    atomic_write_json(args.output_file, all_results)
    summary = calculate_summary(all_results, args)
    summary_path = args.output_file.with_name(
        f"{args.output_file.stem}_summary.json"
    )
    atomic_write_json(summary_path, summary)
    for path in rank_paths:
        path.unlink()
        error_path = Path(f"{path}.errors.jsonl")
        if error_path.exists():
            error_path.unlink()
    manifest_path(args.output_file).unlink()
    return all_results, summary, summary_path


def main():
    args = parse_args()
    args.script_path = Path(__file__)
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

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
    model, processor = load_model(args, accelerator)

    if args.smoke:
        for row in rows[: min(2, len(rows))]:
            result = generate_one(
                row, args, model, processor, accelerator.device
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
                    row, args, model, processor, accelerator.device
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
