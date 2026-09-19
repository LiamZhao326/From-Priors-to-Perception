"""Shared PriorPair inference utilities.

This module contains only dataset, result, and frame-index helpers. Model-
specific loading and generation remain in each ``priorpair_*_infer.py`` script.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


FULL_OAV_FORMAT_REQUIREMENT = (
    "Important output-format requirement: Your response must include all "
    "three requested fields—Observation, Attribution, and Verdict—in that "
    "order. Do not omit Observation or Attribution, and do not answer with "
    "only the Verdict."
)


def extract_turn(sample: Dict[str, Any], role: str) -> str:
    for turn in sample.get("conversations", []):
        if turn.get("from") == role:
            value = turn.get("value", "")
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError(f"Sample has no non-empty {role!r} conversation turn")


def extract_prompt(sample: Dict[str, Any]) -> str:
    prompt = extract_turn(sample, "human")
    if prompt.startswith("<video>\n"):
        prompt = prompt[len("<video>\n") :]
    elif prompt.startswith("<video>"):
        prompt = prompt[len("<video>") :]
    return prompt.strip()


def append_full_oav_format_requirement(prompt: str) -> str:
    """Append a format-only reminder without replacing the task prompt."""
    return f"{prompt.rstrip()}\n\n{FULL_OAV_FORMAT_REQUIREMENT}"


def extract_ground_truth(sample: Dict[str, Any]) -> str:
    return extract_turn(sample, "gpt").strip()


def sample_video_path(sample: Dict[str, Any], video_root: Path) -> Path:
    videos = sample.get("video")
    if not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("Each PriorPair sample must contain exactly one video")
    relative_path = videos[0]
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("PriorPair video path must be a non-empty string")
    return video_root / relative_path


def load_test_samples(
    test_data_path: Path,
    video_root: Path,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not test_data_path.is_file():
        raise FileNotFoundError(f"Test JSONL not found: {test_data_path}")
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video root not found: {video_root}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    samples: List[Dict[str, Any]] = []
    with test_data_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {test_data_path}:{line_number}"
                ) from error

            video_path = sample_video_path(sample, video_root)
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video not found at {test_data_path}:{line_number}: "
                    f"{video_path}"
                )
            extract_prompt(sample)
            extract_ground_truth(sample)
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break

    if not samples:
        raise ValueError(f"No samples loaded from {test_data_path}")
    return samples


def make_result(
    sample_index: int,
    sample: Dict[str, Any],
    model_output: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "sample_index": sample_index,
        "video": sample["video"][0],
        "prompt": extract_prompt(sample),
        "ground_truth": extract_ground_truth(sample),
        "model_output": model_output,
        "error": error,
    }


def uniform_frame_indices(frame_count: int, num_frames: int) -> List[int]:
    """Return exactly ``num_frames`` indices spanning the complete video."""
    if frame_count <= 0:
        raise ValueError("Video contains no frames")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        return [(frame_count - 1) // 2]
    last_index = frame_count - 1
    return [
        int(round(position * last_index / (num_frames - 1)))
        for position in range(num_frames)
    ]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, path)


def validate_results(
    results: Sequence[Dict[str, Any]],
    samples: Sequence[Dict[str, Any]],
    require_complete: bool = True,
) -> None:
    seen = set()
    for result in results:
        sample_index = result.get("sample_index")
        if not isinstance(sample_index, int):
            raise ValueError("Every result must have an integer sample_index")
        if sample_index in seen:
            raise ValueError(f"Duplicate result sample_index: {sample_index}")
        if not 0 <= sample_index < len(samples):
            raise ValueError(f"Out-of-range result sample_index: {sample_index}")
        seen.add(sample_index)

        sample = samples[sample_index]
        expected = make_result(sample_index, sample)
        for field in ("video", "prompt", "ground_truth"):
            if result.get(field) != expected[field]:
                raise ValueError(
                    f"Result {sample_index} has mismatched {field}"
                )

    if require_complete:
        expected_indices = set(range(len(samples)))
        if seen != expected_indices:
            missing = sorted(expected_indices - seen)
            raise ValueError(
                f"Incomplete result set: {len(seen)}/{len(samples)}; "
                f"missing indices={missing[:20]}"
            )


def prepare_new_output(output_path: Path, rank: Optional[int] = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite result: {output_path}")
    if rank is None:
        return output_path
    rank_path = rank_output_path(output_path, rank)
    if rank_path.exists():
        raise FileExistsError(f"Refusing to overwrite rank result: {rank_path}")
    return rank_path


def rank_output_path(output_path: Path, rank: int) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.rank_{rank}{output_path.suffix}"
    )


def assigned_indices(sample_count: int, rank: int, world_size: int) -> range:
    if not 0 <= rank < world_size:
        raise ValueError(f"Invalid rank/world_size: {rank}/{world_size}")
    return range(rank, sample_count, world_size)


def merge_rank_results(
    output_path: Path,
    world_size: int,
    samples: Sequence[Dict[str, Any]],
    timeout_seconds: int = 21600,
) -> List[Dict[str, Any]]:
    rank_paths = [rank_output_path(output_path, rank) for rank in range(world_size)]
    deadline = time.monotonic() + timeout_seconds
    while not all(path.is_file() for path in rank_paths):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in rank_paths if not path.is_file()]
            raise TimeoutError(f"Timed out waiting for rank files: {missing}")
        time.sleep(2)

    results: List[Dict[str, Any]] = []
    for rank_path in rank_paths:
        with rank_path.open("r", encoding="utf-8") as input_file:
            rank_results = json.load(input_file)
        if not isinstance(rank_results, list):
            raise ValueError(f"Rank result is not a list: {rank_path}")
        results.extend(rank_results)

    results.sort(key=lambda item: item["sample_index"])
    validate_results(results, samples, require_complete=True)
    atomic_write_json(output_path, results)
    for rank_path in rank_paths:
        rank_path.unlink()
    return results


def load_resumable_results(
    output_path: Path,
    samples: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as input_file:
        results = json.load(input_file)
    if not isinstance(results, list):
        raise ValueError(f"Existing result is not a list: {output_path}")
    validate_results(results, samples, require_complete=False)
    return {result["sample_index"]: result for result in results}


def save_result_map(
    output_path: Path,
    result_map: Dict[int, Dict[str, Any]],
) -> None:
    ordered = [result_map[index] for index in sorted(result_map)]
    atomic_write_json(output_path, ordered)


def successful_indices(results: Iterable[Dict[str, Any]]) -> set[int]:
    return {
        result["sample_index"]
        for result in results
        if result.get("error") is None and result.get("model_output") is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PriorPair inference inputs without loading a model."
    )
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument(
        "--test-data-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--expected-samples", type=int, default=336)
    args = parser.parse_args()

    samples = load_test_samples(args.test_data_path, args.video_root)
    if len(samples) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} samples, found {len(samples)}"
        )

    prompts = [extract_prompt(sample) for sample in samples]
    physical_count = sum(
        prompt.startswith("Task: Physical-validity verification")
        for prompt in prompts
    )
    target_event_count = sum(
        prompt.startswith("Task: Target-event occurrence verification")
        for prompt in prompts
    )
    if physical_count + target_event_count != len(samples):
        raise ValueError("Found a prompt outside the two approved task formats")

    dry_results = [
        make_result(index, sample, model_output="dry-validation")
        for index, sample in enumerate(samples)
    ]
    validate_results(dry_results, samples, require_complete=True)

    video_paths = [sample["video"][0] for sample in samples]
    duplicate_entries = len(video_paths) - len(set(video_paths))
    print(
        "PriorPair input validation passed: "
        f"samples={len(samples)}, physical={physical_count}, "
        f"target_event={target_event_count}, "
        f"duplicate_video_entries={duplicate_entries}"
    )


if __name__ == "__main__":
    main()
