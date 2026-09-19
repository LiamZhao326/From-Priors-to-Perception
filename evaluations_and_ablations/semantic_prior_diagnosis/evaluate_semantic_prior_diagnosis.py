#!/usr/bin/env python3
"""Evaluate Observation--Verdict errors on PriorPair negative counterparts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PACKAGE_ROOT / "priorpair_benchmark" / "metrics"
if str(METRICS_DIR) not in sys.path:
    sys.path.insert(0, str(METRICS_DIR))

from priorpair_metrics_common import (  # noqa: E402
    ANTI_PHYSICS_CATEGORIES,
    get_category,
    parse_task_verdict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--phyar-results", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="RAS_API_KEY")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible API base URL (or set OPENAI_BASE_URL).",
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_text_block(markdown: str, heading: str) -> str:
    position = markdown.find(heading)
    if position < 0:
        raise ValueError(f"Prompt file is missing heading: {heading}")
    match = re.search(
        r"```text\s*\n(.*?)\n```",
        markdown[position + len(heading) :],
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"No fenced text block found after heading: {heading}")
    return match.group(1).strip()


def load_prompts(path: Path) -> dict[str, str]:
    markdown = path.read_text(encoding="utf-8")
    return {
        "Anti-physics": extract_text_block(
            markdown, "## I–III: Anti-physics Observation Judge"
        ),
        "Counter-intuitive": extract_text_block(
            markdown, "## IV: Counter-intuitive Observation Judge"
        ),
    }


def load_results(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or type(record.get("sample_index")) is not int:
            raise ValueError(f"{path}: invalid result record")
        index = record["sample_index"]
        if index in indexed:
            raise ValueError(f"{path}: duplicate sample_index={index}")
        for field in ("video", "prompt", "ground_truth", "model_output"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{path}: sample_index={index} has invalid {field}")
        indexed[index] = record
    return indexed


def extract_section(text: str, heading: str, following_heading: str) -> str:
    match = re.search(
        rf"(?:^|\n)\s*(?:\*\*)?{re.escape(heading)}(?:\*\*)?\s*:\s*"
        rf"(.*?)(?=\n\s*(?:\*\*)?{re.escape(following_heading)}"
        rf"(?:\*\*)?\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Could not extract {heading} section")
    return match.group(1).strip()


def extract_target_event(prompt: str) -> str:
    match = re.search(
        r"^\s*Target\s+event\s*:\s*(.+?)\s*$",
        prompt,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        raise ValueError("Counter-intuitive prompt has no Target event field")
    return match.group(1).strip()


def validate_judgment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Judge output is not a JSON object")
    if set(value) != {"reasoning", "observation_correct"}:
        raise ValueError("Judge output must contain reasoning and observation_correct")
    if not isinstance(value["reasoning"], str) or not value["reasoning"].strip():
        raise ValueError("Judge reasoning must be a non-empty string")
    if type(value["observation_correct"]) is not int or value[
        "observation_correct"
    ] not in (0, 1):
        raise ValueError("observation_correct must be integer 0 or 1")
    return {
        "reasoning": value["reasoning"].strip(),
        "observation_correct": value["observation_correct"],
    }


def call_judge(
    client: Any,
    model: str,
    prompt: str,
    max_retries: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=max_output_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Judge returned an empty response")
            return validate_judgment(json.loads(content))
        except Exception as error:
            last_error = error
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(2.0 * delay, 60.0)
    raise RuntimeError(f"Judge failed after {max_retries} attempts") from last_error


def format_judge_prompt(template: str, record: dict[str, Any], stream: str) -> str:
    values = {
        "ground_truth_observation": extract_section(
            record["ground_truth"], "Observation", "Attribution"
        ),
        "model_observation": extract_section(
            record["model_output"], "Observation", "Attribution"
        ),
    }
    if stream == "Counter-intuitive":
        values["target_event"] = extract_target_event(record["prompt"])
    return template.format(**values)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary, path)


def build_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stream in ("Anti-physics", "Counter-intuitive"):
        stream_samples = [sample for sample in samples if sample["stream"] == stream]
        summary[stream] = {}
        for system in ("base", "phyar"):
            correct_observation = sum(
                sample[system]["observation_correct"] for sample in stream_samples
            )
            correct_verdict = sum(
                sample[system]["verdict_correct"] for sample in stream_samples
            )
            conditional_wrong = sum(
                sample[system]["observation_correct"]
                and not sample[system]["verdict_correct"]
                for sample in stream_samples
            )
            count = len(stream_samples)
            summary[stream][system] = {
                "sample_count": count,
                "observation_correct_count": correct_observation,
                "observation_accuracy_percent": (
                    100.0 * correct_observation / count if count else None
                ),
                "verdict_correct_count": correct_verdict,
                "verdict_accuracy_percent": (
                    100.0 * correct_verdict / count if count else None
                ),
                "conditional_wrong_verdict_count": conditional_wrong,
                "conditional_correct_observation_count": correct_observation,
                "p_wrong_verdict_given_correct_observation_percent": (
                    100.0 * conditional_wrong / correct_observation
                    if correct_observation
                    else None
                ),
            }
    return summary


def main() -> None:
    args = parse_args()
    if args.max_retries < 1 or args.max_output_tokens < 1:
        raise ValueError("Retry and output-token limits must be positive")
    if args.request_timeout <= 0 or args.request_delay < 0:
        raise ValueError("Timeout must be positive and delay non-negative")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    prompts = load_prompts(args.prompt_file)
    base = load_results(args.base_results)
    phyar = load_results(args.phyar_results)
    if set(base) != set(phyar):
        raise ValueError("Base and PhyAR result files contain different sample indices")
    negative_indices = sorted(
        index for index, record in base.items() if "_negative" in record["video"]
    )
    if args.max_samples is not None:
        negative_indices = negative_indices[: args.max_samples]
    for index in negative_indices:
        if base[index]["video"] != phyar[index]["video"]:
            raise ValueError(f"sample_index={index} refers to different videos")

    if args.dry_run:
        print(
            f"Validated {len(negative_indices)} negative counterparts; "
            f"base={args.base_results}; phyar={args.phyar_results}"
        )
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set")
    if not args.base_url:
        raise RuntimeError("--base-url or OPENAI_BASE_URL is required")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("The openai package is required") from error

    completed: dict[int, dict[str, Any]] = {}
    if args.output_file.exists() and not args.overwrite:
        with args.output_file.open("r", encoding="utf-8") as input_file:
            previous = json.load(input_file)
        completed = {
            sample["sample_index"]: sample for sample in previous.get("samples", [])
        }

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
        max_retries=0,
    )
    for ordinal, index in enumerate(negative_indices, start=1):
        if index in completed:
            continue
        category = get_category(base[index]["video"])
        stream = (
            "Anti-physics"
            if category in ANTI_PHYSICS_CATEGORIES
            else "Counter-intuitive"
        )
        ground_truth_verdict = parse_task_verdict(
            base[index]["ground_truth"], category, base[index]["prompt"]
        )
        if ground_truth_verdict is None:
            raise ValueError(f"Could not parse ground-truth verdict at index {index}")
        sample: dict[str, Any] = {
            "sample_index": index,
            "video": base[index]["video"],
            "category": category,
            "stream": stream,
            "ground_truth_verdict": ground_truth_verdict,
        }
        for system, record in (("base", base[index]), ("phyar", phyar[index])):
            judgment = call_judge(
                client,
                args.model,
                format_judge_prompt(prompts[stream], record, stream),
                args.max_retries,
                args.max_output_tokens,
            )
            prediction = parse_task_verdict(
                record["model_output"], category, record["prompt"]
            )
            if prediction is None:
                raise ValueError(
                    f"Could not parse {system} verdict at sample_index={index}"
                )
            sample[system] = {
                "observation_correct": bool(judgment["observation_correct"]),
                "predicted_verdict": prediction,
                "verdict_correct": prediction == ground_truth_verdict,
            }
            if args.request_delay:
                time.sleep(args.request_delay)
        completed[index] = sample
        ordered = [completed[key] for key in sorted(completed)]
        payload = {
            "schema_version": 1,
            "scope": "All prior-violating negative counterparts in PriorPair.",
            "sources": {
                "base": str(args.base_results.resolve()),
                "phyar": str(args.phyar_results.resolve()),
            },
            "evaluation_metadata": {
                "judge_model": args.model,
                "prompt_file": str(args.prompt_file.resolve()),
            },
            "summary": build_summary(ordered),
            "samples": ordered,
        }
        atomic_write_json(args.output_file, payload)
        print(f"[{ordinal}/{len(negative_indices)}] sample_index={index}", flush=True)


if __name__ == "__main__":
    main()
