#!/usr/bin/env python3
"""Evaluate PriorPair RAS with an OpenAI-compatible Gemini endpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from priorpair_metrics_common import (
    CATEGORIES,
    CATEGORY_SHORT_NAMES,
    GROUPS,
    get_model_name,
    load_and_validate_pairs,
    resolve_input_files,
    select_pairs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = (
    SCRIPT_DIR.parent.parent / "prompts" / "ras_evaluation_prompts.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use an OpenAI-compatible Gemini endpoint to assign hierarchical "
            "PriorPair Reasoning Alignment Scores, then report strict pair-wise RAS."
        )
    )
    parser.add_argument("--inputs", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("."),
    )
    parser.add_argument("--pattern", default="*_results.json")
    parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate at most this many new samples per input file.",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate only these sample indices; intended for targeted smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing evaluated output instead of resuming it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "After bounded retries, log a failed sample and continue. "
            "The missing sample remains pending for a later resume run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and prompt extraction without calling the API or writing files.",
    )
    return parser.parse_args()


def extract_text_block(markdown: str, heading: str) -> str:
    heading_index = markdown.find(heading)
    if heading_index < 0:
        raise ValueError(f"Prompt file is missing heading: {heading}")
    following = markdown[heading_index + len(heading) :]
    match = re.search(r"```text\s*\n(.*?)\n```", following, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No fenced text block found after heading: {heading}")
    block = match.group(1).strip()
    if not block:
        raise ValueError(f"Empty prompt block after heading: {heading}")
    return block


def load_prompts(prompt_file: Path) -> dict[str, str]:
    markdown = prompt_file.read_text(encoding="utf-8")
    return {
        "Anti-physics": extract_text_block(
            markdown, "## I–III: Physical-validity RAS system prompt"
        ),
        "Counter-intuitive": extract_text_block(
            markdown, "## IV: Target-event occurrence RAS system prompt"
        ),
        "input_template": extract_text_block(
            markdown, "## Shared evaluation input template"
        ),
    }


def validate_judge_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Judge output is not a JSON object")
    expected_keys = {"reasoning", "accuracy", "score"}
    if set(value) != expected_keys:
        raise ValueError(
            f"Judge output keys must be exactly {sorted(expected_keys)}; "
            f"received {sorted(value)}"
        )
    reasoning = value["reasoning"]
    accuracy = value["accuracy"]
    score = value["score"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("evaluation.reasoning must be a non-empty string")
    if type(accuracy) is not int or accuracy not in (0, 1):
        raise ValueError("evaluation.accuracy must be integer 0 or 1")
    if type(score) is not int or score not in range(1, 6):
        raise ValueError("evaluation.score must be an integer from 1 to 5")
    allowed_scores = (1, 2) if accuracy == 0 else (3, 4, 5)
    if score not in allowed_scores:
        raise ValueError(
            f"Judge contract violated: accuracy={accuracy}, score={score}"
        )
    return {
        "reasoning": reasoning.strip(),
        "accuracy": accuracy,
        "score": score,
    }


def call_gemini(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=max_output_tokens,
            )
            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("Gemini returned an empty response")
            return validate_judge_result(json.loads(response_text))
        except Exception as error:  # API, transport, schema, or JSON failure
            last_error = error
            if attempt == max_retries:
                break
            print(
                f"  attempt {attempt}/{max_retries} failed: {error}; "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)
    raise RuntimeError(f"Gemini evaluation failed after {max_retries} attempts") from last_error


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary, path)


def load_resume_output(
    output_path: Path,
    source_samples: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)
    if not isinstance(records, list):
        raise ValueError(f"{output_path}: resume output must be a JSON list")

    completed: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or type(record.get("sample_index")) is not int:
            raise ValueError(f"{output_path}: invalid resume record")
        sample_index = record["sample_index"]
        if sample_index in completed:
            raise ValueError(f"{output_path}: duplicate sample_index={sample_index}")
        if sample_index not in source_samples:
            raise ValueError(
                f"{output_path}: unexpected sample_index={sample_index}"
            )
        if record.get("video") != source_samples[sample_index].get("video"):
            raise ValueError(
                f"{output_path}: sample_index={sample_index} does not match source video"
            )
        validate_judge_result(record.get("evaluation"))
        completed[sample_index] = record
    return completed


def strict_pair_ras(positive: dict[str, Any], negative: dict[str, Any]) -> float:
    positive_eval = positive["evaluation"]
    negative_eval = negative["evaluation"]
    if positive_eval["accuracy"] == negative_eval["accuracy"]:
        return (positive_eval["score"] + negative_eval["score"]) / 2.0
    if positive_eval["accuracy"] == 0:
        return float(positive_eval["score"])
    return float(negative_eval["score"])


def print_ras_summary(output_path: Path) -> None:
    pairs = load_and_validate_pairs(output_path)
    print(f"\nStrict pair-wise RAS: {output_path.name}")
    print(f"{'Group':<22} {'Pairs':>6} {'RAS':>8}")
    print("-" * 40)
    for group in CATEGORIES + ["Anti-physics", "Counter-intuitive", "Overall"]:
        selected = select_pairs(pairs, group)
        scores = [
            strict_pair_ras(pair["positive"], pair["negative"])
            for pair in selected
        ]
        ras = sum(scores) / len(scores) if scores else None
        label = CATEGORY_SHORT_NAMES.get(group, group)
        value = "N/A" if ras is None else f"{ras:.3f}"
        print(f"{label:<22} {len(selected):>6d} {value:>8}")


def output_path_for(input_path: Path, output_dir: Path, model: str) -> Path:
    stem = re.sub(r"_results$", "", input_path.stem)
    model_slug = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()
    return output_dir / f"{stem}_evaluated_{model_slug}.json"


def evaluate_file(
    input_path: Path,
    output_path: Path,
    prompts: dict[str, str],
    args: argparse.Namespace,
    client: Any,
) -> None:
    pairs = load_and_validate_pairs(input_path)
    source_samples = {
        sample["sample_index"]: sample
        for pair in pairs
        for sample in (pair["positive"], pair["negative"])
    }
    if args.overwrite:
        completed: dict[int, dict[str, Any]] = {}
    else:
        completed = load_resume_output(output_path, source_samples)

    pending = [
        source_samples[index]
        for index in sorted(source_samples)
        if index not in completed
    ]
    if args.sample_indices is not None:
        requested_indices = set(args.sample_indices)
        unexpected_indices = sorted(requested_indices - set(source_samples))
        if unexpected_indices:
            raise ValueError(
                f"{input_path}: unexpected --sample-indices={unexpected_indices}"
            )
        pending = [
            sample
            for sample in pending
            if sample["sample_index"] in requested_indices
        ]
    if args.max_samples is not None:
        pending = pending[: args.max_samples]

    print(
        f"\n{input_path.name}: total={len(source_samples)}, "
        f"completed={len(completed)}, pending_this_run={len(pending)}"
    )
    failed_this_run = 0
    for ordinal, sample in enumerate(pending, start=1):
        category = next(
            pair["category"]
            for pair in pairs
            if sample["sample_index"]
            in (pair["positive"]["sample_index"], pair["negative"]["sample_index"])
        )
        stream = "Anti-physics" if category in CATEGORIES[:6] else "Counter-intuitive"
        user_prompt = prompts["input_template"].format(
            question=sample["prompt"],
            ground_truth=sample["ground_truth"],
            model_output=sample["model_output"],
        )
        try:
            evaluation = call_gemini(
                client=client,
                model=args.model,
                system_prompt=prompts[stream],
                user_prompt=user_prompt,
                max_retries=args.max_retries,
                max_output_tokens=args.max_output_tokens,
            )
        except Exception as error:
            if not args.continue_on_error:
                raise
            failed_this_run += 1
            print(
                f"  [{ordinal}/{len(pending)}] FAILED "
                f"sample_index={sample['sample_index']}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue
        evaluated = dict(sample)
        evaluated["evaluation"] = evaluation
        evaluated["evaluation_metadata"] = {
            "judge_provider": "OpenAI-compatible API",
            "judge_model": args.model,
            "judge_base_url": args.base_url,
            "prompt_file": str(args.prompt_file.resolve()),
            "rubric_stream": stream,
        }
        completed[sample["sample_index"]] = evaluated
        atomic_write_json(output_path, [completed[index] for index in sorted(completed)])
        print(
            f"  [{ordinal}/{len(pending)}] sample_index={sample['sample_index']} "
            f"accuracy={evaluation['accuracy']} score={evaluation['score']}",
            flush=True,
        )
        if args.request_delay > 0:
            time.sleep(args.request_delay)

    if len(completed) == len(source_samples):
        print_ras_summary(output_path)
    else:
        print(
            f"Partial output saved to {output_path}: "
            f"{len(completed)}/{len(source_samples)} samples complete; "
            f"failed_this_run={failed_this_run}"
        )


def main() -> None:
    args = parse_args()
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    if args.max_samples is not None and args.sample_indices is not None:
        raise ValueError("--max-samples and --sample-indices cannot be combined")
    if args.sample_indices is not None:
        if not args.sample_indices or any(index < 0 for index in args.sample_indices):
            raise ValueError("--sample-indices must contain non-negative integers")
        if len(args.sample_indices) != len(set(args.sample_indices)):
            raise ValueError("--sample-indices cannot contain duplicates")
    if args.request_delay < 0:
        raise ValueError("--request-delay cannot be negative")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")

    prompt_file = args.prompt_file.resolve()
    prompts = load_prompts(prompt_file)
    files = resolve_input_files(args.inputs, args.results_dir, args.pattern)

    if args.dry_run:
        print(f"Prompt file: {prompt_file}")
        print(
            "Prompt lengths: "
            f"Anti-physics={len(prompts['Anti-physics'])}, "
            f"Counter-intuitive={len(prompts['Counter-intuitive'])}, "
            f"input_template={len(prompts['input_template'])}"
        )
        for file_path in files:
            pairs = load_and_validate_pairs(file_path)
            print(
                f"Validated {file_path}: {2 * len(pairs)} samples, "
                f"{len(pairs)} pairs, model={get_model_name(file_path)}"
            )
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is not set. "
            "No API key is stored in this script."
        )
    if not args.base_url:
        raise RuntimeError(
            "--base-url or OPENAI_BASE_URL is required for API evaluation."
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is required to run API evaluation."
        ) from error

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
        max_retries=0,
    )
    output_dir = args.output_dir.resolve()
    for input_path in files:
        evaluate_file(
            input_path=input_path,
            output_path=output_path_for(input_path, output_dir, args.model),
            prompts=prompts,
            args=args,
            client=client,
        )


if __name__ == "__main__":
    main()
