#!/usr/bin/env python3
"""Compute Relative Prior Bias (RPB) from PriorPair task Verdicts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from priorpair_metrics_common import (
    CATEGORIES,
    CATEGORY_SHORT_NAMES,
    all_group_stats,
    get_model_name,
    load_and_validate_pairs,
    resolve_input_files,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Relative Prior Bias magnitude: "
            "100 * abs(Acc_pos - Acc_neg) / (Acc_pos + Acc_neg)."
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
        "--summary-file",
        type=Path,
        default=None,
        help="Optional JSON output. No file is written when omitted.",
    )
    return parser.parse_args()


def add_rpb(stats: dict) -> dict:
    positive = stats["positive_accuracy"]
    negative = stats["negative_accuracy"]
    denominator = None if positive is None else positive + negative
    rpb = (
        None
        if not denominator
        else 100.0 * abs(positive - negative) / denominator
    )
    return {
        **{
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in stats.items()
        },
        "rpb": None if rpb is None else round(rpb, 4),
    }


def calculate_file(file_path: Path) -> dict:
    pairs = load_and_validate_pairs(file_path)
    return {
        "model": get_model_name(file_path),
        "source_file": str(file_path),
        "samples": 2 * len(pairs),
        "pairs": len(pairs),
        "metrics": {
            group: add_rpb(stats)
            for group, stats in all_group_stats(pairs).items()
        },
    }


def display_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def print_results(results: list[dict]) -> None:
    display_groups = CATEGORIES + ["Anti-physics", "Counter-intuitive", "Overall"]
    for result in results:
        print(f"\n{result['model']}  ({result['pairs']} pairs)")
        print(
            f"{'Group':<22} {'Pairs':>6} {'Pos Acc':>10} {'Neg Acc':>10} "
            f"{'Overall':>10} {'RPB':>10}"
        )
        print("-" * 73)
        for group in display_groups:
            stats = result["metrics"][group]
            label = CATEGORY_SHORT_NAMES.get(group, group)
            print(
                f"{label:<22} {stats['pairs']:>6d} "
                f"{display_value(stats['positive_accuracy']):>10} "
                f"{display_value(stats['negative_accuracy']):>10} "
                f"{display_value(stats['point_accuracy']):>10} "
                f"{display_value(stats['rpb']):>10}"
            )
        print("Lower RPB indicates less imbalance between the two pair roles.")


def main() -> None:
    args = parse_args()
    files = resolve_input_files(args.inputs, args.results_dir, args.pattern)
    results = [calculate_file(path) for path in files]
    results.sort(
        key=lambda item: (
            item["metrics"]["Overall"]["rpb"] is not None,
            item["metrics"]["Overall"]["rpb"] or float("-inf"),
        ),
        reverse=True,
    )
    print_results(results)

    if args.summary_file is not None:
        summary_file = args.summary_file.resolve()
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with summary_file.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, ensure_ascii=False, indent=2)
        print(f"\nSaved summary: {summary_file}")


if __name__ == "__main__":
    main()
