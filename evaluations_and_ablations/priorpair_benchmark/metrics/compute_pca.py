#!/usr/bin/env python3
"""Compute task-specific point accuracy and PCA from PriorPair inference JSON."""

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
            "Compute PriorPair point-wise accuracy and Pair-wise Consistency "
            "Accuracy directly from task-specific Verdicts."
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


def rounded_stats(stats: dict) -> dict:
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in stats.items()
    }


def calculate_file(file_path: Path) -> dict:
    pairs = load_and_validate_pairs(file_path)
    stats = {
        name: rounded_stats(values)
        for name, values in all_group_stats(pairs).items()
    }
    return {
        "model": get_model_name(file_path),
        "source_file": str(file_path),
        "samples": 2 * len(pairs),
        "pairs": len(pairs),
        "metrics": stats,
    }


def print_results(results: list[dict]) -> None:
    display_groups = CATEGORIES + ["Anti-physics", "Counter-intuitive", "Overall"]
    for result in results:
        print(f"\n{result['model']}  ({result['pairs']} pairs)")
        print(
            f"{'Group':<22} {'Pairs':>6} {'Pos Acc':>9} {'Neg Acc':>9} "
            f"{'Point Acc':>10} {'PCA':>8} {'Parsed':>9}"
        )
        print("-" * 79)
        for group in display_groups:
            stats = result["metrics"][group]
            label = CATEGORY_SHORT_NAMES.get(group, group)
            print(
                f"{label:<22} {stats['pairs']:>6d} "
                f"{stats['positive_accuracy']:>8.2f}% "
                f"{stats['negative_accuracy']:>8.2f}% "
                f"{stats['point_accuracy']:>9.2f}% "
                f"{stats['pca']:>7.2f}% "
                f"{stats['parse_coverage']:>8.2f}%"
            )


def main() -> None:
    args = parse_args()
    files = resolve_input_files(args.inputs, args.results_dir, args.pattern)
    results = [calculate_file(path) for path in files]
    results.sort(key=lambda item: item["metrics"]["Overall"]["pca"], reverse=True)
    print_results(results)

    if args.summary_file is not None:
        summary_file = args.summary_file.resolve()
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with summary_file.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, ensure_ascii=False, indent=2)
        print(f"\nSaved summary: {summary_file}")


if __name__ == "__main__":
    main()
