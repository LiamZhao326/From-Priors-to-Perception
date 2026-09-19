#!/usr/bin/env python3
"""Compare two repeated RAS evaluations at the strict pair level."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from priorpair_metrics_common import load_and_validate_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--second-run", type=Path, required=True)
    return parser.parse_args()


def strict_pair_ras(pair: dict) -> float:
    positive = pair["positive"]["evaluation"]
    negative = pair["negative"]["evaluation"]
    if positive["accuracy"] == negative["accuracy"]:
        return (positive["score"] + negative["score"]) / 2.0
    return float(
        positive["score"] if positive["accuracy"] == 0 else negative["score"]
    )


def pair_key(pair: dict) -> tuple[int, int]:
    return (
        pair["positive"]["sample_index"],
        pair["negative"]["sample_index"],
    )


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def pearson(first: list[float], second: list[float]) -> float:
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second)
    )
    first_norm = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_norm = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    if not first_norm or not second_norm:
        raise ValueError("Spearman correlation is undefined for a constant run")
    return numerator / (first_norm * second_norm)


def main() -> None:
    args = parse_args()
    first_pairs = {
        pair_key(pair): strict_pair_ras(pair)
        for pair in load_and_validate_pairs(args.first_run)
    }
    second_pairs = {
        pair_key(pair): strict_pair_ras(pair)
        for pair in load_and_validate_pairs(args.second_run)
    }
    if set(first_pairs) != set(second_pairs):
        raise ValueError("The two runs contain different pair sets")
    keys = sorted(first_pairs)
    first = [first_pairs[key] for key in keys]
    second = [second_pairs[key] for key in keys]
    rho = pearson(rankdata(first), rankdata(second))
    mae = sum(abs(left - right) for left, right in zip(first, second)) / len(keys)
    print(f"pairs={len(keys)}")
    print(f"spearman_rho={rho:.6f}")
    print(f"mae={mae:.6f}")


if __name__ == "__main__":
    main()
