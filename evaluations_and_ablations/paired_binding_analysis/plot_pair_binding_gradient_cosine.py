#!/usr/bin/env python3
"""Render the pair-binding gradient-cosine mechanism figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RANDOM_COLOR = "#D97706"
MATCHED_COLOR = "#2C6EBA"
INCREASE_COLOR = "#789BC4"
DECREASE_COLOR = "#C2C2C2"
TEXT_COLOR = "#202020"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--paired-only",
        action="store_true",
        help="Render only the paired cosine plot, leaving the schematic untouched.",
    )
    return parser.parse_args()


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0.0:
        raise ValueError("Cannot compute cosine for a zero gradient sketch")
    return float(np.dot(first, second) / denominator)


def load_paired_cosines(vector_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(vector_path.read_text(encoding="utf-8"))
    gradients = {
        record["record_key"]: np.asarray(record["gradient_sketch"], dtype=np.float64)
        for record in payload["per_record_gradients"]
    }
    pair_ids = payload["plan"]["selected_pair_ids_in_window_order"]
    random_mapping = payload["plan"]["random_negative_for_positive_pair_id"]

    matched = []
    random_values = []
    for pair_id in pair_ids:
        positive = gradients[f"pair_{pair_id}:positive"]
        matched_negative = gradients[f"pair_{pair_id}:negative"]
        random_pair_id = random_mapping[str(pair_id)]
        random_negative = gradients[f"pair_{random_pair_id}:negative"]
        matched.append(cosine(positive, matched_negative))
        random_values.append(cosine(positive, random_negative))
    return np.asarray(random_values), np.asarray(matched)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def draw_arrow(
    axis: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    label: str,
    label_offset: tuple[float, float],
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    delta = end - start
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "mutation_scale": 12,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    midpoint = start + 0.55 * delta
    axis.text(
        midpoint[0] + label_offset[0],
        midpoint[1] + label_offset[1],
        label,
        color=color,
        fontsize=9,
        ha="center",
        va="center",
    )


def render_schematic(
    output_dir: Path, random_mean: float, matched_mean: float
) -> None:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True, sharey=True)
    panels = (
        (axes[0], random_mean, RANDOM_COLOR, "Random counterpart"),
        (axes[1], matched_mean, MATCHED_COLOR, "Matched counterpart"),
    )
    for axis, similarity, condition_color, title in panels:
        angle = math.acos(float(np.clip(similarity, -1.0, 1.0)))
        positive = np.asarray([1.0, 0.0])
        negative = np.asarray([math.cos(angle), math.sin(angle)])
        resultant = positive + negative
        origin = np.zeros(2)

        axis.axhline(0.0, color="#D0D0D0", linewidth=0.7, zorder=0)
        axis.axvline(0.0, color="#D0D0D0", linewidth=0.7, zorder=0)
        draw_arrow(
            axis,
            origin,
            positive,
            "#4B5563",
            r"$-g^{+}$",
            (0.0, -0.10),
        )
        draw_arrow(
            axis,
            origin,
            negative,
            condition_color,
            r"$-g^{-}$",
            (-0.10, 0.02),
        )
        draw_arrow(
            axis,
            origin,
            resultant,
            "#111111",
            r"$-(g^{+}+g^{-})$",
            (0.08, 0.06),
            linestyle="--",
            linewidth=1.5,
        )
        arc_angles = np.linspace(0.0, angle, 60)
        arc_radius = 0.27
        axis.plot(
            arc_radius * np.cos(arc_angles),
            arc_radius * np.sin(arc_angles),
            color=condition_color,
            linewidth=1.2,
        )
        axis.text(
            0.49,
            0.90,
            rf"mean cosine = {similarity:.3f}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color=condition_color,
        )
        axis.set_title(title, fontweight="semibold")
        axis.set_xlim(-0.12, 1.18)
        axis.set_ylim(-0.16, 1.18)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)

    figure.suptitle(
        "True counterparts yield more compatible pair-level updates",
        fontsize=11,
        fontweight="semibold",
        color=TEXT_COLOR,
        y=1.02,
    )
    figure.tight_layout(w_pad=1.5)
    save_figure(figure, output_dir / "gradient_cosine_schematic")


def render_paired_plot(
    output_dir: Path, random_values: np.ndarray, matched_values: np.ndarray
) -> None:
    configure_style()
    differences = matched_values - random_values
    increased = differences > 0
    order = np.argsort(differences)
    random_ordered = random_values[order]
    matched_ordered = matched_values[order]
    increased_ordered = increased[order]

    random_x = 0.0
    matched_x = 1.0

    figure, axis = plt.subplots(figsize=(5.2, 5.0))
    for random_value, matched_value, rose in zip(
        random_ordered, matched_ordered, increased_ordered
    ):
        axis.plot(
            [random_x, matched_x],
            [random_value, matched_value],
            color=INCREASE_COLOR if rose else DECREASE_COLOR,
            alpha=0.38 if rose else 0.60,
            linewidth=0.75,
            zorder=1,
        )
    axis.scatter(
        np.full_like(random_ordered, random_x),
        random_ordered,
        s=17,
        color=RANDOM_COLOR,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        zorder=2,
    )
    axis.scatter(
        np.full_like(matched_ordered, matched_x),
        matched_ordered,
        s=17,
        color=MATCHED_COLOR,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        zorder=2,
    )

    random_mean = float(random_values.mean())
    matched_mean = float(matched_values.mean())
    axis.scatter(
        [random_x, matched_x],
        [random_mean, matched_mean],
        marker="D",
        s=54,
        color=[RANDOM_COLOR, MATCHED_COLOR],
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
    )
    axis.plot(
        [random_x, matched_x],
        [random_mean, matched_mean],
        color="black",
        linewidth=1.7,
        zorder=3,
    )
    axis.axhline(0.0, color="#8F8F8F", linewidth=0.8, linestyle="--", zorder=0)

    axis.annotate(
        f"Mean {random_mean:.3f}",
        xy=(random_x, random_mean),
        xytext=(-10, -10),
        textcoords="offset points",
        ha="right",
        va="top",
        color=RANDOM_COLOR,
        fontsize=8.5,
        fontweight="semibold",
    )
    axis.annotate(
        f"Mean {matched_mean:.3f}",
        xy=(matched_x, matched_mean),
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=MATCHED_COLOR,
        fontsize=8.5,
        fontweight="semibold",
    )
    axis.text(
        0.50,
        1.035,
        f"Mean difference: {differences.mean():+.3f}; "
        f"{100.0 * increased.mean():.0f}% of comparisons increase",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color=TEXT_COLOR,
        fontweight="bold",
    )

    lower = min(-0.20, float(min(random_values.min(), matched_values.min())) - 0.04)
    upper = max(0.45, float(max(random_values.max(), matched_values.max())) + 0.05)
    axis.set_xlim(-0.34, 1.34)
    axis.set_ylim(lower, upper)
    axis.set_xticks([random_x, matched_x], ["Random\npairs", "Matched\npairs"])
    for label in axis.get_xticklabels():
        label.set_color(TEXT_COLOR)
        label.set_fontweight("bold")
        label.set_horizontalalignment("center")
        label.set_multialignment("center")
        label.set_linespacing(0.85)
    axis.set_ylabel("Gradient cosine similarity", fontweight="bold")
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
    axis.tick_params(axis="x", pad=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(figure, output_dir / "gradient_cosine_paired_plot")


def main() -> None:
    args = parse_args()
    vector_path = args.analysis_dir / "gradient_direction_vectors.json"
    if not vector_path.is_file():
        raise FileNotFoundError(f"Gradient vector JSON not found: {vector_path}")
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    random_values, matched_values = load_paired_cosines(vector_path)
    if not args.paired_only:
        render_schematic(
            args.analysis_dir,
            float(random_values.mean()),
            float(matched_values.mean()),
        )
    render_paired_plot(args.analysis_dir, random_values, matched_values)
    print(
        f"Rendered figures in {args.analysis_dir}; "
        f"random_mean={random_values.mean():.6f}, "
        f"matched_mean={matched_values.mean():.6f}, "
        f"increased={100.0 * (matched_values > random_values).mean():.0f}%"
    )


if __name__ == "__main__":
    main()
