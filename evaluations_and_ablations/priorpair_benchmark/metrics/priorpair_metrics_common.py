#!/usr/bin/env python3
"""Shared validation and task-specific verdict parsing for PriorPair metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


CATEGORIES = [
    "I_A_Coherence_Violation",
    "I_B_Causal_Reversal",
    "II_A_Existence_Violation",
    "II_B_Identity_Violation",
    "III_A_Dynamic_Violation",
    "III_B_Constraint_Violation",
    "IV_A_Near_Miss",
    "IV_B_Consequence_Arrest",
]

CATEGORY_SHORT_NAMES = {
    "I_A_Coherence_Violation": "I_A",
    "I_B_Causal_Reversal": "I_B",
    "II_A_Existence_Violation": "II_A",
    "II_B_Identity_Violation": "II_B",
    "III_A_Dynamic_Violation": "III_A",
    "III_B_Constraint_Violation": "III_B",
    "IV_A_Near_Miss": "IV_A",
    "IV_B_Consequence_Arrest": "IV_B",
}

ANTI_PHYSICS_CATEGORIES = CATEGORIES[:6]
COUNTER_INTUITIVE_CATEGORIES = CATEGORIES[6:]
GROUPS = {
    **{category: [category] for category in CATEGORIES},
    "Anti-physics": ANTI_PHYSICS_CATEGORIES,
    "Counter-intuitive": COUNTER_INTUITIVE_CATEGORIES,
    "Overall": CATEGORIES,
}


def is_negative(video_path: str) -> bool:
    return "_negative" in video_path


def get_category(video_path: str) -> str:
    matches = [category for category in CATEGORIES if category in video_path]
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely identify a PriorPair category for {video_path!r}: "
            f"{matches}"
        )
    return matches[0]


def get_stream(category: str) -> str:
    if category in ANTI_PHYSICS_CATEGORIES:
        return "Anti-physics"
    if category in COUNTER_INTUITIVE_CATEGORIES:
        return "Counter-intuitive"
    raise ValueError(f"Unknown PriorPair category: {category!r}")


def get_pair_id(video_path: str) -> str:
    path = Path(video_path)
    clean_parent = str(path.parent).replace("_negative_aligned", "").replace(
        "_negative", ""
    )
    clean_stem = path.stem
    if clean_stem.endswith("_rev"):
        clean_stem = clean_stem[: -len("_rev")]
    return f"{clean_parent}/{clean_stem}"


def get_model_name(file_path: Path) -> str:
    name = file_path.stem
    name = re.sub(r"_evaluated(?:_[A-Za-z0-9_.-]+)?$", "", name)
    name = re.sub(r"_results$", "", name)
    return name


def _normalise_text(text: str) -> str:
    return (
        text.lower()
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
    )


def _verdict_region(text: str) -> str:
    """Prefer the final explicit Verdict field; otherwise inspect the ending."""
    normalised = _normalise_text(text)
    matches = list(
        re.finditer(r"(?:\*\*)?verdict(?:\*\*)?\s*[:\-]", normalised)
    )
    if matches:
        return normalised[matches[-1].end() :].strip()

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalised)]
    paragraphs = [part for part in paragraphs if part]
    if paragraphs:
        return paragraphs[-1][-1000:]
    return normalised[-1000:]


def _last_label_match(region: str, patterns: dict[int, Iterable[str]]) -> int | None:
    matches: list[tuple[int, int, int]] = []
    for label, label_patterns in patterns.items():
        for pattern in label_patterns:
            for match in re.finditer(pattern, region, flags=re.IGNORECASE):
                matches.append((match.start(), match.end(), label))
    if not matches:
        return None

    matches.sort(key=lambda item: (item[0], item[1]))
    final_start = matches[-1][0]
    final_labels = {label for start, _, label in matches if start == final_start}
    if len(final_labels) != 1:
        return None
    return final_labels.pop()


PHYSICAL_VALIDITY_PATTERNS = {
    1: (
        r"\bthe\s+video\s+is\s+physically\s+valid\b",
        r"\bphysically\s+valid\b",
        r"\bvalid\b",
        r"\bthe\s+video\s+is\s+real\b",
        r"\bdoes\s+not\s+violate\b",
        r"\bcomplies?\s+with\b",
    ),
    0: (
        r"\bthe\s+video\s+is\s+physically\s+invalid\b",
        r"\bphysically\s+invalid\b",
        r"\binvalid\b",
        r"\bthe\s+video\s+is\s+forged\b",
        r"(?<!not\s)\bviolates?\b",
        r"\bphysically\s+impossible\b",
    ),
}

TARGET_EVENT_PATTERNS = {
    1: (
        r"\bthe\s+target\s+event\s+occurred\b",
        r"\btarget\s+event\s+occurred\b",
        r"\btarget\s+event\b.{0,400}(?<!not\s)\boccurred\b",
        r"\btarget\s+event\b.{0,400}(?<!not\s)\boccurs\b",
        r"\bthe\s+target\s+event\s+(?:did|does)\s+happen\b",
        r"\bthe\s+specified\s+event\s+occurred\b",
    ),
    0: (
        r"\bthe\s+target\s+event\s+did\s+not\s+occur\b",
        r"\btarget\s+event\s+did\s+not\s+occur\b",
        r"\btarget\s+event\b.{0,400}\b(?:did|does|has)\s+not\s+occur(?:red)?\b",
        r"\bthe\s+target\s+event\s+(?:did|does)\s+not\s+happen\b",
        r"\bthe\s+specified\s+event\s+did\s+not\s+occur\b",
    ),
}


LEADING_PHYSICAL_VALIDITY_PATTERNS = (
    (
        0,
        re.compile(
            r"^\s*(?:[-*]\s*)?(?:the\s+video\s+is\s+)?"
            r"(?:not\s+physically\s+valid|physically\s+not\s+valid|"
            r"physically\s+invalid|not\s+valid|invalid)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        1,
        re.compile(
            r"^\s*(?:[-*]\s*)?(?:the\s+video\s+is\s+)?"
            r"(?:physically\s+valid|valid)\b",
            flags=re.IGNORECASE,
        ),
    ),
)


def _leading_physical_validity_label(region: str) -> int | None:
    """Resolve an explicit physical-validity label at the Verdict opening."""
    for label, pattern in LEADING_PHYSICAL_VALIDITY_PATTERNS:
        if pattern.search(region):
            return label
    return None


def _extract_target_event(question: str) -> str | None:
    match = re.search(
        r"^\s*Target\s+event\s*:\s*(.+?)\s*$",
        question,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None
    target = re.sub(r"\s+", " ", _normalise_text(match.group(1))).strip()
    return target.rstrip(". !?") or None


def parse_task_verdict(
    text: str,
    category: str,
    question: str | None = None,
) -> int | None:
    """Return 1 for the task-positive label, 0 for the task-negative label."""
    if not isinstance(text, str) or not text.strip():
        return None
    region = _verdict_region(text)
    if category in ANTI_PHYSICS_CATEGORIES:
        # The first explicit statement in the final Verdict region is the
        # model's binary answer. Prefer it over later explanatory phrases such
        # as "not physically valid", whose nested "valid" token would
        # otherwise be mistaken for a positive label.
        leading_label = _leading_physical_validity_label(region)
        if leading_label is not None:
            return leading_label
    patterns = (
        PHYSICAL_VALIDITY_PATTERNS
        if category in ANTI_PHYSICS_CATEGORIES
        else TARGET_EVENT_PATTERNS
    )
    label = _last_label_match(region, patterns)
    if label is not None:
        return label

    # Some zero-shot models answer IV with the target-event sentence itself
    # (for example, "The glass door opens.") instead of the requested wrapper.
    # This is still an unambiguous positive Verdict and can be resolved without
    # asking an LLM judge, because the target event is explicitly in the prompt.
    if category in COUNTER_INTUITIVE_CATEGORIES and question:
        target_event = _extract_target_event(question)
        if target_event:
            normalised_region = re.sub(r"\s+", " ", region).replace("**", "")
            if target_event in normalised_region:
                return 1
    return None


def _validate_sample(sample: Any, position: int, file_path: Path) -> None:
    if not isinstance(sample, dict):
        raise ValueError(f"{file_path}: item {position} is not a JSON object")
    sample_index = sample.get("sample_index")
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError(
            f"{file_path}: item {position} has invalid sample_index={sample_index!r}"
        )
    for field in ("video", "prompt", "ground_truth"):
        if not isinstance(sample.get(field), str) or not sample[field]:
            raise ValueError(
                f"{file_path}: sample_index={sample_index} has invalid {field!r}"
            )
    if not isinstance(sample.get("model_output"), str):
        raise ValueError(
            f"{file_path}: sample_index={sample_index} has no string model_output"
        )


def load_and_validate_pairs(file_path: Path) -> list[dict[str, Any]]:
    """Load a complete inference result and return validated adjacent pairs."""
    with file_path.open("r", encoding="utf-8") as input_file:
        samples = json.load(input_file)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{file_path}: expected a non-empty JSON list")

    by_index: dict[int, dict[str, Any]] = {}
    for position, sample in enumerate(samples):
        _validate_sample(sample, position, file_path)
        sample_index = sample["sample_index"]
        if sample_index in by_index:
            raise ValueError(f"{file_path}: duplicate sample_index={sample_index}")
        by_index[sample_index] = sample

    expected = set(range(len(samples)))
    actual = set(by_index)
    if actual != expected:
        missing = sorted(expected - actual)[:20]
        unexpected = sorted(actual - expected)[:20]
        raise ValueError(
            f"{file_path}: sample_index must be contiguous from 0 to "
            f"{len(samples) - 1}; missing={missing}, unexpected={unexpected}"
        )
    if len(samples) % 2:
        raise ValueError(f"{file_path}: expected an even number of samples")

    pairs: list[dict[str, Any]] = []
    for positive_index in range(0, len(samples), 2):
        negative_index = positive_index + 1
        positive = by_index[positive_index]
        negative = by_index[negative_index]
        positive_video = positive["video"]
        negative_video = negative["video"]

        if is_negative(positive_video):
            raise ValueError(
                f"{file_path}: sample_index={positive_index} is not positive: "
                f"{positive_video}"
            )
        if not is_negative(negative_video):
            raise ValueError(
                f"{file_path}: sample_index={negative_index} is not negative: "
                f"{negative_video}"
            )

        positive_category = get_category(positive_video)
        negative_category = get_category(negative_video)
        if positive_category != negative_category:
            raise ValueError(
                f"{file_path}: pair {positive_index // 2} crosses categories: "
                f"{positive_category} vs {negative_category}"
            )
        if get_pair_id(positive_video) != get_pair_id(negative_video):
            raise ValueError(
                f"{file_path}: pair {positive_index // 2} has mismatched paths: "
                f"{positive_video} vs {negative_video}"
            )

        ground_truth_labels = []
        prediction_labels = []
        for sample in (positive, negative):
            ground_truth_label = parse_task_verdict(
                sample["ground_truth"], positive_category, sample["prompt"]
            )
            if ground_truth_label is None:
                raise ValueError(
                    f"{file_path}: sample_index={sample['sample_index']} has an "
                    "unparseable ground-truth Verdict"
                )
            ground_truth_labels.append(ground_truth_label)
            prediction_labels.append(
                parse_task_verdict(
                    sample["model_output"], positive_category, sample["prompt"]
                )
            )

        pairs.append(
            {
                "pair_index": positive_index // 2,
                "pair_id": get_pair_id(positive_video),
                "category": positive_category,
                "stream": get_stream(positive_category),
                "positive": positive,
                "negative": negative,
                "positive_ground_truth": ground_truth_labels[0],
                "negative_ground_truth": ground_truth_labels[1],
                "positive_prediction": prediction_labels[0],
                "negative_prediction": prediction_labels[1],
                "positive_correct": prediction_labels[0] == ground_truth_labels[0],
                "negative_correct": prediction_labels[1] == ground_truth_labels[1],
            }
        )
    return pairs


def select_pairs(pairs: list[dict[str, Any]], group_name: str) -> list[dict[str, Any]]:
    categories = set(GROUPS[group_name])
    return [pair for pair in pairs if pair["category"] in categories]


def calculate_accuracy_stats(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pair_count = len(pairs)
    if not pair_count:
        return {
            "pairs": 0,
            "positive_accuracy": None,
            "negative_accuracy": None,
            "point_accuracy": None,
            "pca": None,
            "parsed_predictions": 0,
            "parse_coverage": None,
        }

    positive_correct = sum(pair["positive_correct"] for pair in pairs)
    negative_correct = sum(pair["negative_correct"] for pair in pairs)
    pair_correct = sum(
        pair["positive_correct"] and pair["negative_correct"] for pair in pairs
    )
    parsed_predictions = sum(
        pair[side] is not None
        for pair in pairs
        for side in ("positive_prediction", "negative_prediction")
    )
    return {
        "pairs": pair_count,
        "positive_accuracy": 100.0 * positive_correct / pair_count,
        "negative_accuracy": 100.0 * negative_correct / pair_count,
        "point_accuracy": 100.0
        * (positive_correct + negative_correct)
        / (2 * pair_count),
        "pca": 100.0 * pair_correct / pair_count,
        "parsed_predictions": parsed_predictions,
        "parse_coverage": 100.0 * parsed_predictions / (2 * pair_count),
    }


def all_group_stats(pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        group_name: calculate_accuracy_stats(select_pairs(pairs, group_name))
        for group_name in GROUPS
    }


def resolve_input_files(
    inputs: list[Path] | None,
    results_dir: Path,
    pattern: str,
) -> list[Path]:
    if inputs:
        files = [path.resolve() for path in inputs]
    else:
        files = sorted(path.resolve() for path in results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            "No result files found; provide --inputs or adjust --results-dir/--pattern"
        )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {missing}")
    return files
