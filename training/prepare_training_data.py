"""Build deterministic PriorPair train/test splits and paired SFT JSONL files."""

import argparse
import json
import os
import random
import re
from pathlib import Path


SEED = 42
TRAIN_RATIO = 0.8
EXPECTED_PAIR_COUNT = 828
EXPECTED_REV_PAIR_COUNT = 146

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

PROMPT_SECTION_TITLES = {
    "physical": "I–III: Physical-validity verification",
    "target_event": "IV: Target-event occurrence verification",
}


def extract_label_samples(record):
    labels = record.get("llm_generated_labels", {})
    if isinstance(labels, list):
        labels = labels[0] if labels else {}
    if not isinstance(labels, dict):
        raise ValueError("llm_generated_labels must be an object")

    positive = labels.get("positive_sample") or labels.get("original_sample") or {}
    negative = labels.get("negative_sample") or labels.get("relative_sample") or {}
    if isinstance(positive, list):
        positive = positive[0] if positive else {}
    if isinstance(negative, list):
        negative = negative[0] if negative else {}
    if not isinstance(positive, dict) or not isinstance(negative, dict):
        raise ValueError("Positive and negative labels must be objects")
    return positive, negative


def pair_key(record):
    return record["pos_video_path"], record["neg_video_path"]


def scenario_key(record):
    metadata = record.get("original_metadata")
    scenario = (
        metadata.get("selected_typical_scenario")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("missing selected_typical_scenario")
    return scenario.strip()


def validate_relative_video_path(dataset_root, raw_path, expected_parent, field_name):
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{field_name} must be a non-empty string")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe {field_name}: {raw_path}")
    if relative_path.parent.as_posix() != expected_parent:
        raise ValueError(
            f"Unexpected {field_name} folder: {raw_path}; expected {expected_parent}/"
        )
    disk_path = dataset_root / relative_path
    if not disk_path.is_file():
        raise FileNotFoundError(f"Missing video referenced by {field_name}: {disk_path}")
    return relative_path


def validate_record(dataset_root, record, category, labels_path, record_index):
    prefix = f"{labels_path.name}[{record_index}]"
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} must be an object")
    filename = record.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{prefix} has no valid filename")

    try:
        positive_path = validate_relative_video_path(
            dataset_root, record.get("pos_video_path"), category, "pos_video_path"
        )
        negative_path = validate_relative_video_path(
            dataset_root,
            record.get("neg_video_path"),
            f"{category}_negative_aligned",
            "neg_video_path",
        )
        positive_label, negative_label = extract_label_samples(record)
        scenario_key(record)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
        raise type(error)(f"{prefix}: {error}") from error

    if positive_path.name != filename:
        raise ValueError(f"{prefix}: filename and positive video path do not match")
    expected_negative_name = filename
    if record.get("type") == "rev":
        expected_negative_name = f"{Path(filename).stem}_rev{Path(filename).suffix}"
    if negative_path.name != expected_negative_name:
        raise ValueError(f"{prefix}: unexpected negative filename")

    positive_response = positive_label.get("sft_response")
    negative_response = negative_label.get("sft_response")
    if not isinstance(positive_response, str) or not positive_response:
        raise ValueError(f"{prefix}: missing positive sft_response")
    if not isinstance(negative_response, str) or not negative_response:
        raise ValueError(f"{prefix}: missing negative sft_response")

    if category.startswith("IV_"):
        metadata = record.get("original_metadata")
        target_event = metadata.get("target_event") if isinstance(metadata, dict) else None
        if not isinstance(target_event, str) or not target_event.strip():
            raise ValueError(f"{prefix}: missing IV target_event")
        expected_verdicts = (
            "The target event occurred.",
            "The target event did not occur.",
        )
    else:
        expected_verdicts = (
            "The video is physically valid.",
            "The video is physically invalid.",
        )

    actual_verdicts = (positive_label.get("verdict"), negative_label.get("verdict"))
    if actual_verdicts != expected_verdicts:
        raise ValueError(
            f"{prefix}: verdicts={actual_verdicts!r}; expected {expected_verdicts!r}"
        )
    if not positive_response.rstrip().endswith(expected_verdicts[0]):
        raise ValueError(f"{prefix}: positive response has an unexpected verdict")
    if not negative_response.rstrip().endswith(expected_verdicts[1]):
        raise ValueError(f"{prefix}: negative response has an unexpected verdict")


def load_source_records(dataset_root, category):
    labels_path = dataset_root / f"{category}_labels.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    records = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {labels_path}")
    seen_pairs = set()
    for record_index, record in enumerate(records):
        validate_record(dataset_root, record, category, labels_path, record_index)
        key = pair_key(record)
        if key in seen_pairs:
            raise ValueError(f"Duplicate pair in {labels_path}: {key}")
        seen_pairs.add(key)
    return records


def canonical_record(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_partition(source_records, train_records, test_records, category):
    source = {pair_key(record): canonical_record(record) for record in source_records}
    train = {pair_key(record): canonical_record(record) for record in train_records}
    test = {pair_key(record): canonical_record(record) for record in test_records}
    if len(source) != len(source_records) or len(train) != len(train_records):
        raise RuntimeError(f"Duplicate pair detected in {category}")
    if len(test) != len(test_records) or train.keys() & test.keys():
        raise RuntimeError(f"Invalid train/test partition in {category}")
    if set(source) != set(train) | set(test):
        raise RuntimeError(f"Incomplete train/test partition in {category}")
    for key, content in source.items():
        if train.get(key, test.get(key)) != content:
            raise RuntimeError(f"Record changed during split for pair {key}")


def allocate_scenario_train_counts(scenario_groups, category_train_count):
    exact = {name: len(records) * TRAIN_RATIO for name, records in scenario_groups.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remaining = category_train_count - sum(counts.values())
    order = sorted(
        scenario_groups,
        key=lambda name: (-(exact[name] - counts[name]), name),
    )
    if remaining < 0 or remaining > len(order):
        raise RuntimeError("Cannot preserve the category-level train total")
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def split_category(source_records, category):
    groups = {}
    for record in source_records:
        groups.setdefault(scenario_key(record), []).append(record)
    category_train_count = int(len(source_records) * TRAIN_RATIO)
    scenario_train_counts = allocate_scenario_train_counts(groups, category_train_count)
    train_records = []
    test_records = []
    for scenario in sorted(groups):
        records = list(groups[scenario])
        random.Random(f"{SEED}:{category}:{scenario}").shuffle(records)
        train_count = scenario_train_counts[scenario]
        train_records.extend(records[:train_count])
        test_records.extend(records[train_count:])
    random.Random(f"{SEED}:{category}:train-order").shuffle(train_records)
    random.Random(f"{SEED}:{category}:test-order").shuffle(test_records)
    validate_partition(source_records, train_records, test_records, category)
    return train_records, test_records


def build_stratified_splits(dataset_root):
    splits = {"train": {}, "test": {}}
    all_pairs = set()
    total_pairs = 0
    total_rev_pairs = 0
    for category in CATEGORIES:
        source_records = load_source_records(dataset_root, category)
        train_records, test_records = split_category(source_records, category)
        splits["train"][category] = train_records
        splits["test"][category] = test_records
        keys = {pair_key(record) for record in source_records}
        if all_pairs & keys:
            raise RuntimeError("A pair appears in multiple categories")
        all_pairs.update(keys)
        total_pairs += len(source_records)
        total_rev_pairs += sum(record.get("type") == "rev" for record in source_records)
    if total_pairs != EXPECTED_PAIR_COUNT or len(all_pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_PAIR_COUNT} unique pairs, found {total_pairs}")
    if total_rev_pairs != EXPECTED_REV_PAIR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_REV_PAIR_COUNT} reversed pairs, found {total_rev_pairs}"
        )
    return splits


def extract_prompt_section(markdown_text, section_title):
    pattern = re.compile(
        rf"^##\s+{re.escape(section_title)}\s*$\s*```text\s*\n(.*?)\n```",
        flags=re.MULTILINE | re.DOTALL,
    )
    matches = pattern.findall(markdown_text)
    if len(matches) != 1:
        raise ValueError(f"Expected one prompt block under {section_title!r}")
    return matches[0].strip()


def load_task_prompts(prompts_path):
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Missing task prompt file: {prompts_path}")
    markdown_text = prompts_path.read_text(encoding="utf-8")
    prompts = {
        key: extract_prompt_section(markdown_text, title)
        for key, title in PROMPT_SECTION_TITLES.items()
    }
    if "{target_event}" in prompts["physical"]:
        raise ValueError("The I–III prompt must not contain {target_event}")
    if prompts["target_event"].count("{target_event}") != 1:
        raise ValueError("The IV prompt must contain exactly one {target_event}")
    return prompts


def render_prompt(category, record, prompts):
    if not category.startswith("IV_"):
        return prompts["physical"]
    metadata = record.get("original_metadata")
    target_event = metadata.get("target_event") if isinstance(metadata, dict) else None
    if not isinstance(target_event, str) or not target_event.strip():
        raise ValueError(f"Missing target_event for pair {pair_key(record)}")
    return prompts["target_event"].replace("{target_event}", target_event.strip())


def build_sample(video_path, response, prompt):
    return {
        "video": [video_path],
        "conversations": [
            {"from": "human", "value": f"<video>\n{prompt}"},
            {"from": "gpt", "value": response},
        ],
    }


def build_jsonl_samples(splits, prompts):
    samples_by_split = {}
    pair_keys_by_split = {}
    for split_name in ("train", "test"):
        samples = []
        keys = set()
        for category in CATEGORIES:
            for record in splits[split_name][category]:
                key = pair_key(record)
                if key in keys:
                    raise RuntimeError(f"Duplicate pair in {split_name}: {key}")
                keys.add(key)
                positive_label, negative_label = extract_label_samples(record)
                prompt = render_prompt(category, record, prompts)
                samples.extend(
                    [
                        build_sample(
                            record["pos_video_path"], positive_label["sft_response"], prompt
                        ),
                        build_sample(
                            record["neg_video_path"], negative_label["sft_response"], prompt
                        ),
                    ]
                )
        samples_by_split[split_name] = samples
        pair_keys_by_split[split_name] = keys
    if pair_keys_by_split["train"] & pair_keys_by_split["test"]:
        raise RuntimeError("Train/test pair overlap detected")
    if len(samples_by_split["train"]) != 1320:
        raise RuntimeError("Expected 660 training pairs")
    if len(samples_by_split["test"]) != 336:
        raise RuntimeError("Expected 168 test pairs")
    return samples_by_split


def atomic_write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_jsonl(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for sample in samples:
                output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build deterministic PriorPair splits and paired SFT JSONL files."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--prompts-path",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/prompts/task_prompts.md.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and construct all records without writing files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    prompts_path = (
        args.prompts_path.resolve()
        if args.prompts_path is not None
        else dataset_root / "prompts" / "task_prompts.md"
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    splits = build_stratified_splits(dataset_root)
    prompts = load_task_prompts(prompts_path)
    samples = build_jsonl_samples(splits, prompts)

    if not args.dry_run:
        for split_name in ("train", "test"):
            split_dir = dataset_root / split_name
            for category in CATEGORIES:
                atomic_write_json(
                    split_dir / f"{category}_labels.json",
                    splits[split_name][category],
                )
            atomic_write_jsonl(
                split_dir / f"{split_name}_dataset.jsonl",
                samples[split_name],
            )

    print(
        "PriorPair preparation passed: "
        f"train_pairs={len(samples['train']) // 2}, "
        f"test_pairs={len(samples['test']) // 2}, "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
