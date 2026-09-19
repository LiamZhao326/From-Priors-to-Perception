"""Generate paired Observation-Attribution-Verdict labels with Gemini."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import load_json, require_env, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--verified-annotations", type=Path, required=True)
    parser.add_argument("--positive-dir", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, required=True)
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "prompts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-suffix", default="")
    parser.add_argument("--type", help="Optional record type, for example 'rev'.")
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_LABEL_MODEL", "gemini-3.1-pro-preview"),
    )
    return parser.parse_args()


def category_definition(scene_prompt: str) -> str:
    match = re.search(
        r"\*\*Context 1: Category Definition:\*\*(.*?)\*\*Context 2: Input Video:\*\*",
        scene_prompt,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Cannot locate the category definition in the scene prompt")
    return match.group(1).strip()


def iv_prompt_section(text: str, section: int) -> str:
    sections = re.findall(
        r"### Input Information.*?(?=^## IV-[AB]:|\Z)",
        text,
        flags=re.DOTALL | re.MULTILINE,
    )
    if section < 0 or section >= len(sections):
        raise ValueError(f"IV label prompt section {section} does not exist")
    return sections[section].strip()


def relative_dataset_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the dataset root {root}: {resolved}") from exc
    return str(relative).replace(os.sep, "/")


def fill_prompt(
    template: str,
    metadata: dict,
    category: str,
    verified_annotation: dict | str,
) -> str:
    judgement = metadata.get("interaction_judgement") or metadata.get("consequence_judgement") or ""
    annotation_text = (
        verified_annotation
        if isinstance(verified_annotation, str)
        else json.dumps(verified_annotation, ensure_ascii=False, indent=2)
    )
    replacements = {
        "{visual_fact_caption}": metadata.get("visual_fact_caption", ""),
        "{target_fallacy_scenario}": metadata.get("target_fallacy_scenario", ""),
        "{selected_typical_scenario}": metadata.get("selected_typical_scenario", ""),
        "{interaction_judgement}": metadata.get("interaction_judgement", ""),
        "{consequence_judgement}": metadata.get("consequence_judgement", ""),
        "{target_event}": metadata.get("target_event", ""),
        "{Insert visual_fact_caption here}": metadata.get("visual_fact_caption", ""),
        "{Insert target_fallacy_scenario here}": metadata.get("target_fallacy_scenario", ""),
        "{Insert selected_typical_scenario here}": metadata.get("selected_typical_scenario", ""),
        "{verified_positive_oav_annotation}": annotation_text,
        "{Insert verified_positive_oav_annotation here}": annotation_text,
        "{verified_original_oav_annotation}": annotation_text,
    }
    for token, value in replacements.items():
        template = template.replace(token, str(value))
    return (
        template
        + "\n\n### Resolved Metadata\n"
        + json.dumps(
            {
                "category": category,
                "visual_fact_caption": metadata.get("visual_fact_caption", ""),
                "target_fallacy_scenario": metadata.get("target_fallacy_scenario", ""),
                "selected_typical_scenario": metadata.get("selected_typical_scenario", ""),
                "target_event": metadata.get("target_event", ""),
                "original_event_judgement": judgement,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def normalized_samples(payload: dict, category: str) -> dict:
    positive = payload.get("positive_sample") or payload.get("original_sample")
    negative = payload.get("negative_sample") or payload.get("relative_sample")
    if not isinstance(positive, dict) or not isinstance(negative, dict):
        raise ValueError("Label response does not contain two sample objects")

    if category.startswith("IV_"):
        positive_verdict = "The target event occurred."
        negative_verdict = "The target event did not occur."
    else:
        positive_verdict = "The video is physically valid."
        negative_verdict = "The video is physically invalid."

    for sample, verdict in ((positive, positive_verdict), (negative, negative_verdict)):
        sample["verdict"] = verdict
        sample["sft_response"] = (
            f"**Observation**: {sample['observation']}\n\n"
            f"**Attribution**: {sample['causal_attribution']}\n\n"
            f"**Verdict**: {verdict}"
        )
    return {"positive_sample": positive, "negative_sample": negative}


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / "config" / "categories.json")
    if args.category not in config:
        raise ValueError(f"Unknown category: {args.category}")
    route = config[args.category]

    template = (args.prompts_dir / route["label_prompt"]).read_text(encoding="utf-8")
    if args.type == "rev":
        template = (args.prompts_dir / route["reverse_label_prompt"]).read_text(encoding="utf-8")
    elif args.category.startswith("IV_"):
        template = iv_prompt_section(template, int(route["label_prompt_section"]))
    elif route["label_prompt"] == "label_prompt.md":
        scene_prompt = (args.prompts_dir / route["scene_prompt"]).read_text(encoding="utf-8")
        template = template.replace(
            "{Insert category_definition here}",
            category_definition(scene_prompt),
        )

    metadata_items = load_json(args.metadata)
    metadata_by_name = {
        Path(str(item["video_path"]).replace("\\", "/")).name: item
        for item in metadata_items
    }
    verified_items = load_json(args.verified_annotations)
    if not isinstance(verified_items, dict):
        raise ValueError("--verified-annotations must contain a JSON object keyed by filename")
    verified_by_name = {
        Path(str(filename).replace("\\", "/")).name: annotation
        for filename, annotation in verified_items.items()
    }
    results = load_json(args.output) if args.output.exists() else []
    processed = {(item["filename"], item.get("type")) for item in results}
    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))

    dataset_root = args.output.resolve().parent
    if not args.negative_dir.name.endswith("_negative"):
        raise ValueError("--negative-dir must end with '_negative'")
    aligned_negative_dir = args.negative_dir.with_name(
        f"{args.negative_dir.name}_aligned"
    )

    positive_files = sorted(args.positive_dir.glob("*.mp4"))
    for index, positive_path in enumerate(positive_files, start=1):
        key = (positive_path.name, args.type)
        if key in processed or positive_path.name not in metadata_by_name:
            continue
        if positive_path.name not in verified_by_name:
            raise ValueError(f"Missing verified annotation for {positive_path.name}")
        negative_name = f"{positive_path.stem}{args.negative_suffix}{positive_path.suffix}"
        negative_path = args.negative_dir / negative_name
        if not negative_path.exists():
            continue
        metadata = metadata_by_name[positive_path.name]
        prompt = fill_prompt(
            template,
            metadata,
            args.category,
            verified_by_name[positive_path.name],
        )
        print(f"[{index}/{len(positive_files)}] {positive_path.name}")
        wait_seconds = 10
        for attempt in range(8):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                labels = normalized_samples(json.loads(response.text), args.category)
                record = {
                    "filename": positive_path.name,
                    "pos_video_path": relative_dataset_path(positive_path, dataset_root),
                    "neg_video_path": relative_dataset_path(
                        aligned_negative_dir / negative_name,
                        dataset_root,
                    ),
                    "original_metadata": metadata,
                    "llm_generated_labels": labels,
                }
                if args.type:
                    record["type"] = args.type
                results.append(record)
                processed.add(key)
                save_json(args.output, results)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(wait_seconds)
                wait_seconds = min(wait_seconds * 2, 120)


if __name__ == "__main__":
    main()
