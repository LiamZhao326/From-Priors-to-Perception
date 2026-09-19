"""Generate image-to-video prompts from PriorPair scene metadata with Gemini."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import load_json, require_env, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "prompts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_PROMPT_MODEL", "gemini-3.1-pro-preview"),
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["AI_Generation"],
    )
    return parser.parse_args()


def frame_parts(video_path: Path, count: int) -> list[types.Part]:
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"Unreadable video: {video_path}")
    parts = []
    for index in np.linspace(0, total - 1, min(count, total), dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        encoded, buffer = cv2.imencode(".jpg", frame)
        if encoded:
            parts.append(types.Part.from_text(text=f"\n[Frame Timestamp: {index / fps:.3f}s]"))
            parts.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))
    capture.release()
    return parts


def category_of(item: dict) -> str:
    explicit = item.get("category")
    if explicit:
        return explicit
    return Path(str(item["video_path"]).replace("\\", "/")).parent.name


def dynamic_context(item: dict, category: str) -> str:
    judgement = item.get("interaction_judgement") or item.get("consequence_judgement") or ""
    return (
        "\n### Resolved Dynamic Inputs\n"
        f"* Camera & Style: {item.get('visual_style_and_camera', '')}\n"
        f"* Original Caption: {item.get('visual_fact_caption', '')}\n"
        f"* Category: {category}\n"
        f"* Target Event: {item.get('target_event', '')}\n"
        f"* Original Event Judgement: {judgement}\n"
        f"* Target Scenario: {item.get('target_fallacy_scenario', '')}\n"
    )


def main() -> None:
    args = parse_args()
    prompts = {
        "standard": (args.prompts_dir / "prompt_generate.md").read_text(encoding="utf-8"),
        "iv": (args.prompts_dir / "prompt_generate_IV.md").read_text(encoding="utf-8"),
    }
    tasks = []
    for metadata_path in sorted(args.metadata_dir.glob("*_llmcaption.json")):
        for item in load_json(metadata_path):
            if item.get("generation_method") in args.methods:
                tasks.append(item)

    results = load_json(args.output) if args.output.exists() else []
    processed = {item["video_path"] for item in results}
    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))

    for index, item in enumerate(tasks, start=1):
        video_path = Path(item["video_path"])
        if item["video_path"] in processed:
            continue
        category = category_of(item)
        prompt = prompts["iv" if category.startswith("IV_") else "standard"]
        parts = [
            *frame_parts(video_path, args.frames),
            types.Part.from_text(text=prompt + dynamic_context(item, category)),
        ]
        print(f"[{index}/{len(tasks)}] {category}/{video_path.name}")
        wait_seconds = 10
        for attempt in range(8):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=[types.Content(parts=parts)],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                generated = json.loads(response.text)["generated_video_prompt"]
                record = {
                    "video_path": item["video_path"],
                    "category": category,
                    "generated_video_prompt": generated,
                }
                results.append(record)
                processed.add(item["video_path"])
                save_json(args.output, results)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(wait_seconds)
                wait_seconds = min(wait_seconds * 2, 120)


if __name__ == "__main__":
    main()
