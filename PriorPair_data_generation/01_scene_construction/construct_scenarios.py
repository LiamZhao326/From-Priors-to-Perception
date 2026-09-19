"""Construct category-specific PriorPair scene metadata with Gemini.

One call produces the visual caption and, where required by the category prompt,
the target scenario and generation-method decision.
"""

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

from common import list_videos, load_json, require_env, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "prompts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_SCENE_MODEL", "gemini-3.1-pro-preview"),
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def extract_frame_parts(video_path: Path, count: int) -> list[types.Part]:
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"Unreadable video metadata: {video_path}")

    indices = np.linspace(0, total_frames - 1, min(count, total_frames), dtype=int)
    parts: list[types.Part] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            continue
        parts.append(types.Part.from_text(text=f"\n[Frame Timestamp: {index / fps:.3f}s]"))
        parts.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))
    capture.release()
    if not parts:
        raise ValueError(f"No frames could be decoded: {video_path}")
    return parts


def generate_metadata(
    client: genai.Client,
    model: str,
    prompt: str,
    video_path: Path,
    frame_count: int,
    retries: int,
) -> dict:
    contents = [types.Part.from_text(text=prompt), *extract_frame_parts(video_path, frame_count)]
    wait_seconds = 10
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(parts=contents)],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            result = json.loads(response.text)
            if isinstance(result, list):
                if len(result) != 1:
                    raise ValueError("Expected one JSON object from Gemini")
                result = result[0]
            if not isinstance(result, dict):
                raise ValueError("Gemini response is not a JSON object")
            return result
        except Exception:
            if attempt == retries:
                raise
            time.sleep(wait_seconds)
            wait_seconds = min(wait_seconds * 2, 120)
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    category_config = load_json(ROOT / "config" / "categories.json")
    if args.category not in category_config:
        raise ValueError(f"Unknown category: {args.category}")

    prompt_path = args.prompts_dir / category_config[args.category]["scene_prompt"]
    prompt = prompt_path.read_text(encoding="utf-8")
    api_key = require_env("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    results = load_json(args.output) if args.output.exists() else []
    processed = {
        Path(str(item["video_path"]).replace("\\", "/")).name for item in results
    }
    videos = list_videos(args.input_dir, recursive=args.recursive)

    for index, video_path in enumerate(videos, start=1):
        if video_path.name in processed:
            print(f"[{index}/{len(videos)}] skip {video_path.name}")
            continue
        print(f"[{index}/{len(videos)}] process {video_path.name}")
        metadata = generate_metadata(
            client,
            args.model,
            prompt,
            video_path,
            args.frames,
            args.retries,
        )
        results.append({"video_path": str(video_path.resolve()), **metadata})
        save_json(args.output, results)
        processed.add(video_path.name)


if __name__ == "__main__":
    main()
