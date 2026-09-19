"""Generate PriorPair counterpart videos from direct prompts with Kling."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import time
from pathlib import Path

import cv2
import jwt
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", default="kling-v3")
    parser.add_argument("--duration", default="5")
    parser.add_argument("--mode", default="pro")
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def auth_headers(access_key: str, secret_key: str) -> dict[str, str]:
    now = int(time.time())
    token = jwt.encode(
        {"iss": access_key, "exp": now + 1800, "nbf": now - 5},
        secret_key,
        algorithm="HS256",
        headers={"alg": "HS256", "typ": "JWT"},
    )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def encode_frame(frame) -> str:
    height, width = frame.shape[:2]
    ratio = width / height
    resolutions = [(1280, 720), (720, 1280), (720, 720), (960, 720), (720, 960)]
    target_width, target_height = min(
        resolutions, key=lambda size: abs(size[0] / size[1] - ratio)
    )
    scale = max(target_width / width, target_height / height)
    resized = cv2.resize(
        frame,
        (max(target_width, math.ceil(width * scale)), max(target_height, math.ceil(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (resized.shape[1] - target_width) // 2
    y = (resized.shape[0] - target_height) // 2
    cropped = resized[y : y + target_height, x : x + target_width]
    ok, buffer = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Failed to encode first frame")
    return base64.b64encode(buffer).decode("ascii")


def first_frame(path: Path) -> str:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        frame = cv2.imread(str(path))
    else:
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            frame = None
    if frame is None:
        raise ValueError(f"Cannot read first-frame source: {path}")
    return encode_frame(frame)


def main() -> None:
    args = parse_args()
    access_key = require_env("KLING_ACCESS_KEY")
    secret_key = require_env("KLING_SECRET_KEY")
    tasks = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("--prompts must contain a JSON list")
    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results.exists() else []
    completed = {item["task_key"] for item in results}

    for index, task in enumerate(tasks, start=1):
        source = Path(task["video_path"])
        category = task["category"]
        key = f"{category}|{source.resolve()}"
        if key in completed:
            continue
        output = args.output_root / category / source.name
        if output.exists():
            continue
        print(f"[{index}/{len(tasks)}] {key}")
        payload = {
            "model_name": args.model,
            "image": first_frame(source),
            "prompt": task["generated_video_prompt"],
            "duration": args.duration,
            "mode": args.mode,
            "sound": "off",
        }
        response = requests.post(
            "https://api-beijing.klingai.com/v1/videos/image2video",
            headers=auth_headers(access_key, secret_key),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        response = response.json()
        if response.get("code") != 0:
            raise RuntimeError(f"Kling submission failed: {response}")
        task_id = response["data"]["task_id"]
        while True:
            status = requests.get(
                f"https://api-beijing.klingai.com/v1/videos/image2video/{task_id}",
                headers=auth_headers(access_key, secret_key),
                timeout=60,
            )
            status.raise_for_status()
            status = status.json()
            data = status.get("data", {})
            if data.get("task_status") == "succeed":
                url = data["task_result"]["videos"][0]["url"]
                break
            if data.get("task_status") in {"failed", "killed"}:
                raise RuntimeError(f"Kling generation failed: {data}")
            time.sleep(10)
        output.parent.mkdir(parents=True, exist_ok=True)
        download = requests.get(url, timeout=300)
        download.raise_for_status()
        output.write_bytes(download.content)
        results.append(
            {
                "task_key": key,
                "source_video_path": str(source),
                "output_video_path": str(output),
                "kling_task_id": task_id,
            }
        )
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        completed.add(key)


if __name__ == "__main__":
    main()
