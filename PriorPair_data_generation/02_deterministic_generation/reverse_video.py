"""Create III-A time-reversal negatives with FFmpeg."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="JSON list of source video paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def reverse_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-vf", "reverse",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", str(destination),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    for raw_path in tqdm(selected, unit="video"):
        source = Path(raw_path)
        if not source.exists():
            print(f"missing: {source}")
            continue
        destination = args.output_dir / f"{source.stem}_rev.mp4"
        reverse_video(source, destination)


if __name__ == "__main__":
    main()
