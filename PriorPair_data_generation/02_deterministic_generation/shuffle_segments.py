"""Create I-A negatives by splitting each video and shuffling the segments."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segments-min", type=int, default=3)
    parser.add_argument("--segments-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def duration_seconds(path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
    )
    return float(output.strip())


def shuffle_video(source: Path, destination: Path, segment_count: int, rng: random.Random) -> list[int]:
    duration = duration_seconds(source)
    if duration < 1.0:
        raise ValueError(f"Video is shorter than one second: {source}")

    original_order = list(range(segment_count))
    order = original_order.copy()
    while order == original_order:
        rng.shuffle(order)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="priorpair_shuffle_") as temp_name:
        temp_dir = Path(temp_name)
        segment_duration = duration / segment_count
        parts: list[Path] = []
        for index in range(segment_count):
            part = temp_dir / f"part_{index:03d}.mp4"
            command = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{index * segment_duration:.6f}", "-i", str(source),
            ]
            if index < segment_count - 1:
                command.extend(["-t", f"{segment_duration:.6f}"])
            command.extend(
                ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-an", str(part)]
            )
            subprocess.run(command, check=True)
            parts.append(part)

        concat_list = temp_dir / "concat.txt"
        concat_list.write_text(
            "".join(f"file '{parts[index].as_posix()}'\n" for index in order),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c", "copy", str(destination),
            ],
            check=True,
        )
    return order


def main() -> None:
    args = parse_args()
    if args.segments_min < 2 or args.segments_min > args.segments_max:
        raise ValueError("Invalid segment-count range")
    rng = random.Random(args.seed)
    videos = sorted(args.input_dir.glob("*.mp4"))
    manifest = []
    for source in tqdm(videos, unit="video"):
        segment_count = rng.randint(args.segments_min, args.segments_max)
        destination = args.output_dir / source.name
        order = shuffle_video(source, destination, segment_count, rng)
        manifest.append(
            {
                "source": str(source.resolve()),
                "output": str(destination.resolve()),
                "segment_count": segment_count,
                "order": order,
            }
        )
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
