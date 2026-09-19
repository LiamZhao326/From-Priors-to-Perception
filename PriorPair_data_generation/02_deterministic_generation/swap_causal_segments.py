"""Create I-B negatives by swapping the effect and cause segments (A|B -> B|A)."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--annotations", type=Path, required=True,
        help="JSON object mapping a video path or filename to the split time in seconds.",
    )
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


def swap_segments(source: Path, destination: Path, split_time: float) -> None:
    duration = duration_seconds(source)
    if not 0 < split_time < duration:
        raise ValueError(f"Split time {split_time} is outside (0, {duration}) for {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="priorpair_causal_") as temp_name:
        temp_dir = Path(temp_name)
        cause = temp_dir / "cause.mp4"
        effect = temp_dir / "effect.mp4"
        common = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-an"]
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", "0", "-t", f"{split_time:.6f}", "-i", str(source),
                *common, str(cause),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{split_time:.6f}", "-i", str(source),
                *common, str(effect),
            ],
            check=True,
        )
        concat_list = temp_dir / "concat.txt"
        concat_list.write_text(
            f"file '{effect.as_posix()}'\nfile '{cause.as_posix()}'\n",
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


def main() -> None:
    args = parse_args()
    raw_annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    split_by_name = {
        Path(path.replace("\\", "/")).name: float(value)
        for path, value in raw_annotations.items()
    }
    sources = [
        path for path in sorted(args.input_dir.glob("*.mp4"))
        if path.name in split_by_name
    ]
    for source in tqdm(sources, unit="video"):
        swap_segments(source, args.output_dir / source.name, split_by_name[source.name])


if __name__ == "__main__":
    main()
