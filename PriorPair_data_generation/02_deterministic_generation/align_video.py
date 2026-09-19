import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


LABEL_FILES = (
    "I_A_Coherence_Violation_labels.json",
    "I_B_Causal_Reversal_labels.json",
    "II_A_Existence_Violation_labels.json",
    "II_B_Identity_Violation_labels.json",
    "III_A_Dynamic_Violation_labels.json",
    "III_B_Constraint_Violation_labels.json",
    "IV_A_Near_Miss_labels.json",
    "IV_B_Consequence_Arrest_labels.json",
)

COPY_DURATION_TOLERANCE_SECONDS = 0.005
START_TIME_TOLERANCE_SECONDS = 0.005


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    stream_duration: float
    frame_count: int
    fps: Fraction
    start_time: float
    time_base: Fraction

    @property
    def frame_duration(self):
        return float(1 / self.fps)


@dataclass(frozen=True)
class AlignmentTask:
    label_file: str
    filename: str
    positive_path: Path
    negative_path: Path
    output_path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create *_negative_aligned videos referenced by the PriorPair labels. "
            "Only timestamps are scaled; encoded video frames are stream-copied."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root containing labels and category folders (default: script directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and classify all pairs without creating directories or videos.",
    )
    parser.add_argument(
        "--label-file",
        action="append",
        choices=LABEL_FILES,
        help=(
            "Process only the selected label file. Repeat this option to "
            "select multiple categories. By default, all label files are used."
        ),
    )
    parser.add_argument(
        "--manifest-name",
        default="alignment_manifest.json",
        help=(
            "Manifest filename written inside --root "
            "(default: alignment_manifest.json)."
        ),
    )
    return parser.parse_args()


def require_executable(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Cannot find {name} in PATH.")
    return path


def parse_fraction(value, field, video_path):
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"Invalid {field} for {video_path}: {value!r}") from exc
    if parsed <= 0:
        raise RuntimeError(f"Non-positive {field} for {video_path}: {value!r}")
    return parsed


def run_probe(ffprobe, video_path, count_frames=False):
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
    command.extend(
        [
            "-show_entries",
            "stream=duration,duration_ts,time_base,avg_frame_rate,nb_frames,"
            "nb_read_frames,start_time",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffprobe returned no diagnostic output"
        raise RuntimeError(f"Cannot probe {video_path}: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {video_path}") from exc


def probe_video(ffprobe, video_path, force_count=False):
    data = run_probe(ffprobe, video_path, count_frames=force_count)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]

    fps = parse_fraction(stream.get("avg_frame_rate", "0/0"), "frame rate", video_path)
    time_base = parse_fraction(stream.get("time_base", "0/0"), "time base", video_path)

    # VideoLLaMA3's FFmpeg loader constructs frame timestamps from
    # probe["format"]["duration"], so alignment must use that same value.
    duration = None
    if data.get("format", {}).get("duration") not in (None, "N/A"):
        duration = float(data["format"]["duration"])
    elif stream.get("duration") not in (None, "N/A"):
        duration = float(stream["duration"])
    elif stream.get("duration_ts") not in (None, "N/A"):
        duration = int(stream["duration_ts"]) * float(time_base)
    if duration is None or duration <= 0:
        raise RuntimeError(f"Cannot determine a positive video duration for {video_path}")

    stream_duration = None
    if stream.get("duration") not in (None, "N/A"):
        stream_duration = float(stream["duration"])
    elif stream.get("duration_ts") not in (None, "N/A"):
        stream_duration = int(stream["duration_ts"]) * float(time_base)

    frame_count = None
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value not in (None, "N/A"):
            frame_count = int(value)
            break
    if frame_count is None and not force_count:
        return probe_video(ffprobe, video_path, force_count=True)
    if frame_count is None or frame_count <= 0:
        raise RuntimeError(f"Cannot determine a positive frame count for {video_path}")

    if stream_duration is None:
        stream_duration = frame_count / float(fps)
    if stream_duration <= 0:
        raise RuntimeError(f"Cannot determine a positive video-stream duration for {video_path}")

    start_value = stream.get("start_time")
    start_time = 0.0 if start_value in (None, "N/A") else float(start_value)
    return VideoInfo(duration, stream_duration, frame_count, fps, start_time, time_base)


def resolve_inside(root, relative_path):
    candidate = (root / Path(relative_path.replace("/", os.sep))).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Label path escapes dataset root: {relative_path}") from exc
    return candidate


def source_negative_path(root, aligned_relative_path):
    parts = list(Path(aligned_relative_path.replace("/", os.sep)).parts)
    if not parts or not parts[0].endswith("_negative_aligned"):
        raise RuntimeError(
            f"Expected a *_negative_aligned label path, got: {aligned_relative_path}"
        )
    parts[0] = parts[0][: -len("_aligned")]
    return resolve_inside(root, str(Path(*parts)))


def build_tasks(root, label_files=LABEL_FILES):
    tasks = []
    seen_outputs = {}
    for label_name in label_files:
        label_path = root / label_name
        if not label_path.is_file():
            raise RuntimeError(f"Missing label file: {label_path}")
        with label_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise RuntimeError(f"Label root must be a list: {label_path}")

        for record_index, record in enumerate(records):
            pos_relative = record.get("pos_video_path")
            neg_relative = record.get("neg_video_path")
            if not pos_relative or not neg_relative:
                raise RuntimeError(
                    f"Missing pair path in {label_name}, record {record_index}"
                )
            positive_path = resolve_inside(root, pos_relative)
            output_path = resolve_inside(root, neg_relative)
            negative_path = source_negative_path(root, neg_relative)
            filename = record.get("filename") or negative_path.name

            previous = seen_outputs.get(output_path)
            current = (positive_path, negative_path)
            if previous is not None and previous != current:
                raise RuntimeError(f"Conflicting label pairs target the same output: {output_path}")
            if previous is not None:
                raise RuntimeError(f"Duplicate aligned output in labels: {output_path}")
            seen_outputs[output_path] = current
            tasks.append(
                AlignmentTask(
                    label_name,
                    filename,
                    positive_path,
                    negative_path,
                    output_path,
                )
            )
    return tasks


def classify_pair(positive, negative):
    duration_difference = abs(positive.duration - negative.duration)
    start_time_difference = abs(positive.start_time - negative.start_time)
    return (
        "copy"
        if (
            duration_difference <= COPY_DURATION_TOLERANCE_SECONDS
            and start_time_difference <= START_TIME_TOLERANCE_SECONDS
        )
        else "retime"
    ), duration_difference, start_time_difference


def timestamp_scale(source_info, target_info):
    # With stream copy, FFmpeg scales packet timestamps but preserves the source
    # frame count.  Subtract one source-frame interval to obtain the timestamp
    # span between the first and final frames, then solve for the scale that
    # reproduces the target container duration.
    source_span = source_info.stream_duration - source_info.frame_duration
    target_span = target_info.duration - source_info.frame_duration
    if source_span <= 0 or target_span <= 0:
        raise RuntimeError(
            "Cannot compute timestamp scale from a single-frame or too-short video"
        )
    return target_span / source_span


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_is_valid(source, output):
    return (
        output.is_file()
        and source.stat().st_size == output.stat().st_size
        and sha256(source) == sha256(output)
    )


def retimed_output_is_valid(ffprobe, source_info, target_info, output):
    if not output.is_file():
        return False, "output does not exist"
    try:
        output_info = probe_video(ffprobe, output, force_count=True)
    except RuntimeError as exc:
        return False, str(exc)

    if output_info.frame_count != source_info.frame_count:
        return False, (
            f"frame count changed: source={source_info.frame_count}, "
            f"output={output_info.frame_count}"
        )

    duration_tolerance = max(
        COPY_DURATION_TOLERANCE_SECONDS,
        float(target_info.time_base),
        float(output_info.time_base),
    )
    duration_error = abs(output_info.duration - target_info.duration)
    if duration_error > duration_tolerance:
        return False, (
            f"duration mismatch: target={target_info.duration:.6f}s, "
            f"output={output_info.duration:.6f}s, error={duration_error:.6f}s"
        )

    start_time_tolerance = max(
        START_TIME_TOLERANCE_SECONDS,
        float(target_info.time_base),
        float(output_info.time_base),
    )
    start_time_error = abs(output_info.start_time - target_info.start_time)
    if start_time_error > start_time_tolerance:
        return False, (
            f"start-time mismatch: target={target_info.start_time:.6f}s, "
            f"output={output_info.start_time:.6f}s, error={start_time_error:.6f}s"
        )

    return True, "ok"


def copy_unchanged(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.part{output.suffix}")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    if not copy_is_valid(source, temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Hash verification failed after copying {source}")
    os.replace(temporary, output)


def retime_stream(ffmpeg, source, output, ratio, target_start_time):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.part{output.suffix}")
    if temporary.exists():
        temporary.unlink()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-itsscale:v:0",
        f"{ratio:.15g}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-copytb",
        "1",
        "-vsync",
        "0",
        "-output_ts_offset",
        f"{target_start_time:.15g}",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        detail = result.stderr.strip() or "ffmpeg returned no diagnostic output"
        raise RuntimeError(f"FFmpeg failed for {source}: {detail}")
    return temporary


def main():
    args = parse_args()
    root = args.root.resolve()
    label_files = tuple(args.label_file) if args.label_file else LABEL_FILES
    manifest_name = Path(args.manifest_name)
    if manifest_name.name != args.manifest_name:
        raise RuntimeError("--manifest-name must be a filename, not a path")
    ffprobe = require_executable("ffprobe")
    ffmpeg = None if args.dry_run else require_executable("ffmpeg")
    tasks = build_tasks(root, label_files)

    print(f"[*] Dataset root: {root}")
    print(f"[*] Found {len(tasks)} labeled positive/negative pairs.")

    summary = {
        "copy": 0,
        "retime": 0,
        "skipped": 0,
        "written": 0,
        "failed": 0,
    }
    manifest = []
    failures = []

    for index, task in enumerate(tasks, 1):
        try:
            if not task.positive_path.is_file():
                raise RuntimeError(f"Missing positive video: {task.positive_path}")
            if not task.negative_path.is_file():
                raise RuntimeError(f"Missing negative video: {task.negative_path}")

            positive_info = probe_video(ffprobe, task.positive_path)
            negative_info = probe_video(ffprobe, task.negative_path)
            mode, difference, start_time_difference = classify_pair(
                positive_info, negative_info
            )
            summary[mode] += 1
            ratio = (
                1.0
                if mode == "copy"
                else timestamp_scale(negative_info, positive_info)
            )
            output_timestamp_offset = positive_info.start_time
            retime_attempts = 0

            print(
                f"[{index}/{len(tasks)}] {mode.upper():6} {task.filename} | "
                f"pos={positive_info.duration:.3f}s neg={negative_info.duration:.3f}s "
                f"duration_delta={difference:.3f}s "
                f"start_delta={start_time_difference:.3f}s"
            )

            status = "planned"
            if not args.dry_run:
                if mode == "copy":
                    if copy_is_valid(task.negative_path, task.output_path):
                        summary["skipped"] += 1
                        status = "already_valid"
                    else:
                        copy_unchanged(task.negative_path, task.output_path)
                        summary["written"] += 1
                        status = "copied"
                else:
                    valid, _ = retimed_output_is_valid(
                        ffprobe, negative_info, positive_info, task.output_path
                    )
                    if valid:
                        summary["skipped"] += 1
                        status = "already_valid"
                    else:
                        for retime_attempts in range(1, 4):
                            temporary = retime_stream(
                                ffmpeg,
                                task.negative_path,
                                task.output_path,
                                ratio,
                                output_timestamp_offset,
                            )
                            valid, reason = retimed_output_is_valid(
                                ffprobe, negative_info, positive_info, temporary
                            )
                            if valid:
                                break

                            output_info = probe_video(
                                ffprobe, temporary, force_count=True
                            )
                            if output_info.frame_count != negative_info.frame_count:
                                temporary.unlink(missing_ok=True)
                                raise RuntimeError(
                                    f"Validation failed for {task.filename}: {reason}"
                                )

                            ratio *= positive_info.duration / output_info.duration
                            output_timestamp_offset += (
                                positive_info.start_time - output_info.start_time
                            )
                            temporary.unlink(missing_ok=True)
                        else:
                            raise RuntimeError(
                                f"Validation failed for {task.filename}: {reason}"
                            )
                        os.replace(temporary, task.output_path)
                        summary["written"] += 1
                        status = "retimed"

            manifest.append(
                {
                    "label_file": task.label_file,
                    "filename": task.filename,
                    "positive_path": str(task.positive_path.relative_to(root)).replace(os.sep, "/"),
                    "negative_path": str(task.negative_path.relative_to(root)).replace(os.sep, "/"),
                    "output_path": str(task.output_path.relative_to(root)).replace(os.sep, "/"),
                    "mode": mode,
                    "status": status,
                    "positive_duration": positive_info.duration,
                    "negative_duration": negative_info.duration,
                    "duration_difference": difference,
                    "start_time_difference": start_time_difference,
                    "positive_start_time": positive_info.start_time,
                    "negative_start_time": negative_info.start_time,
                    "copy_duration_tolerance": COPY_DURATION_TOLERANCE_SECONDS,
                    "start_time_tolerance": START_TIME_TOLERANCE_SECONDS,
                    "timestamp_scale": ratio,
                    "output_timestamp_offset": output_timestamp_offset,
                    "retime_attempts": retime_attempts,
                    "negative_stream_duration": negative_info.stream_duration,
                    "negative_frame_count": negative_info.frame_count,
                }
            )
        except Exception as exc:
            summary["failed"] += 1
            failures.append(f"{task.label_file} | {task.filename} | {exc}")
            print(f"    ERROR: {exc}")

    print("\n[*] Summary")
    print(f"    unchanged copy pairs : {summary['copy']}")
    print(f"    timestamp-scale pairs: {summary['retime']}")
    if not args.dry_run:
        print(f"    written              : {summary['written']}")
        print(f"    already valid        : {summary['skipped']}")
    print(f"    failed               : {summary['failed']}")

    if args.dry_run:
        print("[*] Dry run complete. No directories or videos were created.")
    else:
        manifest_path = root / manifest_name
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"summary": summary, "items": manifest, "failures": failures},
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
        print(f"[*] Alignment manifest: {manifest_path}")

    if failures:
        print("\nFailed items:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
