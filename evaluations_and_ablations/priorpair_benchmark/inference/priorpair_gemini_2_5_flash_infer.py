import argparse
import os
import time
from pathlib import Path

import cv2
from google import genai
from google.genai import types
from tqdm import tqdm

from priorpair_inference_common import (
    extract_prompt,
    load_resumable_results,
    load_test_samples,
    make_result,
    sample_video_path,
    save_result_map,
    successful_indices,
    uniform_frame_indices,
    validate_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Gemini 2.5 Flash on PriorPair.")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--request-interval", type=float, default=4.0)
    return parser.parse_args()


def make_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def extract_frame_parts(video_path, num_frames):
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or fps <= 0:
            raise ValueError(f"Invalid video metadata: frames={frame_count}, fps={fps}")
        parts = [
            types.Part(
                text=(
                    f"The following are {num_frames} frames uniformly sampled "
                    "from a video in chronological order."
                )
            )
        ]
        for frame_index in uniform_frame_indices(frame_count, num_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Failed to decode frame {frame_index}")
            encoded, buffer = cv2.imencode(".jpg", frame)
            if not encoded:
                raise ValueError(f"Failed to encode frame {frame_index}")
            parts.append(
                types.Part(text=f"[Frame timestamp: {frame_index / fps:.3f}s]")
            )
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=buffer.tobytes(),
                    )
                )
            )
        return parts
    finally:
        capture.release()


def main():
    args = parse_args()
    if args.num_frames <= 0 or args.max_new_tokens <= 0 or args.max_retries <= 0:
        raise ValueError("num_frames, max_new_tokens, and max_retries must be positive")
    if args.request_interval < 0:
        raise ValueError("request_interval must be non-negative")
    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    result_map = load_resumable_results(args.output_path, samples)
    completed = successful_indices(result_map.values())
    client = make_client()

    for sample_index, sample in enumerate(tqdm(samples, desc="Gemini inference")):
        if sample_index in completed:
            continue
        video_path = sample_video_path(sample, args.video_root)
        last_error = None
        for attempt in range(args.max_retries):
            try:
                parts = extract_frame_parts(video_path, args.num_frames)
                parts.append(types.Part(text="Question: " + extract_prompt(sample)))
                response = client.models.generate_content(
                    model=args.model,
                    contents=[types.Content(parts=parts)],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=args.max_new_tokens,
                    ),
                )
                if not response.text:
                    raise ValueError("API returned an empty response")
                result_map[sample_index] = make_result(
                    sample_index,
                    sample,
                    model_output=response.text.strip(),
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt + 1 < args.max_retries:
                    time.sleep(min(10 * (2**attempt), 120))
        if last_error is not None:
            result_map[sample_index] = make_result(
                sample_index,
                sample,
                error=f"{type(last_error).__name__}: {last_error}",
            )
        save_result_map(args.output_path, result_map)
        if args.request_interval > 0:
            time.sleep(args.request_interval)

    results = [result_map[index] for index in sorted(result_map)]
    validate_results(results, samples, require_complete=True)
    errors = sum(result["error"] is not None for result in results)
    print(f"Saved {len(results)} results to {args.output_path}; errors={errors}")


if __name__ == "__main__":
    main()
