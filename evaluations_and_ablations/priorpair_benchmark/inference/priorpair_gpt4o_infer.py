import argparse
import base64
import os
import time
from pathlib import Path

import cv2
from openai import OpenAI
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
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o on PriorPair.")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--request-interval", type=float, default=0.0)
    return parser.parse_args()


def make_client(request_timeout):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client_args = {
        "api_key": api_key,
        "timeout": request_timeout,
        "max_retries": 0,
    }
    if base_url:
        client_args["base_url"] = base_url
    return OpenAI(**client_args)


def extract_frame_content(video_path, num_frames, image_detail):
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or fps <= 0:
            raise ValueError(f"Invalid video metadata: frames={frame_count}, fps={fps}")
        content = [
            {
                "type": "text",
                "text": (
                    f"The following are {num_frames} frames uniformly sampled "
                    "from a video in chronological order."
                ),
            }
        ]
        for frame_index in uniform_frame_indices(frame_count, num_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Failed to decode frame {frame_index}")
            encoded, buffer = cv2.imencode(".jpg", frame)
            if not encoded:
                raise ValueError(f"Failed to encode frame {frame_index}")
            timestamp = frame_index / fps
            content.append(
                {"type": "text", "text": f"[Frame timestamp: {timestamp:.3f}s]"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(buffer).decode("ascii"),
                        "detail": image_detail,
                    },
                }
            )
        return content
    finally:
        capture.release()


def main():
    args = parse_args()
    if (
        args.num_frames <= 0
        or args.max_new_tokens <= 0
        or args.max_retries <= 0
        or args.request_timeout <= 0
    ):
        raise ValueError(
            "num_frames, max_new_tokens, max_retries, and request_timeout "
            "must be positive"
        )
    if args.request_interval < 0:
        raise ValueError("request_interval must be non-negative")
    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    result_map = load_resumable_results(args.output_path, samples)
    completed = successful_indices(result_map.values())
    client = make_client(args.request_timeout)

    for sample_index, sample in enumerate(tqdm(samples, desc="GPT-4o inference")):
        if sample_index in completed:
            continue
        video_path = sample_video_path(sample, args.video_root)
        last_error = None
        for attempt in range(args.max_retries):
            try:
                content = extract_frame_content(
                    video_path,
                    args.num_frames,
                    args.image_detail,
                )
                content.append(
                    {"type": "text", "text": "Question: " + extract_prompt(sample)}
                )
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
                response_text = response.choices[0].message.content
                if not response_text:
                    raise ValueError("API returned an empty response")
                result_map[sample_index] = make_result(
                    sample_index,
                    sample,
                    model_output=response_text.strip(),
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt + 1 < args.max_retries:
                    time.sleep(min(5 * (2**attempt), 120))
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
