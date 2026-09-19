import argparse
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from priorpair_inference_common import (
    atomic_write_json,
    extract_prompt,
    load_test_samples,
    make_result,
    merge_rank_results,
    prepare_new_output,
    sample_video_path,
    uniform_frame_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MiniCPM-V 2.6 on PriorPair.")
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def read_frames(video_path, num_frames):
    video_reader = VideoReader(str(video_path), ctx=cpu(0))
    indices = uniform_frame_indices(len(video_reader), num_frames)
    frames = video_reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame.astype("uint8")) for frame in frames]


def main():
    args = parse_args()
    for path, label in ((args.code_path, "code"), (args.model_path, "model")):
        if not path.is_dir():
            raise FileNotFoundError(f"MiniCPM {label} directory not found: {path}")
    if args.num_frames <= 0 or args.max_new_tokens <= 0:
        raise ValueError("num_frames and max_new_tokens must be positive")
    sys.path.insert(0, str(args.code_path))
    os.chdir(args.code_path)

    accelerator = Accelerator()
    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    local_output_path = prepare_new_output(args.output_path, accelerator.process_index)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},
    ).eval()

    indexed_samples = list(enumerate(samples))
    local_results = []
    with accelerator.split_between_processes(indexed_samples) as local_samples:
        progress = tqdm(
            local_samples,
            desc=f"MiniCPM rank {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        )
        for sample_index, sample in progress:
            video_path = sample_video_path(sample, args.video_root)
            try:
                frames = read_frames(video_path, args.num_frames)
                messages = [
                    {
                        "role": "user",
                        "content": frames + [extract_prompt(sample)],
                    }
                ]
                max_input_length = int(4096 + len(frames) * 199)
                with torch.inference_mode():
                    answer = model.chat(
                        image=None,
                        msgs=messages,
                        tokenizer=tokenizer,
                        max_inp_length=max_input_length,
                        max_new_tokens=args.max_new_tokens,
                        sampling=False,
                        num_beams=1,
                        repetition_penalty=1.0,
                        use_image_id=False,
                        max_slice_nums=2,
                    )
                result = make_result(sample_index, sample, model_output=answer.strip())
            except Exception as error:
                result = make_result(
                    sample_index,
                    sample,
                    error=f"{type(error).__name__}: {error}",
                )
            local_results.append(result)

    atomic_write_json(local_output_path, local_results)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        results = merge_rank_results(
            args.output_path,
            accelerator.num_processes,
            samples,
        )
        errors = sum(result["error"] is not None for result in results)
        print(f"Saved {len(results)} results to {args.output_path}; errors={errors}")


if __name__ == "__main__":
    main()
