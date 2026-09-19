import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from accelerate import Accelerator
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

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


SYSTEM_MESSAGE = (
    "You are a multimodal video analysis assistant, required to answer "
    "user questions based on temporal video clips."
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate InternVL2.5-8B on PriorPair.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def build_transform(input_size=448):
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB")),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_video(video_path, num_frames, input_size=448):
    video_reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    frame_indices = uniform_frame_indices(len(video_reader), num_frames)
    transform = build_transform(input_size)
    pixel_values = []
    num_patches_list = []
    for frame_index in frame_indices:
        image = Image.fromarray(video_reader[frame_index].asnumpy()).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        pixel_values.append(tensor)
        num_patches_list.append(1)
    return torch.cat(pixel_values), num_patches_list


def main():
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if args.num_frames <= 0 or args.max_new_tokens <= 0:
        raise ValueError("num_frames and max_new_tokens must be positive")

    accelerator = Accelerator()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InternVL2.5 inference")
    torch.cuda.set_device(accelerator.local_process_index)

    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    local_output_path = prepare_new_output(args.output_path, accelerator.process_index)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map={"": accelerator.device},
    ).eval()
    model.system_message = SYSTEM_MESSAGE
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    indexed_samples = list(enumerate(samples))
    local_results = []
    with accelerator.split_between_processes(indexed_samples) as local_samples:
        progress = tqdm(
            local_samples,
            desc=f"InternVL2.5 rank {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        )
        for sample_index, sample in progress:
            video_path = sample_video_path(sample, args.video_root)
            try:
                pixel_values, num_patches_list = load_video(
                    video_path,
                    args.num_frames,
                )
                pixel_values = pixel_values.to(
                    device=accelerator.device,
                    dtype=torch.bfloat16,
                    non_blocking=True,
                )
                video_prefix = "".join(
                    f"Frame{index + 1}: <image>\n"
                    for index in range(len(num_patches_list))
                )
                with torch.inference_mode():
                    answer = model.chat(
                        tokenizer,
                        pixel_values,
                        video_prefix + extract_prompt(sample),
                        generation_config.copy(),
                        num_patches_list=num_patches_list,
                        history=None,
                    )
                result = make_result(sample_index, sample, model_output=answer.strip())
                del pixel_values
                torch.cuda.empty_cache()
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
