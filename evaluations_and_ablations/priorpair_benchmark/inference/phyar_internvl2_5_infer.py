import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator
from decord import VideoReader, cpu
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode


SYSTEM_MESSAGE = (
    "You are a multimodal video analysis assistant, required to answer "
    "user questions based on temporal video clips."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run InternVL2.5-8B inference on the PriorPair test set."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="PEFT adapter directory.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Restrict inference to the first N samples for a smoke test.",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if args.adapter_path is not None:
        if not args.adapter_path.is_dir():
            raise FileNotFoundError(
                f"Adapter directory not found: {args.adapter_path}"
            )
        adapter_config_path = args.adapter_path / "adapter_config.json"
        adapter_weights_path = (
            args.adapter_path / "adapter_model.safetensors"
        )
        if not adapter_config_path.is_file():
            raise FileNotFoundError(
                f"Adapter config not found: {adapter_config_path}"
            )
        if not adapter_weights_path.is_file():
            raise FileNotFoundError(
                f"Adapter weights not found: {adapter_weights_path}"
            )
    if not args.video_root.is_dir():
        raise FileNotFoundError(f"Video root not found: {args.video_root}")
    if not args.test_data_path.is_file():
        raise FileNotFoundError(
            f"Test JSONL not found: {args.test_data_path}"
        )
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(
                lambda image: (
                    image.convert("RGB") if image.mode != "RGB" else image
                )
            ),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio,
    target_ratios,
    width,
    height,
    image_size,
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image,
    min_num=1,
    max_num=1,
    image_size=448,
    use_thumbnail=True,
):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    target_ratios = {
        (width, height)
        for block_count in range(min_num, max_num + 1)
        for width in range(1, block_count + 1)
        for height in range(1, block_count + 1)
        if min_num <= width * height <= max_num
    }
    target_ratios = sorted(
        target_ratios,
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        original_width,
        original_height,
        image_size,
    )
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    block_count = target_ratio[0] * target_ratio[1]
    resized_image = image.resize((target_width, target_height))

    processed_images = []
    blocks_per_row = target_width // image_size
    for index in range(block_count):
        box = (
            (index % blocks_per_row) * image_size,
            (index // blocks_per_row) * image_size,
            ((index % blocks_per_row) + 1) * image_size,
            ((index // blocks_per_row) + 1) * image_size,
        )
        processed_images.append(resized_image.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def get_frame_indices(frame_count, num_segments):
    if frame_count <= 0:
        raise ValueError("Video contains no frames")

    max_frame = frame_count - 1
    segment_size = float(max_frame) / num_segments
    indices = [
        int(segment_size / 2 + np.round(segment_size * index))
        for index in range(num_segments)
    ]
    return np.asarray(indices, dtype=np.int64)


def load_video(
    video_path,
    input_size=448,
    max_num=1,
    num_segments=16,
):
    video_reader = VideoReader(
        str(video_path),
        ctx=cpu(0),
        num_threads=1,
    )
    frame_indices = get_frame_indices(len(video_reader), num_segments)
    transform = build_transform(input_size=input_size)
    pixel_values_list = []
    num_patches_list = []

    for frame_index in frame_indices:
        frame = video_reader[int(frame_index)].asnumpy()
        image = Image.fromarray(frame).convert("RGB")
        tiles = dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    return torch.cat(pixel_values_list), num_patches_list


def read_test_samples(test_data_path, max_samples=None):
    samples = []
    with test_data_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {test_data_path}:{line_number}"
                ) from error
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def extract_prompt(sample):
    for turn in sample.get("conversations", []):
        if turn.get("from") == "human":
            prompt = turn.get("value", "")
            return (
                prompt.replace("<video>\n", "", 1)
                .replace("<video>", "", 1)
                .strip()
            )
    raise ValueError("Sample has no human prompt")


def extract_ground_truth(sample):
    for turn in sample.get("conversations", []):
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    raise ValueError("Sample has no GPT ground truth")


def rank_output_path(output_path, process_index):
    return output_path.with_name(
        f"{output_path.stem}.rank_{process_index}{output_path.suffix}"
    )


def main():
    args = parse_args()
    validate_args(args)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    accelerator = Accelerator()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for InternVL2.5-8B inference")
    torch.cuda.set_device(accelerator.local_process_index)

    if accelerator.is_main_process:
        print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if accelerator.is_main_process:
        print(
            "Loading InternVL2.5-8B in bfloat16 "
            "(FlashAttention disabled)"
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map={"": accelerator.device},
    ).eval()
    base_model.system_message = SYSTEM_MESSAGE

    if args.adapter_path is not None:
        if accelerator.is_main_process:
            print(f"Loading PEFT adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            is_trainable=False,
        ).eval()
    else:
        model = base_model

    test_samples = read_test_samples(
        args.test_data_path,
        max_samples=args.max_samples,
    )
    if accelerator.is_main_process:
        print(
            f"Loaded {len(test_samples)} sample(s) from "
            f"{args.test_data_path}"
        )

    indexed_samples = list(enumerate(test_samples))
    local_results = []
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    with accelerator.split_between_processes(
        indexed_samples
    ) as local_samples:
        progress = tqdm(
            local_samples,
            desc=f"Inference rank {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        )
        for sample_index, sample in progress:
            video_relative_path = sample["video"][0]
            video_path = args.video_root / video_relative_path

            result = {
                "sample_index": sample_index,
                "video": video_relative_path,
                "prompt": None,
                "ground_truth": None,
                "model_output": None,
                "error": None,
            }

            try:
                if not video_path.is_file():
                    raise FileNotFoundError(f"Video not found: {video_path}")

                prompt = extract_prompt(sample)
                ground_truth = extract_ground_truth(sample)
                result["prompt"] = prompt
                result["ground_truth"] = ground_truth

                pixel_values, num_patches_list = load_video(
                    video_path,
                    max_num=1,
                    num_segments=args.num_frames,
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
                question = video_prefix + prompt

                with torch.inference_mode():
                    answer = model.chat(
                        tokenizer,
                        pixel_values,
                        question,
                        generation_config.copy(),
                        num_patches_list=num_patches_list,
                        history=None,
                    )
                result["model_output"] = answer.strip()
                del pixel_values
                torch.cuda.empty_cache()
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                accelerator.print(
                    f"Sample {sample_index} failed ({video_path}): "
                    f"{result['error']}"
                )

            local_results.append(result)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    local_output_path = rank_output_path(
        args.output_path,
        accelerator.process_index,
    )
    with local_output_path.open("w", encoding="utf-8") as output_file:
        json.dump(local_results, output_file, ensure_ascii=False, indent=2)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_results = []
        for process_index in range(accelerator.num_processes):
            process_output_path = rank_output_path(
                args.output_path,
                process_index,
            )
            with process_output_path.open(
                "r",
                encoding="utf-8",
            ) as process_file:
                all_results.extend(json.load(process_file))
            process_output_path.unlink()

        all_results.sort(key=lambda result: result["sample_index"])
        with args.output_path.open("w", encoding="utf-8") as output_file:
            json.dump(all_results, output_file, ensure_ascii=False, indent=2)

        error_count = sum(
            result["error"] is not None for result in all_results
        )
        print(
            f"Saved {len(all_results)} result(s) to {args.output_path}; "
            f"errors={error_count}"
        )


if __name__ == "__main__":
    main()
