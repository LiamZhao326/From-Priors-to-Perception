import argparse
import os
import sys
from pathlib import Path


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
VISIBLE_DEVICES = [
    device.strip()
    for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if device.strip()
]
if WORLD_SIZE > 1:
    if VISIBLE_DEVICES:
        if LOCAL_RANK >= len(VISIBLE_DEVICES):
            raise RuntimeError("LOCAL_RANK exceeds CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES[LOCAL_RANK]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL_RANK)

import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from priorpair_inference_common import (
    append_full_oav_format_requirement,
    assigned_indices,
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
    parser = argparse.ArgumentParser(description="Evaluate Flash-VStream-7B on PriorPair.")
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--vision-tower-path",
        type=Path,
        required=True,
    )
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
    return video_reader.get_batch(indices).asnumpy()


def main():
    args = parse_args()
    for path, label in (
        (args.code_path, "code"),
        (args.model_path, "model"),
        (args.vision_tower_path, "vision tower"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Flash-VStream {label} directory not found: {path}")
    if args.num_frames <= 0 or args.max_new_tokens <= 0:
        raise ValueError("num_frames and max_new_tokens must be positive")

    sys.path.insert(0, str(args.code_path))
    os.chdir(args.code_path)
    from flash_vstream.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from flash_vstream.conversation import conv_templates
    from flash_vstream.mm_utils import get_model_name_from_path, tokenizer_image_token
    from flash_vstream.model.builder import load_pretrained_model
    from flash_vstream.utils import disable_torch_init
    from transformers import AutoConfig

    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    local_output_path = prepare_new_output(args.output_path, LOCAL_RANK)
    disable_torch_init()
    config = AutoConfig.from_pretrained(args.model_path)
    config.mm_vision_tower = str(args.vision_tower_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=str(args.model_path),
        model_base=None,
        model_name=get_model_name_from_path(str(args.model_path)),
        load_8bit=False,
        load_4bit=False,
        device="cuda",
        config=config,
    )
    model.use_video_streaming_mode = True
    model.eval()

    local_results = []
    indices = assigned_indices(len(samples), LOCAL_RANK, WORLD_SIZE)
    for sample_index in tqdm(indices, desc=f"Flash-VStream rank {LOCAL_RANK}"):
        sample = samples[sample_index]
        video_path = sample_video_path(sample, args.video_root)
        try:
            frames = read_frames(video_path, args.num_frames)
            image_tensor = image_processor.preprocess(
                frames,
                return_tensors="pt",
            )["pixel_values"].to("cuda", dtype=torch.float16)
            model.video_embedding_memory = []
            with torch.inference_mode():
                model.embed_video_streaming(image_tensor.unsqueeze(0))

            conversation = conv_templates["vicuna_v1"].copy()
            conversation.append_message(
                conversation.roles[0],
                DEFAULT_IMAGE_TOKEN
                + "\n"
                + append_full_oav_format_requirement(extract_prompt(sample)),
            )
            conversation.append_message(conversation.roles[1], None)
            input_ids = tokenizer_image_token(
                conversation.get_prompt(),
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to("cuda")
            input_length = input_ids.shape[1]
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=None,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            generated_ids = (
                output_ids[:, input_length:]
                if output_ids.shape[1] > input_length
                else output_ids
            )
            response = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()
            result = make_result(sample_index, sample, model_output=response)
            del frames, image_tensor, input_ids, output_ids
            torch.cuda.empty_cache()
        except Exception as error:
            result = make_result(
                sample_index,
                sample,
                error=f"{type(error).__name__}: {error}",
            )
        local_results.append(result)

    atomic_write_json(local_output_path, local_results)
    if LOCAL_RANK == 0:
        results = merge_rank_results(args.output_path, WORLD_SIZE, samples)
        errors = sum(result["error"] is not None for result in results)
        print(f"Saved {len(results)} results to {args.output_path}; errors={errors}")


if __name__ == "__main__":
    main()
