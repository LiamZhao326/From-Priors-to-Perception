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
    parser = argparse.ArgumentParser(description="Evaluate Video-ChatGPT-7B on PriorPair.")
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--projection-path", type=Path, required=True)
    parser.add_argument("--vision-tower-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def pool_spatio_temporal_features(features):
    time_steps, _, channels = features.shape
    temporal_tokens = torch.mean(features, dim=1)
    padding_size = 100 - time_steps
    if padding_size < 0:
        raise ValueError("Video-ChatGPT supports at most 100 temporal tokens")
    if padding_size:
        temporal_tokens = torch.cat(
            (
                temporal_tokens,
                torch.zeros(
                    padding_size,
                    channels,
                    device=features.device,
                    dtype=features.dtype,
                ),
            ),
            dim=0,
        )
    spatial_tokens = torch.mean(features, dim=0)
    return torch.cat([temporal_tokens, spatial_tokens], dim=0).half()


def main():
    args = parse_args()
    for path, label in (
        (args.code_path, "code"),
        (args.model_path, "model"),
        (args.projection_path, "projection"),
        (args.vision_tower_path, "vision tower"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Video-ChatGPT {label} path not found: {path}")
    if args.num_frames <= 0 or args.num_frames > 100:
        raise ValueError("num_frames must be in [1, 100]")

    sys.path.insert(0, str(args.code_path))
    from video_chatgpt.constants import (
        DEFAULT_VIDEO_PATCH_TOKEN,
        DEFAULT_VIDEO_TOKEN,
        DEFAULT_VID_END_TOKEN,
        DEFAULT_VID_START_TOKEN,
    )
    from video_chatgpt.eval import model_utils as video_chatgpt_model_utils
    from video_chatgpt.model.utils import KeywordsStoppingCriteria
    from video_chatgpt.utils import disable_torch_init
    from video_chatgpt.video_conversation import SeparatorStyle, conv_templates

    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    local_output_path = prepare_new_output(args.output_path, LOCAL_RANK)
    disable_torch_init()

    original_torch_load = torch.load
    original_clip_vision_model = video_chatgpt_model_utils.CLIPVisionModel
    original_clip_image_processor = video_chatgpt_model_utils.CLIPImageProcessor
    try:
        def safe_load(*load_args, **load_kwargs):
            load_kwargs["map_location"] = "cpu"
            return original_torch_load(*load_args, **load_kwargs)

        class LocalCLIPVisionModel:
            @staticmethod
            def from_pretrained(*load_args, **load_kwargs):
                return original_clip_vision_model.from_pretrained(
                    str(args.vision_tower_path),
                    *load_args[1:],
                    **load_kwargs,
                )

        class LocalCLIPImageProcessor:
            @staticmethod
            def from_pretrained(*load_args, **load_kwargs):
                return original_clip_image_processor.from_pretrained(
                    str(args.vision_tower_path),
                    *load_args[1:],
                    **load_kwargs,
                )

        torch.load = safe_load
        video_chatgpt_model_utils.CLIPVisionModel = LocalCLIPVisionModel
        video_chatgpt_model_utils.CLIPImageProcessor = LocalCLIPImageProcessor
        model, vision_tower, tokenizer, image_processor, video_token_len = (
            video_chatgpt_model_utils.initialize_model(
                str(args.model_path),
                str(args.projection_path),
                device="cuda",
            )
        )
    finally:
        torch.load = original_torch_load
        video_chatgpt_model_utils.CLIPVisionModel = original_clip_vision_model
        video_chatgpt_model_utils.CLIPImageProcessor = original_clip_image_processor

    model.to("cuda").half().eval()
    vision_tower.to("cuda").half().eval()
    replace_token = DEFAULT_VID_START_TOKEN + (
        DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    ) + DEFAULT_VID_END_TOKEN

    local_results = []
    indices = assigned_indices(len(samples), LOCAL_RANK, WORLD_SIZE)
    for sample_index in tqdm(indices, desc=f"Video-ChatGPT rank {LOCAL_RANK}"):
        sample = samples[sample_index]
        video_path = sample_video_path(sample, args.video_root)
        try:
            video_reader = VideoReader(str(video_path), ctx=cpu(0))
            frame_indices = uniform_frame_indices(len(video_reader), args.num_frames)
            frames = video_reader.get_batch(frame_indices).asnumpy()
            image_tensor = image_processor.preprocess(
                list(frames),
                return_tensors="pt",
            )["pixel_values"].half().to("cuda")

            feature_chunks = []
            with torch.inference_mode():
                for start in range(0, image_tensor.shape[0], 16):
                    hidden_states = vision_tower(
                        image_tensor[start : start + 16],
                        output_hidden_states=True,
                    ).hidden_states[-2][:, 1:]
                    feature_chunks.append(hidden_states)
            video_features = pool_spatio_temporal_features(
                torch.cat(feature_chunks, dim=0)
            )

            state = conv_templates["video-chatgpt_v1"].copy()
            state.append_message(
                state.roles[0],
                append_full_oav_format_requirement(extract_prompt(sample))
                + "\n"
                + DEFAULT_VIDEO_TOKEN,
            )
            state.append_message(state.roles[1], None)
            prompt = state.get_prompt().replace(DEFAULT_VIDEO_TOKEN, replace_token, 1)
            input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).to("cuda")
            stop_string = state.sep if state.sep_style != SeparatorStyle.TWO else state.sep2
            stopping_criteria = KeywordsStoppingCriteria(
                [stop_string],
                tokenizer,
                input_ids,
            )
            input_length = input_ids.shape[1]
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    video_spatio_temporal_features=video_features.unsqueeze(0),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=[stopping_criteria],
                )
            response = tokenizer.batch_decode(
                output_ids[:, input_length:],
                skip_special_tokens=True,
            )[0].strip()
            if response.endswith(stop_string):
                response = response[: -len(stop_string)].strip()
            result = make_result(sample_index, sample, model_output=response)
            del image_tensor, video_features, input_ids, output_ids
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
