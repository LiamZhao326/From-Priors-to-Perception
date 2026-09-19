import argparse
import os
import sys
from functools import partial
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
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VideoLLaMA2-7B-16F on PriorPair.")
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


def main():
    args = parse_args()
    for path, label in (
        (args.code_path, "code"),
        (args.model_path, "model"),
        (args.vision_tower_path, "vision tower"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"VideoLLaMA2 {label} directory not found: {path}")
    if args.num_frames <= 0 or args.max_new_tokens <= 0:
        raise ValueError("num_frames and max_new_tokens must be positive")

    sys.path.insert(0, str(args.code_path))
    from videollama2.constants import DEFAULT_VIDEO_TOKEN
    from videollama2.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_video,
        tokenizer_multimodal_token,
    )
    import videollama2.model as videollama2_model
    from videollama2.model import load_pretrained_model
    from videollama2.utils import disable_torch_init

    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    local_output_path = prepare_new_output(args.output_path, LOCAL_RANK)
    disable_torch_init()
    original_auto_config = videollama2_model.AutoConfig

    class LocalVisionAutoConfig:
        @classmethod
        def from_pretrained(cls, *config_args, **config_kwargs):
            config = original_auto_config.from_pretrained(
                *config_args,
                **config_kwargs,
            )
            config.mm_vision_tower = str(args.vision_tower_path)
            return config

    videollama2_model.AutoConfig = LocalVisionAutoConfig
    try:
        tokenizer, model, processor, _ = load_pretrained_model(
            model_path=str(args.model_path),
            model_base=None,
            model_name=get_model_name_from_path(str(args.model_path)),
            torch_dtype=torch.float16,
            use_flash_attn=True,
            device_map={"": "cuda"},
        )
    finally:
        videollama2_model.AutoConfig = original_auto_config
    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    model.half().eval()
    video_processor = partial(
        process_video,
        processor=processor,
        aspect_ratio=None,
        num_frames=args.num_frames,
    )
    system_message = (
        "<<SYS>>\nYou are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe.\n<</SYS>>"
    )

    local_results = []
    indices = assigned_indices(len(samples), LOCAL_RANK, WORLD_SIZE)
    for sample_index in tqdm(indices, desc=f"VideoLLaMA2 rank {LOCAL_RANK}"):
        sample = samples[sample_index]
        video_path = sample_video_path(sample, args.video_root)
        try:
            tensor = video_processor(str(video_path)).to("cuda", dtype=torch.float16)
            video_tensor = [(tensor, "video")]
            conversation = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": DEFAULT_VIDEO_TOKEN
                    + "\n"
                    + append_full_oav_format_requirement(extract_prompt(sample)),
                },
            ]
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer_multimodal_token(
                prompt,
                tokenizer,
                DEFAULT_VIDEO_TOKEN,
                return_tensors="pt",
            ).unsqueeze(0).long().to("cuda")
            attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
            stopping_criteria = KeywordsStoppingCriteria(
                [tokenizer.eos_token],
                tokenizer,
                input_ids,
            )
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=video_tensor,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )[0].strip()
            result = make_result(sample_index, sample, model_output=response)
            del tensor, video_tensor, input_ids, attention_mask, output_ids
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
