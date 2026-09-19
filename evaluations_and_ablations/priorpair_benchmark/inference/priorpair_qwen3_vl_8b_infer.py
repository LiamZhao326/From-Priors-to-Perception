import argparse
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from priorpair_inference_common import (
    extract_prompt,
    load_resumable_results,
    load_test_samples,
    make_result,
    sample_video_path,
    save_result_map,
    successful_indices,
    validate_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-8B-Instruct on PriorPair."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--device-map-strategy",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        default="auto",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Qwen model directory not found: {args.model_path}")
    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    result_map = load_resumable_results(args.output_path, samples)
    completed = successful_indices(result_map.values())

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device_map_strategy,
        trust_remote_code=True,
    ).eval()
    print(f"Qwen3-VL device map strategy: {args.device_map_strategy}")
    print(f"Qwen3-VL device map: {model.hf_device_map}")

    for sample_index, sample in enumerate(tqdm(samples, desc="Qwen3-VL inference")):
        if sample_index in completed:
            continue
        video_path = sample_video_path(sample, args.video_root)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(video_path),
                            "max_pixels": 640 * 360,
                            "nframes": args.num_frames,
                        },
                        {"type": "text", "text": extract_prompt(sample)},
                    ],
                }
            ]
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not args.disable_thinking,
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if video_inputs is not None:
                video_inputs, video_metadata = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadata = list(video_metadata)
            else:
                video_metadata = None
            inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata,
                padding=True,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            ).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            generated_ids = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs.input_ids, output_ids)
            ]
            response = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            result_map[sample_index] = make_result(
                sample_index,
                sample,
                model_output=response,
            )
        except Exception as error:
            result_map[sample_index] = make_result(
                sample_index,
                sample,
                error=f"{type(error).__name__}: {error}",
            )
        save_result_map(args.output_path, result_map)

    results = [result_map[index] for index in sorted(result_map)]
    validate_results(results, samples, require_complete=True)
    errors = sum(result["error"] is not None for result in results)
    print(f"Saved {len(results)} results to {args.output_path}; errors={errors}")


if __name__ == "__main__":
    main()
