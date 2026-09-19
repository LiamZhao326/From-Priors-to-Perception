import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from priorpair_inference_common import (
    append_full_oav_format_requirement,
    atomic_write_json,
    extract_prompt,
    load_test_samples,
    make_result,
    merge_rank_results,
    prepare_new_output,
    sample_video_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VideoLLaMA3-7B on PriorPair.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(args, accelerator):
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["vision_encoder", "mm_projector"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
        device_map={"": accelerator.device},
    )
    model.eval()
    model.config.use_cache = True
    return model, processor


def generate_response(model, processor, video_path, prompt, args, device):
    prompt = append_full_oav_format_requirement(prompt)
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": str(video_path),
                        "nframes": args.num_frames,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = {
        key: (
            value.to(
                device=device,
                dtype=(
                    torch.bfloat16
                    if torch.is_floating_point(value)
                    else value.dtype
                ),
            )
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if args.num_frames <= 0 or args.max_new_tokens <= 0:
        raise ValueError("num_frames and max_new_tokens must be positive")

    accelerator = Accelerator()
    set_seed(args.seed, device_specific=False)
    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    local_output_path = prepare_new_output(args.output_path, accelerator.process_index)
    model, processor = load_model(args, accelerator)

    indexed_samples = list(enumerate(samples))
    local_results = []
    with accelerator.split_between_processes(indexed_samples) as local_samples:
        progress = tqdm(
            local_samples,
            desc=f"VideoLLaMA3 rank {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        )
        for sample_index, sample in progress:
            video_path = sample_video_path(sample, args.video_root)
            try:
                response = generate_response(
                    model,
                    processor,
                    video_path,
                    extract_prompt(sample),
                    args,
                    accelerator.device,
                )
                result = make_result(sample_index, sample, model_output=response)
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
