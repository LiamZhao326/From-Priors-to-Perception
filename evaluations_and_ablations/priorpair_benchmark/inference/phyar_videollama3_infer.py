import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run VideoLLaMA3-7B with a PriorPair PEFT adapter on the PriorPair "
            "test set."
        )
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args):
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not args.adapter_path.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {args.adapter_path}"
        )
    for file_name in ("adapter_config.json", "adapter_model.safetensors"):
        adapter_file = args.adapter_path / file_name
        if not adapter_file.is_file():
            raise FileNotFoundError(f"Adapter file not found: {adapter_file}")
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


def extract_turn(sample, role):
    for turn in sample.get("conversations", []):
        if turn.get("from") == role:
            value = turn.get("value", "")
            if value:
                return value
    raise ValueError(f"Sample has no non-empty {role} turn")


def extract_prompt(sample):
    prompt = extract_turn(sample, "human")
    return (
        prompt.replace("<video>\n", "", 1)
        .replace("<video>", "", 1)
        .strip()
    )


def read_test_samples(test_data_path, video_root, max_samples=None):
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

            videos = sample.get("video")
            if not isinstance(videos, list) or len(videos) != 1:
                raise ValueError(
                    f"Expected one video at {test_data_path}:{line_number}"
                )
            video_path = video_root / videos[0]
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video not found at {test_data_path}:{line_number}: "
                    f"{video_path}"
                )
            extract_prompt(sample)
            extract_turn(sample, "gpt")
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break
    if not samples:
        raise ValueError("No test samples were loaded")
    return samples


def rank_output_path(output_path, process_index):
    return output_path.with_name(
        f"{output_path.stem}.rank_{process_index}{output_path.suffix}"
    )


def prepare_output_paths(args, accelerator):
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {args.output_path}"
        )
    local_output_path = rank_output_path(
        args.output_path,
        accelerator.process_index,
    )
    if local_output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite stale rank result: {local_output_path}"
        )
    accelerator.wait_for_everyone()
    return local_output_path


def load_model(args, accelerator):
    accelerator.print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = (
            processor.tokenizer.eos_token_id
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=[
            "vision_encoder",
            "mm_projector",
        ],
    )
    accelerator.print(
        "Loading VideoLLaMA3-7B in 4-bit NF4 with bfloat16 compute"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
        device_map={"": accelerator.device},
    )
    accelerator.print(f"Loading PEFT adapter from {args.adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    return model, processor


def prepare_inputs(sample, video_path, processor, args, device):
    prompt = extract_prompt(sample)
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
                non_blocking=(device.type == "cuda"),
            )
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }
    return prompt, inputs


def generate_response(model, processor, inputs, args):
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    if output_ids.shape[1] > input_length:
        generated_ids = output_ids[:, input_length:]
    else:
        generated_ids = output_ids
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def run_inference(args, samples, model, processor, accelerator):
    indexed_samples = list(enumerate(samples))
    local_results = []
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
                prompt, inputs = prepare_inputs(
                    sample,
                    video_path,
                    processor,
                    args,
                    accelerator.device,
                )
                result["prompt"] = prompt
                result["ground_truth"] = extract_turn(sample, "gpt")
                result["model_output"] = generate_response(
                    model,
                    processor,
                    inputs,
                    args,
                )
                del inputs
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                accelerator.print(
                    f"Sample {sample_index} failed ({video_path}): "
                    f"{result['error']}"
                )
            local_results.append(result)
    return local_results


def merge_results(args, samples, accelerator):
    all_results = []
    rank_paths = []
    for process_index in range(accelerator.num_processes):
        process_output_path = rank_output_path(
            args.output_path,
            process_index,
        )
        if not process_output_path.is_file():
            raise FileNotFoundError(
                f"Rank output is missing: {process_output_path}"
            )
        with process_output_path.open(
            "r",
            encoding="utf-8",
        ) as process_file:
            all_results.extend(json.load(process_file))
        rank_paths.append(process_output_path)

    all_results.sort(key=lambda result: result["sample_index"])
    actual_indices = [result["sample_index"] for result in all_results]
    expected_indices = list(range(len(samples)))
    if actual_indices != expected_indices:
        raise RuntimeError(
            "Merged results do not cover every test sample exactly once"
        )

    with args.output_path.open("w", encoding="utf-8") as output_file:
        json.dump(all_results, output_file, ensure_ascii=False, indent=2)
    for rank_path in rank_paths:
        rank_path.unlink()

    error_count = sum(result["error"] is not None for result in all_results)
    print(
        f"Saved {len(all_results)} result(s) to {args.output_path}; "
        f"errors={error_count}"
    )
    if error_count:
        raise RuntimeError(
            f"Inference completed with {error_count} failed sample(s)"
        )


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    accelerator = Accelerator()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VideoLLaMA3 inference")
    torch.cuda.set_device(accelerator.local_process_index)

    samples = read_test_samples(
        args.test_data_path,
        args.video_root,
        max_samples=args.max_samples,
    )
    accelerator.print(
        f"Loaded {len(samples)} sample(s) from {args.test_data_path}; "
        f"frames={args.num_frames}; adapter={args.adapter_path}"
    )
    local_output_path = prepare_output_paths(args, accelerator)
    model, processor = load_model(args, accelerator)
    local_results = run_inference(
        args,
        samples,
        model,
        processor,
        accelerator,
    )

    with local_output_path.open("w", encoding="utf-8") as output_file:
        json.dump(local_results, output_file, ensure_ascii=False, indent=2)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merge_results(args, samples, accelerator)


if __name__ == "__main__":
    main()
