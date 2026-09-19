import argparse
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from tqdm import tqdm

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


NATIVE_NUM_FRAMES = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Video-LLaVA-7B on PriorPair.")
    parser.add_argument("--code-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    for path, label in ((args.code_path, "code"), (args.model_path, "model")):
        if not path.is_dir():
            raise FileNotFoundError(f"Video-LLaVA {label} directory not found: {path}")
    sys.path.insert(0, str(args.code_path))
    from transformers.modeling_attn_mask_utils import (
        AttentionMaskConverter,
        _create_4d_causal_attention_mask,
        _prepare_4d_attention_mask,
    )
    from transformers.models.bloom import modeling_bloom
    from transformers.models.clip import modeling_clip
    from transformers.models.opt import modeling_opt
    from transformers import AutoConfig

    def bloom_expand_mask(mask, tgt_length):
        batch_size, source_length = mask.shape
        expanded_mask = ~mask[:, None, None, :].to(torch.bool)
        return expanded_mask.expand(
            batch_size,
            1,
            tgt_length,
            source_length,
        )

    def bloom_make_causal_mask(
        input_shape,
        device,
        past_key_values_length=0,
    ):
        batch_size, target_length = input_shape
        mask = torch.ones(
            (target_length, target_length + past_key_values_length),
            dtype=torch.bool,
            device=device,
        )
        if past_key_values_length:
            mask[:, :past_key_values_length] = False
        mask[:, past_key_values_length:] = torch.triu(
            mask[:, past_key_values_length:],
            diagonal=1,
        )
        return mask[None, None, :, :].expand(
            batch_size,
            1,
            target_length,
            target_length + past_key_values_length,
        )

    def opt_expand_mask(mask, dtype, tgt_len=None):
        return AttentionMaskConverter._expand_mask(
            mask=mask,
            dtype=dtype,
            tgt_len=tgt_len,
        )

    def opt_make_causal_mask(
        input_shape,
        dtype,
        device=None,
        past_key_values_length=0,
    ):
        if device is None:
            device = torch.device("cpu")
        return _create_4d_causal_attention_mask(
            input_shape=input_shape,
            dtype=dtype,
            device=device,
            past_key_values_length=past_key_values_length,
        )

    if not hasattr(modeling_bloom, "_expand_mask"):
        modeling_bloom._expand_mask = bloom_expand_mask
    if not hasattr(modeling_bloom, "_make_causal_mask"):
        modeling_bloom._make_causal_mask = bloom_make_causal_mask
    if not hasattr(modeling_opt, "_expand_mask"):
        modeling_opt._expand_mask = opt_expand_mask
    if not hasattr(modeling_opt, "_make_causal_mask"):
        modeling_opt._make_causal_mask = opt_make_causal_mask
    if not hasattr(modeling_clip, "_expand_mask"):
        modeling_clip._expand_mask = _prepare_4d_attention_mask

    original_auto_config_register = AutoConfig.register

    def compatible_auto_config_register(model_type, config, exist_ok=False):
        return original_auto_config_register(
            model_type,
            config,
            exist_ok=True if model_type == "llava" else exist_ok,
        )

    AutoConfig.register = compatible_auto_config_register
    try:
        from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from videollava.conversation import SeparatorStyle, conv_templates
        from videollava.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            tokenizer_image_token,
        )
        from videollava.model.builder import load_pretrained_model
        from videollava.utils import disable_torch_init
    finally:
        AutoConfig.register = original_auto_config_register

    accelerator = Accelerator()
    samples = load_test_samples(args.test_data_path, args.video_root, args.max_samples)
    local_output_path = prepare_new_output(args.output_path, accelerator.process_index)
    disable_torch_init()
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path=str(args.model_path),
        model_base=None,
        model_name=get_model_name_from_path(str(args.model_path)),
        load_8bit=False,
        load_4bit=False,
        device=f"cuda:{accelerator.local_process_index}",
        cache_dir=str(args.code_path / "cache_dir"),
    )

    original_model_forward = model.forward

    def compatible_model_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        return_dict=None,
        cache_position=None,
    ):
        del cache_position
        return original_model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict,
        )

    model.forward = compatible_model_forward
    model.eval()
    video_processor = processor["video"]
    model_num_frames = model.get_video_tower().config.num_frames
    if model_num_frames != NATIVE_NUM_FRAMES:
        raise ValueError(
            f"Unexpected Video-LLaVA native frame count: {model_num_frames}"
        )

    indexed_samples = list(enumerate(samples))
    local_results = []
    with accelerator.split_between_processes(indexed_samples) as local_samples:
        progress = tqdm(
            local_samples,
            desc=f"Video-LLaVA rank {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        )
        for sample_index, sample in progress:
            video_path = sample_video_path(sample, args.video_root)
            try:
                raw_tensor = video_processor(
                    str(video_path),
                    return_tensors="pt",
                )["pixel_values"]
                if isinstance(raw_tensor, list):
                    tensor = [
                        value.to(model.device, dtype=torch.float16)
                        for value in raw_tensor
                    ]
                else:
                    tensor = raw_tensor.to(model.device, dtype=torch.float16)

                frame_tokens = " ".join(
                    [DEFAULT_IMAGE_TOKEN] * model_num_frames
                )
                conversation = conv_templates["llava_v1"].copy()
                conversation.append_message(
                    conversation.roles[0],
                    frame_tokens
                    + "\n"
                    + append_full_oav_format_requirement(extract_prompt(sample)),
                )
                conversation.append_message(conversation.roles[1], None)
                input_ids = tokenizer_image_token(
                    conversation.get_prompt(),
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(model.device)
                stop_string = (
                    conversation.sep
                    if conversation.sep_style != SeparatorStyle.TWO
                    else conversation.sep2
                )
                stopping_criteria = KeywordsStoppingCriteria(
                    [stop_string],
                    tokenizer,
                    input_ids,
                )
                input_length = input_ids.shape[1]
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=tensor,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                    )
                generated_ids = (
                    output_ids[0, input_length:]
                    if output_ids.shape[1] > input_length
                    else output_ids[0]
                )
                response = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()
                if response.endswith(stop_string):
                    response = response[: -len(stop_string)].strip()
                result = make_result(sample_index, sample, model_output=response)
                del tensor, raw_tensor, input_ids, output_ids, generated_ids
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
