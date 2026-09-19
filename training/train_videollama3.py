import argparse
import json
import math
import os
import random
from pathlib import Path

import bitsandbytes.optim as bnb_optim
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

IGNORE_INDEX = -100
ASSISTANT_END_TOKEN = "<|im_end|>"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Square-root-balanced pair-wise QLoRA fine-tuning of "
            "VideoLLaMA3-7B on PriorPair."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument(
        "--scheduler-epochs",
        type=int,
        default=3,
        help="Epoch horizon used to parameterize the learning-rate schedule.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Restrict the dataset to the first N pairs for a smoke run.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate all pairs and preprocess the first pair without "
            "loading the model or writing weights."
        ),
    )
    parser.add_argument(
        "--balance-audit-only",
        action="store_true",
        help=(
            "Validate square-root sampling across all epochs without "
            "loading the processor or model."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not args.video_root.is_dir():
        raise FileNotFoundError(f"Video root not found: {args.video_root}")
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Training JSONL not found: {args.data_path}")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be positive")
    if args.scheduler_epochs < args.num_epochs:
        raise ValueError("--scheduler-epochs must be at least --num-epochs")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.gradient_accumulation_steps % 2 != 0:
        raise ValueError(
            "--gradient-accumulation-steps must be even so a positive/"
            "negative pair can never be split across optimizer updates"
        )
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")
    if args.lora_r <= 0:
        raise ValueError("--lora-r must be positive")
    if args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive when provided")


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


def extract_answer(sample):
    for turn in sample.get("conversations", []):
        if turn.get("from") == "gpt":
            answer = turn.get("value", "")
            if answer:
                return answer
    raise ValueError("Sample has no non-empty GPT answer")


def is_negative_video_path(relative_path):
    category = Path(relative_path).parts[0]
    return category.endswith("_negative") or category.endswith(
        "_negative_aligned"
    )


def validate_pair(pair, pair_index, video_root):
    if len(pair) != 2:
        raise ValueError(f"Pair {pair_index} does not contain two samples")

    positive, negative = pair
    positive_relative = positive.get("video", [None])[0]
    negative_relative = negative.get("video", [None])[0]
    if not positive_relative or not negative_relative:
        raise ValueError(f"Pair {pair_index} has a missing video path")
    if is_negative_video_path(positive_relative):
        raise ValueError(
            f"Pair {pair_index} first sample is not positive: "
            f"{positive_relative}"
        )
    if not is_negative_video_path(negative_relative):
        raise ValueError(
            f"Pair {pair_index} second sample is not negative: "
            f"{negative_relative}"
        )

    positive_category = Path(positive_relative).parts[0]
    negative_category = Path(negative_relative).parts[0]
    expected_negative_categories = {
        f"{positive_category}_negative",
        f"{positive_category}_negative_aligned",
    }
    if negative_category not in expected_negative_categories:
        raise ValueError(
            f"Pair {pair_index} category mismatch: "
            f"{positive_category} vs. {negative_category}"
        )

    positive_name = Path(positive_relative).name
    negative_name = Path(negative_relative).name
    expected_negative_names = {
        positive_name,
        f"{Path(positive_name).stem}_rev.mp4",
    }
    if negative_name not in expected_negative_names:
        raise ValueError(
            f"Pair {pair_index} filename mismatch: "
            f"{positive_name} vs. {negative_name}"
        )

    for role, sample in (("positive", positive), ("negative", negative)):
        relative_path = sample["video"][0]
        video_path = video_root / relative_path
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Pair {pair_index} {role} video not found: {video_path}"
            )
        extract_prompt(sample)
        extract_answer(sample)


class PairedVideoLLaMA3Dataset(Dataset):
    """
    The atomic dataset item is one ordered [positive, negative] pair.

    Every original pair is retained once per epoch. Categories smaller than
    the largest category are upsampled to ceil(sqrt(n_c * n_max)). Extra
    occurrences are selected in shuffled cycles so exposure is distributed
    as evenly as possible while changing reproducibly between epochs.
    """

    def __init__(
        self,
        data_path,
        video_root,
        max_pairs=None,
        sampling_seed=42,
    ):
        samples = []
        with data_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {data_path}:{line_number}"
                    ) from error

        if len(samples) % 2 != 0:
            raise ValueError(
                f"Training JSONL contains an odd number of samples: "
                f"{len(samples)}"
            )

        pairs = [
            [samples[index], samples[index + 1]]
            for index in range(0, len(samples), 2)
        ]
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        if not pairs:
            raise ValueError("No training pairs were loaded")

        for pair_index, pair in enumerate(pairs):
            validate_pair(pair, pair_index, video_root)

        self.original_pairs = pairs
        self.sampling_seed = sampling_seed
        self.category_pairs = {}
        for pair in self.original_pairs:
            category = Path(pair[0]["video"][0]).parts[0]
            self.category_pairs.setdefault(category, []).append(pair)

        self.category_original_counts = {
            category: len(category_pairs)
            for category, category_pairs in sorted(
                self.category_pairs.items()
            )
        }
        self.max_category_count = max(
            self.category_original_counts.values()
        )
        self.category_target_counts = {
            category: math.ceil(
                math.sqrt(count * self.max_category_count)
            )
            for category, count in self.category_original_counts.items()
        }
        self.balanced_pair_count = sum(
            self.category_target_counts.values()
        )
        self.current_epoch = None
        self.pairs = []
        self.set_epoch(0)

    @property
    def sampling_metadata(self):
        return {
            "sampling_strategy": "sqrt_upsample_to_largest_category",
            "sampling_seed": self.sampling_seed,
            "original_pair_count": len(self.original_pairs),
            "balanced_pairs_per_epoch": self.balanced_pair_count,
            "max_category_count": self.max_category_count,
            "category_original_counts": self.category_original_counts,
            "category_target_counts": self.category_target_counts,
        }

    def set_epoch(self, epoch_index):
        if epoch_index < 0:
            raise ValueError("epoch_index must be non-negative")

        rng = random.Random(
            self.sampling_seed + epoch_index * 1_000_003
        )
        balanced_pairs = []
        for category in sorted(self.category_pairs):
            original_category_pairs = self.category_pairs[category]
            balanced_pairs.extend(original_category_pairs)

            remaining = (
                self.category_target_counts[category]
                - len(original_category_pairs)
            )
            while remaining > 0:
                shuffled_cycle = list(original_category_pairs)
                rng.shuffle(shuffled_cycle)
                take_count = min(remaining, len(shuffled_cycle))
                balanced_pairs.extend(shuffled_cycle[:take_count])
                remaining -= take_count

        if len(balanced_pairs) != self.balanced_pair_count:
            raise RuntimeError(
                "Balanced pair count mismatch: "
                f"{len(balanced_pairs)} vs. "
                f"{self.balanced_pair_count}"
            )

        realized_counts = {}
        for pair in balanced_pairs:
            category = Path(pair[0]["video"][0]).parts[0]
            realized_counts[category] = (
                realized_counts.get(category, 0) + 1
            )
        if realized_counts != self.category_target_counts:
            raise RuntimeError(
                "Balanced category counts do not match their targets: "
                f"{realized_counts} vs. "
                f"{self.category_target_counts}"
            )

        self.pairs = balanced_pairs
        self.current_epoch = epoch_index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def collate_one_pair(batch):
    if len(batch) != 1:
        raise ValueError(f"Expected one pair per batch, received {len(batch)}")
    return batch[0]


def pair_identity(pair):
    return tuple(sample["video"][0] for sample in pair)


def count_pair_identities(pairs):
    counts = {}
    for pair in pairs:
        identity = pair_identity(pair)
        counts[identity] = counts.get(identity, 0) + 1
    return counts


def audit_pairwise_accumulation(pair_count, accumulation_steps):
    """
    Verify that every optimizer window contains complete positive/negative
    pairs. The final partial window is allowed only when it also ends at a
    pair boundary.
    """
    stream = [
        (pair_index, role)
        for pair_index in range(pair_count)
        for role in ("positive", "negative")
    ]
    for window_start in range(0, len(stream), accumulation_steps):
        window = stream[window_start : window_start + accumulation_steps]
        pair_roles = {}
        for pair_index, role in window:
            pair_roles.setdefault(pair_index, set()).add(role)
        split_pairs = {
            pair_index: roles
            for pair_index, roles in pair_roles.items()
            if roles != {"positive", "negative"}
        }
        if split_pairs:
            raise RuntimeError(
                "Pair-wise accumulation audit failed at micro-batch window "
                f"{window_start}: {split_pairs}"
            )


def prepare_sample_inputs(
    sample,
    processor,
    video_root,
    num_frames,
    device,
):
    video_relative_path = sample["video"][0]
    video_path = video_root / video_relative_path
    prompt = extract_prompt(sample)
    answer = extract_answer(sample)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": str(video_path),
                        "nframes": num_frames,
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

    prompt_length = inputs["input_ids"].shape[1]
    answer_tokens = processor.tokenizer(
        answer + ASSISTANT_END_TOKEN,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    answer_length = answer_tokens.shape[1]

    labels = torch.full(
        (1, prompt_length + answer_length),
        IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )
    labels[:, prompt_length:] = answer_tokens
    inputs["input_ids"] = torch.cat(
        [inputs["input_ids"], answer_tokens],
        dim=1,
    )
    inputs["attention_mask"] = torch.cat(
        [
            inputs["attention_mask"],
            torch.ones_like(answer_tokens),
        ],
        dim=1,
    )
    inputs["labels"] = labels

    if "position_ids" in inputs:
        position_ids = inputs["position_ids"]
        answer_position_ids = torch.arange(
            int(position_ids[0, -1].item()) + 1,
            int(position_ids[0, -1].item()) + 1 + answer_length,
            device=device,
            dtype=position_ids.dtype,
        ).unsqueeze(0)
        inputs["position_ids"] = torch.cat(
            [position_ids, answer_position_ids],
            dim=1,
        )

    supervised_tokens = int((labels != IGNORE_INDEX).sum().item())
    if supervised_tokens == 0:
        raise ValueError(
            f"No assistant tokens are supervised: {video_relative_path}"
        )
    return inputs


def calculate_compression_aligned_loss(outputs, uncompressed_labels):
    """
    VideoLLaMA3 compresses visual tokens inside forward(). All supervised
    answer tokens occur after the visual prompt, so removing the compression
    offset from the start of the shifted labels aligns them with the returned
    logits. This preserves the loss formulation used for PriorPair
    VideoLLaMA3 training script.
    """
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = uncompressed_labels[..., 1:].contiguous()
    compression_offset = shift_labels.shape[1] - shift_logits.shape[1]
    if compression_offset < 0:
        raise RuntimeError(
            "VideoLLaMA3 returned more logits than uncompressed labels: "
            f"{shift_logits.shape[1]} vs. {shift_labels.shape[1]}"
        )

    aligned_labels = shift_labels[:, compression_offset:]
    if aligned_labels.shape[1] != shift_logits.shape[1]:
        raise RuntimeError(
            "Compressed logits and aligned labels have different lengths: "
            f"{shift_logits.shape[1]} vs. {aligned_labels.shape[1]}"
        )
    if int((aligned_labels != IGNORE_INDEX).sum().item()) == 0:
        raise RuntimeError("Compression removed all supervised answer tokens")

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        aligned_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def forward_sample(
    sample,
    model,
    processor,
    video_root,
    num_frames,
    device,
):
    inputs = prepare_sample_inputs(
        sample,
        processor,
        video_root,
        num_frames,
        device,
    )
    outputs = model(**inputs)
    loss = calculate_compression_aligned_loss(
        outputs,
        inputs["labels"],
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss for {sample['video'][0]}: {loss.item()}"
        )
    # Return only the
    # externally computed compression-aligned loss.  In particular, do not
    # keep the full model output alive until backward, because it also owns
    # the unused internal outputs.loss graph.
    return loss


def build_lora_target_modules(num_hidden_layers):
    target_modules = []
    for layer_index in range(num_hidden_layers):
        layer_prefix = f"model.layers.{layer_index}"
        target_modules.extend(
            [
                f"{layer_prefix}.self_attn.q_proj",
                f"{layer_prefix}.self_attn.k_proj",
                f"{layer_prefix}.self_attn.v_proj",
                f"{layer_prefix}.self_attn.o_proj",
                f"{layer_prefix}.mlp.gate_proj",
                f"{layer_prefix}.mlp.up_proj",
                f"{layer_prefix}.mlp.down_proj",
            ]
        )
    target_modules.extend(
        [
            "model.mm_projector.readout.0",
            "model.mm_projector.readout.2",
        ]
    )
    return target_modules


def validate_lora_target_modules(model, target_modules):
    module_names = {name for name, _ in model.named_modules()}
    missing = [
        module_name
        for module_name in target_modules
        if module_name not in module_names
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} LoRA target modules do not exist; "
            f"first entries: {missing[:5]}"
        )


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def verify_trainable_parameters(model):
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("The model has no trainable parameters")

    non_lora = [
        name for name in trainable_names if "lora_" not in name
    ]
    if non_lora:
        raise RuntimeError(
            "Unexpected non-LoRA trainable parameters: "
            f"{non_lora[:5]}"
        )
    vision_trainable = [
        name for name in trainable_names if "vision_encoder" in name
    ]
    if vision_trainable:
        raise RuntimeError(
            "The vision encoder is not fully frozen: "
            f"{vision_trainable[:5]}"
        )
    if not any("mm_projector" in name for name in trainable_names):
        raise RuntimeError("No multimodal-projector LoRA is trainable")
    if not any(".layers." in name for name in trainable_names):
        raise RuntimeError("No language-model LoRA is trainable")
    return trainable_names


def optimizer_update(
    model,
    optimizer,
    scheduler,
    accelerator,
    max_grad_norm,
):
    accelerator.clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
    )
    optimizer.step()
    step_was_skipped = accelerator.optimizer_step_was_skipped
    if not step_was_skipped:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return not step_was_skipped


def save_adapter(model, processor, output_dir, accelerator):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            safe_serialization=True,
        )
        processor.save_pretrained(output_dir)
        accelerator.print(f"Saved LoRA adapter to {output_dir}")
    accelerator.wait_for_everyone()


def run_preflight(args):
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = (
            processor.tokenizer.eos_token_id
        )
    dataset = PairedVideoLLaMA3Dataset(
        args.data_path,
        args.video_root,
        max_pairs=args.max_pairs,
        sampling_seed=args.seed,
    )
    audit_pairwise_accumulation(
        len(dataset),
        args.gradient_accumulation_steps,
    )
    print(
        f"Validated {len(dataset)} complete pair(s); "
        f"gradient_accumulation_steps="
        f"{args.gradient_accumulation_steps}; pair_split=0"
    )
    print(
        "Square-root balance: "
        f"original_pairs={len(dataset.original_pairs)}, "
        f"balanced_pairs_per_epoch={len(dataset)}, "
        f"category_original_counts="
        f"{dataset.category_original_counts}, "
        f"category_target_counts={dataset.category_target_counts}"
    )

    for role, sample in zip(("positive", "negative"), dataset[0]):
        inputs = prepare_sample_inputs(
            sample,
            processor,
            args.video_root,
            args.num_frames,
            torch.device("cpu"),
        )
        supervised_tokens = int(
            (inputs["labels"] != IGNORE_INDEX).sum().item()
        )
        print(
            f"First-pair {role}: video={sample['video'][0]}, "
            f"input_tokens={inputs['input_ids'].shape[1]}, "
            f"supervised_tokens={supervised_tokens}, "
            f"pixel_values={tuple(inputs['pixel_values'].shape)}"
        )


def run_balance_audit(args):
    dataset = PairedVideoLLaMA3Dataset(
        args.data_path,
        args.video_root,
        max_pairs=args.max_pairs,
        sampling_seed=args.seed,
    )
    original_identity_counts = count_pair_identities(
        dataset.original_pairs
    )
    epoch_repeat_signatures = []

    for epoch_index in range(args.num_epochs):
        dataset.set_epoch(epoch_index)
        audit_pairwise_accumulation(
            len(dataset),
            args.gradient_accumulation_steps,
        )
        epoch_identity_counts = count_pair_identities(dataset.pairs)
        missing_originals = [
            identity
            for identity, original_count
            in original_identity_counts.items()
            if epoch_identity_counts.get(identity, 0) < original_count
        ]
        if missing_originals:
            raise RuntimeError(
                f"Epoch {epoch_index + 1} omitted original pairs: "
                f"{missing_originals[:5]}"
            )

        repeat_signature = tuple(
            sorted(
                (
                    identity,
                    count - original_identity_counts.get(identity, 0),
                )
                for identity, count in epoch_identity_counts.items()
                if count > original_identity_counts.get(identity, 0)
            )
        )
        epoch_repeat_signatures.append(repeat_signature)

        first_pass = tuple(
            pair_identity(pair) for pair in dataset.pairs
        )
        dataset.set_epoch(epoch_index)
        second_pass = tuple(
            pair_identity(pair) for pair in dataset.pairs
        )
        if first_pass != second_pass:
            raise RuntimeError(
                f"Epoch {epoch_index + 1} sampling is not reproducible"
            )

        print(
            f"Epoch {epoch_index + 1}: "
            f"balanced_pairs={len(dataset)}, "
            f"all_original_pairs_present=yes, pair_split=0"
        )

    if args.num_epochs > 1 and len(set(epoch_repeat_signatures)) == 1:
        raise RuntimeError(
            "Repeated-pair selections did not change across epochs"
        )

    four_gpu_pairs_per_rank = math.ceil(len(dataset) / 4)
    four_gpu_updates_per_epoch = math.ceil(
        four_gpu_pairs_per_rank
        * 2
        / args.gradient_accumulation_steps
    )
    print(
        "Square-root balance audit passed: "
        f"original_pairs={len(dataset.original_pairs)}, "
        f"balanced_pairs_per_epoch={len(dataset)}, "
        f"category_original_counts="
        f"{dataset.category_original_counts}, "
        f"category_target_counts={dataset.category_target_counts}, "
        f"four_gpu_updates_per_epoch={four_gpu_updates_per_epoch}, "
        f"four_gpu_total_updates="
        f"{four_gpu_updates_per_epoch * args.num_epochs}"
    )


def train(args):
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VideoLLaMA3 training")
    torch.cuda.set_device(accelerator.local_process_index)

    final_checkpoint = args.output_dir / f"checkpoint_epoch_{args.num_epochs}"
    if final_checkpoint.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing final adapter: "
            f"{final_checkpoint}"
        )

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = (
            processor.tokenizer.eos_token_id
        )

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset = PairedVideoLLaMA3Dataset(
        args.data_path,
        args.video_root,
        max_pairs=args.max_pairs,
        sampling_seed=args.seed,
    )
    audit_pairwise_accumulation(
        len(dataset),
        args.gradient_accumulation_steps,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_one_pair,
        num_workers=0,
        pin_memory=True,
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
        f"Loading VideoLLaMA3-7B on {accelerator.device} in 4-bit NF4"
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
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.get_vision_encoder().requires_grad_(False)

    target_modules = build_lora_target_modules(
        config.num_hidden_layers
    )
    validate_lora_target_modules(model, target_modules)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable_names = verify_trainable_parameters(model)
    total_parameters, trainable_parameters = count_parameters(model)
    accelerator.print(
        f"Parameters: total={total_parameters:,}, "
        f"trainable={trainable_parameters:,} "
        f"({100 * trainable_parameters / total_parameters:.4f}%)"
    )
    accelerator.print(
        f"Trainable tensor groups={len(trainable_names)}; "
        f"LoRA target modules={len(target_modules)}"
    )

    optimizer = bnb_optim.AdamW8bit(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
    )

    pairs_per_rank = math.ceil(
        len(dataset) / accelerator.num_processes
    )
    micro_batches_per_rank = pairs_per_rank * 2
    updates_per_epoch = math.ceil(
        micro_batches_per_rank
        / args.gradient_accumulation_steps
    )
    actual_training_steps = updates_per_epoch * args.num_epochs
    total_scheduler_steps = updates_per_epoch * args.scheduler_epochs
    warmup_steps = int(total_scheduler_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_scheduler_steps,
    )

    # Keep the scheduler outside accelerator.prepare(). AcceleratedScheduler
    # would advance it once per process when split_batches=False, while the
    # step count above is already expressed in per-rank optimizer updates.
    model, optimizer, dataloader = accelerator.prepare(
        model,
        optimizer,
        dataloader,
    )
    if len(dataloader) != pairs_per_rank:
        raise RuntimeError(
            "Unexpected prepared dataloader length: "
            f"{len(dataloader)} vs. expected {pairs_per_rank}"
        )
    optimizer.zero_grad(set_to_none=True)

    accelerator.print(
        "Square-root balance: "
        f"original_pairs={len(dataset.original_pairs)}, "
        f"balanced_pairs_per_epoch={len(dataset)}, "
        f"category_original_counts="
        f"{dataset.category_original_counts}, "
        f"category_target_counts={dataset.category_target_counts}"
    )
    accelerator.print(
        f"Pair schedule: global_pairs={len(dataset)}, "
        f"pairs_per_rank={pairs_per_rank}, "
        f"micro_batches_per_rank={micro_batches_per_rank}, "
        f"updates_per_epoch={updates_per_epoch}, "
        f"training_epochs={args.num_epochs}, "
        f"scheduler_epochs={args.scheduler_epochs}, "
        f"actual_training_steps={actual_training_steps}, "
        f"total_scheduler_steps={total_scheduler_steps}, "
        f"warmup_steps={warmup_steps}"
    )

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "training_config.json").open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                }
                | {
                    "num_processes": accelerator.num_processes,
                    "pairs_per_rank": pairs_per_rank,
                    "updates_per_epoch": updates_per_epoch,
                    "actual_training_steps": actual_training_steps,
                    "total_scheduler_steps": total_scheduler_steps,
                    "warmup_steps": warmup_steps,
                }
                | dataset.sampling_metadata,
                output_file,
                ensure_ascii=False,
                indent=2,
            )
        (args.output_dir / "epoch_loss_metrics.jsonl").write_text(
            "",
            encoding="utf-8",
        )

    global_update_step = 0
    for epoch_index in range(args.num_epochs):
        dataset.set_epoch(epoch_index)
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch_index)
        model.train()
        accumulated_micro_batches = 0
        running_loss = 0.0
        processed_samples = 0
        group_loss_sums = [0.0, 0.0, 0.0, 0.0]
        group_sample_counts = [0, 0, 0, 0]
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch_index + 1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for pair in progress:
            if len(pair) != 2:
                raise RuntimeError("A dataloader item split a PriorPair pair")

            positive_category = Path(pair[0]["video"][0]).parts[0]
            group_offset = 2 if positive_category.startswith("IV_") else 0
            pair_losses = []
            for sample_index, sample in enumerate(pair):
                loss = forward_sample(
                    sample,
                    model,
                    processor,
                    args.video_root,
                    args.num_frames,
                    accelerator.device,
                )
                # Accelerator.backward() already scales the loss by
                # gradient_accumulation_steps.
                accelerator.backward(loss)
                loss_value = loss.detach().float().item()
                accumulated_micro_batches += 1
                processed_samples += 1
                running_loss += loss_value
                group_index = group_offset + sample_index
                group_loss_sums[group_index] += loss_value
                group_sample_counts[group_index] += 1
                pair_losses.append(loss_value)

                if (
                    accumulated_micro_batches
                    == args.gradient_accumulation_steps
                ):
                    if optimizer_update(
                        model,
                        optimizer,
                        scheduler,
                        accelerator,
                        args.max_grad_norm,
                    ):
                        global_update_step += 1
                    accumulated_micro_batches = 0

                del loss

            progress.set_postfix(
                pair_loss=f"{sum(pair_losses) / len(pair_losses):.4f}",
                update=global_update_step,
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        if accumulated_micro_batches:
            # Each rank receives an equal padded number of complete pairs.
            # Correct the final partial-window scale before its optimizer step.
            gradient_scale = (
                args.gradient_accumulation_steps
                / accumulated_micro_batches
            )
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)
            if optimizer_update(
                model,
                optimizer,
                scheduler,
                accelerator,
                args.max_grad_norm,
            ):
                global_update_step += 1

        local_stats = torch.tensor(
            [
                running_loss,
                float(processed_samples),
                group_loss_sums[0],
                float(group_sample_counts[0]),
                group_loss_sums[1],
                float(group_sample_counts[1]),
                group_loss_sums[2],
                float(group_sample_counts[2]),
                group_loss_sums[3],
                float(group_sample_counts[3]),
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )
        global_stats = accelerator.reduce(
            local_stats,
            reduction="sum",
        )
        average_loss = (
            global_stats[0].item() / global_stats[1].item()
            if global_stats[1].item() > 0
            else 0.0
        )
        group_average_losses = [
            (
                global_stats[2 + 2 * group_index].item()
                / global_stats[3 + 2 * group_index].item()
                if global_stats[3 + 2 * group_index].item() > 0
                else 0.0
            )
            for group_index in range(4)
        ]
        group_global_counts = [
            int(global_stats[3 + 2 * group_index].item())
            for group_index in range(4)
        ]
        accelerator.print(
            f"Epoch {epoch_index + 1} complete: "
            f"samples_per_rank={processed_samples}, "
            f"global_average_loss={average_loss:.6f}, "
            f"i_iii_positive_loss={group_average_losses[0]:.6f}, "
            f"i_iii_negative_loss={group_average_losses[1]:.6f}, "
            f"iv_positive_loss={group_average_losses[2]:.6f}, "
            f"iv_negative_loss={group_average_losses[3]:.6f}, "
            f"updates={global_update_step}, "
            f"lr={scheduler.get_last_lr()[0]:.6e}"
        )
        if accelerator.is_main_process:
            epoch_metrics = {
                "epoch": epoch_index + 1,
                "global_average_loss": average_loss,
                "i_iii_positive_loss": group_average_losses[0],
                "i_iii_negative_loss": group_average_losses[1],
                "iv_positive_loss": group_average_losses[2],
                "iv_negative_loss": group_average_losses[3],
                "i_iii_positive_count": group_global_counts[0],
                "i_iii_negative_count": group_global_counts[1],
                "iv_positive_count": group_global_counts[2],
                "iv_negative_count": group_global_counts[3],
                "global_update_step": global_update_step,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            with (
                args.output_dir / "epoch_loss_metrics.jsonl"
            ).open("a", encoding="utf-8") as output_file:
                output_file.write(
                    json.dumps(epoch_metrics, ensure_ascii=False) + "\n"
                )
        save_adapter(
            model,
            processor,
            args.output_dir / f"checkpoint_epoch_{epoch_index + 1}",
            accelerator,
        )

    if global_update_step != actual_training_steps:
        raise RuntimeError(
            "Optimizer/scheduler step mismatch: "
            f"{global_update_step} vs. expected {actual_training_steps}"
        )


def main():
    args = parse_args()
    validate_args(args)
    if args.balance_audit_only:
        run_balance_audit(args)
        return
    if args.preflight_only:
        run_preflight(args)
        return
    train(args)


if __name__ == "__main__":
    main()
