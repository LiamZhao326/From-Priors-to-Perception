import torch
import json
import os
import itertools
from collections import defaultdict
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, get_cosine_schedule_with_warmup
import bitsandbytes.optim as bnb_optim
from accelerate import Accelerator
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
from accelerate.utils import set_seed
set_seed(42)

_DEBUG_PRINTED = False
# ======= 1. 路径与基础配置 =======
VIDEO_ROOT = "path/to/your/videos"
MODEL_PATH = "path/to/your/model"
DATA_PATH = "path/to/your/train_dataset.jsonl"
OUTPUT_DIR = "./lora_results"

def probe_gradients_and_weights(model, loss, step):
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n[Step {step}] 💥 致命崩溃：Loss 当前值为 {loss.item()}")
        
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_max = param.grad.abs().max()
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"[Step {step}] 💥 梯度已损坏 (NaN/Inf) -> 案发层: {name}")
            elif grad_max > 20.0: 
                print(f"[Step {step}] ⚠️ 梯度极度膨胀 (Max={grad_max:.2f}) -> 案发层: {name}")
                
def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total param: {total_params:,}")
    print(f"trainable param: {trainable_params:,}")
    print(f"trainable ratio: {trainable_params / total_params * 100:.4f}%")
    return trainable_params, total_params

class PairedVideoLLaMA3Dataset(Dataset):
    def __init__(self, data_path):
        self.pairs = []

        with open(data_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                sample1 = json.loads(lines[i])
                sample2 = json.loads(lines[i+1])
                self.pairs.append([sample1, sample2])
            else:
                print(f"⚠️ 警告：文件末尾发现落单的单行数据（第 {i} 行），已自动舍弃。")
                
        print(f"\n📦 大道至简！按行相邻读取完成，共打包了 {len(self.pairs)} 组正负样本对！\n")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def forward_pass(sample, model, processor, accelerator):
    global _DEBUG_PRINTED
    video_rel_path = sample['video'][0]
    video_path = os.path.join(VIDEO_ROOT, video_rel_path)
    
    user_content = []
    assistant_text = ""
    for turn in sample["conversations"]:
        if turn["from"] == "human":
            user_content = [
                {"type": "video", "video": {"video_path": video_path, "nframes": 16}},
                {"type": "text", "text": turn["value"].replace("<video>\n", "")}
            ]
        else:
            assistant_text = turn["value"]

    conversation = [{"role": "user", "content": user_content}]
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)

    input_ids = inputs['input_ids']
    prompt_len = input_ids.shape[1]

    answer_text = assistant_text + "<|im_end|>"
    answer_tokens = processor.tokenizer(
        answer_text, 
        return_tensors='pt', 
        add_special_tokens=False
    ).input_ids.to(accelerator.device)

    labels = torch.full((1, prompt_len + answer_tokens.shape[1]), -100, dtype=torch.long, device=accelerator.device)
    labels[:, prompt_len:] = answer_tokens

    inputs['input_ids'] = torch.cat([input_ids, answer_tokens], dim=1)
    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(answer_tokens)], dim=1)
    inputs['labels'] = labels
    
    outputs = model(**inputs)

    if _DEBUG_PRINTED and accelerator.is_main_process:
        import re
        import torch.nn.functional as F
        
        print("\n" + "="*80)
        print("🚀 [硬核解剖] 首次前向传播底层数据核对 (全量打印)")
        print("="*80)
        
        decoded_inputs = processor.tokenizer.decode(inputs['input_ids'][0])
        compressed_inputs = re.sub(r'(<image>){3,}', lambda m: f"\n[... 这里省略了 {len(m.group(0))//7} 个 <image> Token ...]\n", decoded_inputs)
        
        print(f"📐 input_ids 长度 (输入): {inputs['input_ids'].shape[1]}")
        print(f"📐 logits 长度 (输出):    {outputs.logits.shape[1]}")
        print("\n📝 [Decode] 完整的 input_ids (已折叠图像 Token):")
        print(compressed_inputs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['labels'][..., 1:].contiguous()

        offset = shift_labels.shape[1] - shift_logits.shape[1]
        print(f"\n📉 模型内部视觉融合，序列被压缩了 {offset} 个 Token。")

        valid_indices = (shift_labels[0] != -100).nonzero(as_tuple=True)[0]
        
        print(f"🔍 总共有 {len(valid_indices)} 个 Token 参与 Loss 计算。")
        print("\n📊 [逐 Token Loss 深度剖析] (全量打印):")
        print("-" * 80)
        
        for i, idx in enumerate(valid_indices):
            true_label_id = shift_labels[0][idx].item()
            true_token = processor.tokenizer.decode([true_label_id]).replace('\n', '\\n')
            
            idx_in_logits = idx - offset
            token_logits = shift_logits[0][idx_in_logits]
            
            probs = F.softmax(token_logits, dim=-1)
            true_prob = probs[true_label_id].item()
            token_loss = -torch.log(torch.tensor(true_prob + 1e-10)).item() 
            
            top_3_logits, top_3_indices = torch.topk(token_logits, 3)
            top_3_tokens = [processor.tokenizer.decode([t_idx.item()]).replace('\n', '\\n') for t_idx in top_3_indices]
            top_3_probs = [probs[t_idx].item() for t_idx in top_3_indices]
            
            print(f"🔹 Step {i+1:<2} | 目标词: '{true_token:<10}' (ID:{true_label_id})")
            print(f"   ➤ 该词 Loss : {token_loss:.4f} | 命中概率: {true_prob:.8f}")
            print(f"   ➤ Top-3 瞎猜: 1. '{top_3_tokens[0]}' ({top_3_probs[0]:.2%}) | 2. '{top_3_tokens[1]}' ({top_3_probs[1]:.2%}) | 3. '{top_3_tokens[2]}' ({top_3_probs[2]:.2%})")
            print("-" * 60)
            
        print(f"\n📉 模型整体平均 Loss (官方结算): {outputs.loss.item():.4f}")
        print("="*80 + "\n")
        _DEBUG_PRINTED = False
        
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs['labels'][..., 1:].contiguous()

    offset = shift_labels.shape[1] - shift_logits.shape[1]
    valid_shift_labels = shift_labels[:, offset:]
    
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = valid_shift_labels.view(-1)

    import torch.nn.functional as F
    real_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

    return real_loss

def train(model, dataloader, optimizer, processor, scheduler, accelerator, lora_save_dir, num_epochs=3, accum_steps=2, resume_batch_idx=0, resume_step_idx=0, resume_epoch_idx=0):
    model.train()
    num_gradient_updates = resume_step_idx
    save_every_n_steps = 10
    
    for epoch in range(resume_epoch_idx, num_epochs):
        total_loss = 0.0
        accum_step = 0
        accelerator.print(f"Starting epoch {epoch+1}/{num_epochs}")

        if epoch == resume_epoch_idx and resume_batch_idx > 0:
            batches_to_skip = resume_batch_idx
            accelerator.print(f"Resuming epoch {epoch+1} by skipping first {batches_to_skip} pairs.")
            effective_dataloader = itertools.islice(dataloader, batches_to_skip, None)
        else:
            effective_dataloader = dataloader
            batches_to_skip = 0

        with tqdm(desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process, total=len(dataloader), initial=batches_to_skip) as pbar:
            for pair_idx, batch in enumerate(effective_dataloader):
                samples_pair = batch[0]
                
                for sample in samples_pair:
                    loss = forward_pass(sample, model, processor, accelerator)
                    loss = loss / accum_steps
                    
                    accelerator.backward(loss)
                    total_loss += loss.item() * accum_steps
                    accum_step += 1

                    if accum_step % accum_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        if not accelerator.optimizer_step_was_skipped:
                            scheduler.step()
                        optimizer.zero_grad()
                        
                        accum_step = 0
                        num_gradient_updates += 1

                        # if (num_gradient_updates > 0) and (num_gradient_updates % save_every_n_steps == 0):
                        #     save_dir = os.path.join(lora_save_dir, f"checkpoint_step_{num_gradient_updates}")
                        #     accelerator.wait_for_everyone()
                        #     try:
                        #         accelerator.save_state(output_dir=save_dir)
                        #         if accelerator.is_main_process:
                        #             unwrapped_model = accelerator.unwrap_model(model)
                        #             unwrapped_model.save_pretrained(save_dir)
                        #             processor.save_pretrained(save_dir)
                        #             accelerator.print(f"Checkpoint saved to {save_dir}")
                        #     except Exception as e:
                        #         accelerator.print(f"Error saving checkpoint: {e}")
                
                current_lr = scheduler.get_last_lr()[0]
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}", lr=f"{current_lr:.2e}")
        
        total_samples = (len(dataloader) - batches_to_skip) * 2
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accelerator.print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

if __name__ == "__main__":
    gradient_accumulation_steps = 4
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    LOAD_FROM_CHECKPOINT = None  
    RESUME_GRADIENT_UPDATES = 0  
    RESUME_EPOCH = 0             
    RESUME_BATCH_IDX = 0         

    accelerator.print("⏳ 加载 Processor ...")
    if LOAD_FROM_CHECKPOINT:
        processor = AutoProcessor.from_pretrained(LOAD_FROM_CHECKPOINT, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    accelerator.print("⏳ 配置 4-bit 量化与加载基座模型 ...")
    with accelerator.main_process_first():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=[
                "vision_encoder",   
                "mm_projector"      
            ]
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            quantization_config=quantization_config,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        if LOAD_FROM_CHECKPOINT:
            accelerator.print(f"⏳ 从 Checkpoint 恢复 LoRA 权重: {LOAD_FROM_CHECKPOINT}")
            model = PeftModel.from_pretrained(model, LOAD_FROM_CHECKPOINT, is_trainable=True)
        else:
            accelerator.print("⏳ 初始化全新 LoRA 参数 ...")
            target_modules = []
            num_decoder_layers = 28
            for i in range(num_decoder_layers):
                target_modules.extend([
                    f"model.layers.{i}.self_attn.q_proj", f"model.layers.{i}.self_attn.k_proj",
                    f"model.layers.{i}.self_attn.v_proj", f"model.layers.{i}.self_attn.o_proj",
                    f"model.layers.{i}.mlp.gate_proj", f"model.layers.{i}.mlp.up_proj", f"model.layers.{i}.mlp.down_proj"
                ])
            target_modules.extend(["model.mm_projector.readout.0", "model.mm_projector.readout.2"])
            
            lora_config = LoraConfig(
                r=16, lora_alpha=32, 
                target_modules=target_modules, 
                lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        count_trainable_parameters(model)

    dataset = PairedVideoLLaMA3Dataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

    optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=5e-5)
    
    num_epochs = 3
    num_update_steps_per_epoch = len(dataloader) 
    num_training_steps = num_epochs * num_update_steps_per_epoch
    
    num_warmup_steps = int(num_training_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    if LOAD_FROM_CHECKPOINT and RESUME_GRADIENT_UPDATES > 0:
        accelerator.print(f"⏳ 手工步进 Scheduler 恢复至第 {RESUME_GRADIENT_UPDATES} 步...")
        for _ in range(RESUME_GRADIENT_UPDATES):
            scheduler.step()

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if LOAD_FROM_CHECKPOINT:
        accelerator.load_state(LOAD_FROM_CHECKPOINT)
        accelerator.print("✅ Optimizer 和 RNG 状态已恢复")

    accelerator.print("🚀 启动加速器训练循环 ...")
    train(model, dataloader, optimizer, processor, scheduler, accelerator, OUTPUT_DIR, num_epochs=num_epochs, accum_steps=gradient_accumulation_steps, resume_batch_idx=RESUME_BATCH_IDX, resume_epoch_idx=RESUME_EPOCH, resume_step_idx=RESUME_GRADIENT_UPDATES)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OUTPUT_DIR = os.path.join(OUTPUT_DIR, "final_version")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        accelerator.print(f"💾 最终 LoRA 权重已保存至 {OUTPUT_DIR}")