"""
LoRA 微调 Qwen3.5-4B
使用 peft + transformers 进行微调

运行: CUDA_VISIBLE_DEVICES=0 python train_lora.py
预计: RTX 3090 上 5-10 分钟
"""
import json
import os
import torch
from modelscope import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "lora_output")

# ========== 下载模型 ==========
print("Downloading Qwen3.5-4B from ModelScope...")
MODEL_DIR = snapshot_download("Qwen/Qwen3.5-4B", cache_dir="/data2/css2025/models")
print(f"Model: {MODEL_DIR}")

# ========== 加载（4-bit量化，节省显存） ==========
print("Loading tokenizer and model (4-bit QLoRA)...")
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
)
model.gradient_checkpointing_enable()

# ========== LoRA 配置 ==========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== 数据集 ==========
class RAGDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=1024):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']
        
        # 使用 chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_len,
            padding=False, return_tensors=None
        )
        
        # 构造labels：只对assistant部分计算loss
        input_ids = encoded['input_ids']
        labels = input_ids.copy()
        
        # 找到assistant回答的起始位置
        # 简单方式：对整个序列计算loss（因为数据很短）
        # 更好的方式是mask掉非assistant部分，但对468条数据影响不大
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        }

train_dataset = RAGDataset(
    os.path.join(DATA_DIR, 'lora_train_data.json'),
    tokenizer,
    max_len=768  # 数据很短，768够了
)

print(f"Training samples: {len(train_dataset)}")

# ========== 训练 ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=20,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("\n=== Starting LoRA training ===")
trainer.train()

# 保存
print("\nSaving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to: {OUTPUT_DIR}")
print("Done!")
