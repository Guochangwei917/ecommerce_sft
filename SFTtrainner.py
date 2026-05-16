import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    #BitsAndBytesConfig,          # 做LoRA vs QLoRA 消融实验时使用
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import matplotlib.pyplot as plt
# 显存监控相关
import threading
import time
import subprocess

# ==========================================
# 1. 配置
# ==========================================
MODEL_NAME = "/root/autodl-tmp/models/JunHowie/Qwen3-8B-Instruct"
DATA_PATH = "/root/autodl-tmp/ecommerce_sft/clean_sft_messages7688.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/ecommerce_sft/qwen3_8b_lora_sft_01"   # 新目录，标号记录不同LoRA实验
CUTOFF_LEN = 2048
TEMPLATE = "qwen3"

# ==========================================
# 2. 加载数据 messages 格式
# ==========================================
raw_data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_data.append(json.loads(line))

formatted_data = []
for item in raw_data:
    if "messages" in item:
        conv = item["messages"]
    elif "conversation" in item:
        conv = item["conversation"]
    else:
        continue

    # 确保以 user 开头
    while conv and conv[0]["role"] == "assistant":
        conv = conv[1:]
    # 确保以 assistant 结尾
    while conv and conv[-1]["role"] == "user":
        conv = conv[:-1]

    if len(conv) >= 2:
        formatted_data.append({"messages": conv})

print(f"有效样本数: {len(formatted_data)}")
dataset = Dataset.from_list(formatted_data)

# ==========================================
# 3. tokenizer
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 4. QLoRA：4-bit 量化配置
# ==========================================
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,                      # 启用 4-bit 量化
#     bnb_4bit_use_double_quant=True,         # 双重量化，进一步节省显存
#     bnb_4bit_quant_type="nf4",              # 使用 NF4 量化类型
#     bnb_4bit_compute_dtype=torch.bfloat16   # 反量化计算时的精度
# )

# ==========================================
# 5. 模型加载（QLoRA：使用量化配置，不再指定 dtype）
# ==========================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config,         # 要用qlora时把下面dtype注释掉再传入量化配置
    dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=False,
)

model.config.use_cache = False

# ==========================================
# 6. LoRA 配置（可扩大 target_modules，QLoRA 显存充足）
# ==========================================
lora_config = LoraConfig(
    r=16,                                   # 可以改成r =8/16/64做消融试验
    lora_alpha=32,
    target_modules=[                        # 完整注意力 + MLP 模块
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ==========================================
# 7. 训练参数（建议开启梯度检查点以进一步优化显存）
# ==========================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_seq_length=CUTOFF_LEN,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                              # 混合精度训练
    gradient_checkpointing=False,            
    report_to="tensorboard",
    save_total_limit=2,
    packing=False,
    remove_unused_columns=True,
    ddp_timeout=180000000,
)

# ==========================================
# 8. SFTTrainer
# ==========================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()

# ==========================================
# 9. 显存监控函数（与之前相同）
# ==========================================
def collect_gpu_mem(interval, stop_event, data):
    while not stop_event.is_set():
        try:
            res = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            mem = int(res.stdout.strip())
        except:
            mem = 0
        data.append((time.time(), mem))
        time.sleep(interval)

mem_data = []
stop_event = threading.Event()
monitor_thread = threading.Thread(target=collect_gpu_mem, args=(5, stop_event, mem_data))
monitor_thread.start()

# ==========================================
# 10. 训练
# ==========================================
trainer.train()

# 停止显存监控并画图
stop_event.set()
monitor_thread.join()

if mem_data:
    times, mems = zip(*mem_data)
    t0 = times[0]
    rel_times = [t - t0 for t in times]
    plt.figure()
    plt.plot(rel_times, mems)
    plt.xlabel('Time (s)')
    plt.ylabel('GPU Memory Used (MB)')
    plt.title('GPU Memory Usage (QLoRA)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "gpu_memory.png"))
    print(f"显存变化图已保存至 {OUTPUT_DIR}/gpu_memory.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存训练配置
config_dict = {
    "model": MODEL_NAME,
    "template": TEMPLATE,
    "cutoff_len": CUTOFF_LEN,
    "lora_r": 16,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    "batch_size": 1,
    "grad_accum": 2,
    "lr": 2e-4,
    "epochs": 1,
    "scheduler": "cosine",
    "warmup_ratio": 0.1,
    "quantization": "4-bit",                # 标记为 QLoRA 实验
}
with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
    json.dump(config_dict, f, ensure_ascii=False, indent=2)

# ==========================================
# 11. 保存 loss 曲线
# ==========================================
log_history = trainer.state.log_history
train_loss = [x["loss"] for x in log_history if "loss" in x]
steps = [x["step"] for x in log_history if "loss" in x]

if train_loss:
    plt.figure()
    plt.plot(steps, train_loss)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
    print(f"Loss 曲线已保存至 {OUTPUT_DIR}/loss.png")

# ==========================================
# 12. 推理样例
# ==========================================
model = trainer.model
model.eval()

test_file = os.path.join(OUTPUT_DIR, "inference_samples.txt")
with open(test_file, "w", encoding="utf-8") as fout:
    for i in range(min(5, len(formatted_data))):
        messages = formatted_data[i]["messages"]
        prompt_messages = messages[:-1]

        try:
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            ).to(model.device)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                prompt,
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=5,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            generated[0][prompt.shape[1]:],
            skip_special_tokens=True
        )
        ground_truth = messages[-1]["content"]

        fout.write(f"=== Sample {i + 1} ===\n")
        fout.write(f"Context:\n{tokenizer.decode(prompt[0], skip_special_tokens=False)}\n\n")
        fout.write(f"Generated:\n{response}\n\n")
        fout.write(f"Ground Truth:\n{ground_truth}\n\n\n")

print(f"推理样例已保存至 {test_file}")

# ==========================================
# 13. 保存 LoRA adapter
# ==========================================
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print("全流程完成！")