# =========================================================
# 精简评估脚本：BERTScore + 核心业务指标（LLM Judge）
# =========================================================
import os
import torch
import pandas as pd
import numpy as np
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

# =========================================================
# 1. 配置区域
# =========================================================
MODEL_PATH = "/root/autodl-tmp/models/JunHowie/Qwen3-8B-Instruct"
LORA_PATH = "/root/autodl-tmp/ecommerce_sft/qwen3_8b_qlora_sft_01/checkpoint-3844" # 使用训练完之后得到的LoRA 权重路径
TEST_FILE = "/root/autodl-tmp/ecommerce_sft/value_data100.json"                    # 测试集（包含 question 和 answer 字段）
OUTPUT_FILE = "/root/autodl-tmp/ecommerce_sft/qwen3_8b_qlora_sft_01/eval_results_01.csv" # 评估结果保存路径

HAS_REFERENCE = True
QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"

SYSTEM_PROMPT = "你是专业的电商客服，请用友好、专业的态度回答用户的问题。"
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
# 采样参数（仅在 DO_SAMPLE=True 时生效）
# TEMPERATURE = 0.7
# TOP_P = 0.9

REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 5

# ======== LLM Judge 配置 ========
JUDGE_CLIENT = OpenAI(
    api_key="",          # 替换为真实 Key,我用的是deepseek的裁判模型接口，和 openai 的使用方式一样
    base_url="https://api.deepseek.com"
)
JUDGE_MODEL = "deepseek-chat"

# =========================================================
# 2. 加载测试集（兼容 messages 格式）
# =========================================================
print("读取测试集...")

# 读取原始 JSONL
test_samples = []
with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            test_samples.append(json.loads(line))

questions = []
references = []

for item in test_samples:
    if "messages" in item:
        user_msgs = [m for m in item["messages"] if m["role"] == "user"]
        assistant_msgs = [m for m in item["messages"] if m["role"] == "assistant"]
        if user_msgs:
            # 取最后一条用户消息作为问题
            questions.append(user_msgs[-1]["content"])
            # 如果有助手回复，则作为参考答案，否则填空
            if assistant_msgs:
                references.append(assistant_msgs[-1]["content"])
            else:
                references.append("")
        else:
            questions.append("")
            references.append("")
    else:
        # 兼容直接的 question/answer 字段（如果你以后改了格式）
        questions.append(item.get("question", ""))
        references.append(item.get("answer", ""))

# 只保留有效问题
valid_mask = [q != "" for q in questions]
questions = [q for q, m in zip(questions, valid_mask) if m]
references = [r for r, m in zip(references, valid_mask) if m]

HAS_REFERENCE = any(ref != "" for ref in references)  # 如果至少有一条非空答案，就认为有参考答案
print(f"测试样本数: {len(questions)}")
if not HAS_REFERENCE:
    references = None
    print("未检测到参考答案，将只进行推理和 LLM Judge 评估（如果开启）。")

# =========================================================
# 3. 加载 tokenizer 和模型（含 LoRA）
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=dtype, device_map="auto", trust_remote_code=True
)
if LORA_PATH and os.path.exists(LORA_PATH):
    from peft import PeftModel
    print(f"加载 LoRA: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.lm_head = model.lm_head.to(torch.bfloat16)
model.eval()

# =========================================================
# 4. 推理函数
# =========================================================
def predict(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

# =========================================================
# 5. LLM Judge 评分函数（仅保留核心业务指标）
# =========================================================

def extract_json(text):
    """安全提取 JSON 对象"""
    try:
        return json.loads(text)
    except Exception:
        pass
    # 尝试提取 ```json ``` 代码块
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # 尝试提取第一个 {...} 对象
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None

def llm_judge(question: str, pred_answer: str, gold_answer: str = None) -> dict:
    """
    调用 DeepSeek 裁判模型评估核心指标。
    如果提供了 gold_answer，则为有参考评分；否则仅基于问题+回复评测。
    """
    if gold_answer and gold_answer.strip():
        ref_part = f"\n标准回答（参考）：\n{gold_answer}\n"
        extra_note = "请结合参考答案的解决思路和关键信息来评判。"
    else:
        ref_part = ""
        extra_note = "请直接根据问题判断模型回复的质量，无需参考答案。"

    prompt = f"""你是一名专业的电商客服质检专家。请对以下模型回复进行评估。
特别说明：回复中如出现类似 [数字x]、[姓名x] 等方括号占位符，是系统生成时引入的无关标记，评阅时请直接忽略它们，不要因为它们扣减准确性或满意度分数。

用户问题：
{question}

标准回答（参考）：
{gold_answer}

模型回复：
{pred_answer}

评估维度（1-10分）：
1. solved（解决率）：回复是否解决了用户问题？0=未解决，1=已解决。
2. accuracy（准确率）：信息、流程、逻辑的准确性（忽略占位符）。
3. satisfaction（满意度）：作为用户，你对回复的满意程度（忽略占位符）。
4. overall（综合分）：整体表现。

请严格输出一个 JSON 对象：
{{
    "solved": 1,
    "accuracy": 8,
    "satisfaction": 9,
    "overall": 8,
    "comment": "简短评价"
}}"""

    try:
        response = JUDGE_CLIENT.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content
        result = extract_json(content)
        if result is None:
            raise ValueError("未能提取有效 JSON")
        result.setdefault("solved", 0)
        result.setdefault("accuracy", 0)
        result.setdefault("satisfaction", 0)
        result.setdefault("overall", 0)
        return result
    except Exception as e:
        print(f"LLM Judge 错误: {e}")
        return {"solved": 0, "accuracy": 0, "satisfaction": 0, "overall": 0, "comment": f"Error: {e}"}

# =========================================================
# 6. 批量推理与评估
# =========================================================
print("\n开始推理并评估...")
predictions = []
judge_results = []
failed_indices = []

for idx, q in enumerate(tqdm(questions, desc="处理进度")):
    # 推理
    try:
        pred = predict(q)
    except Exception as e:
        print(f"\n样本 {idx} 推理失败: {e}")
        pred = ""
        failed_indices.append(idx)
    predictions.append(pred)

    # LLM Judge：只要有推理成功就进行评分（无参考答案模式）
    if idx not in failed_indices:
        gold = references[idx] if HAS_REFERENCE and references else None
        judge_r = llm_judge(q, pred, gold_answer=gold)
    else:
        judge_r = None
    judge_results.append(judge_r)

# =========================================================
# 7. 保存详细结果
# =========================================================
save_list = []
for i in range(len(questions)):
    item = {
        "question": questions[i],
        "prediction": predictions[i],
    }
    if HAS_REFERENCE:
        item["reference"] = references[i]
    if judge_results[i] is not None:
        for k, v in judge_results[i].items():
            item[f"judge_{k}"] = v
    save_list.append(item)

save_df = pd.DataFrame(save_list)
save_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"详细结果已保存至: {OUTPUT_FILE}")

# =========================================================
# 8. 计算汇总指标（BERTScore + 核心业务指标）
# =========================================================
# 剔除推理失败的样本
valid_indices = [i for i in range(len(predictions)) if i not in failed_indices]
valid_preds = [predictions[i] for i in valid_indices]
valid_refs = [references[i] for i in valid_indices] if HAS_REFERENCE else None
valid_judges = [judge_results[i] for i in valid_indices]


# ----- LLM Judge 汇总 -----
judge_summary = {}
if any(j is not None for j in valid_judges):
    # 提取各有效裁判结果的字段
    solved_list = [j["solved"] for j in valid_judges if j is not None]
    accuracy_list = [j["accuracy"] for j in valid_judges if j is not None]
    satisfaction_list = [j["satisfaction"] for j in valid_judges if j is not None]
    overall_list = [j["overall"] for j in valid_judges if j is not None]

    # 自动解决率：解决样本数 / 有效样本数
    auto_resolution_rate = np.mean(solved_list) if solved_list else 0
    # 平均准确率
    avg_accuracy = np.mean(accuracy_list) if accuracy_list else 0
    # 平均满意度
    avg_satisfaction = np.mean(satisfaction_list) if satisfaction_list else 0
    # 平均综合分
    avg_overall = np.mean(overall_list) if overall_list else 0

    judge_summary = {
        "auto_resolution_rate": auto_resolution_rate,   # 自动解决率
        "avg_accuracy": avg_accuracy,                   # 平均准确率
        "avg_satisfaction": avg_satisfaction,           # 平均满意度
        "avg_overall": avg_overall                      # 平均综合分
    }

    print("--- 核心业务指标（LLM Judge） ---")
    print(f"自动解决率: {auto_resolution_rate:.2%}")
    print(f"平均准确率: {avg_accuracy:.2f} / 10")
    print(f"平均满意度: {avg_satisfaction:.2f} / 10")
    print(f"平均综合分: {avg_overall:.2f} / 10")
    
summary = {}
if judge_summary:
    summary.update(judge_summary)

pd.DataFrame([summary]).to_csv(OUTPUT_FILE.replace(".csv", "_core_metrics.csv"), index=False)
print("\n核心指标汇总文件已保存。")
print("全部完成！")