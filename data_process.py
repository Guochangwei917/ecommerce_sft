import re
import json
import pandas as pd
from pathlib import Path

# ================= 配置区 =================
# 建议：把 txt 文件和这个脚本放在同一个文件夹，直接写文件名
INPUT_PATH = "ecommerce_sft/京东客服对话.txt"
OUTPUT_PATH = "commerce_sft/clean_sft_messages.jsonl"

# ================= 清洗函数区（保持不变）=================
def clean_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)

    # 全角符号统一替换
    text = text.replace("［", "[").replace("］", "]")
    text = text.replace("", "").replace("", "")

    # 🔥 核心1：全覆盖过滤京东商品链接，你示例的链接会被完全删掉
    text = re.sub(r"https?://(www\.)?item\.jd\.com/\d+\.html(\?\S*)?", "", text)
    # 兜底：过滤所有其他京东相关的http/https链接（可选，不需要可以注释掉）
    # text = re.sub(r"https?://\S*jd\.com\S*", "", text)

    # 🔥 核心2：精准过滤单独出现的6-20位数字编号（商品ID/订单号/物流单号）
    # 规则：只删单独出现的编号，不和文字绑定的正常数字完全保留
    text = re.sub(r"(?<![\u4e00-\u9fa5a-zA-Z])\d{6,20}(?![\u4e00-\u9fa5a-zA-Z])", "", text)

    # 🔥 核心3：过滤重复的数字串（比如"16641813169?16641813169"）
    text = re.sub(r"(\d{6,})([^\d]+?\1)+", "", text)

    # 过滤11位手机号（带/不带分隔符）
    text = re.sub(r"1[3-9]\d{9}", "", text)
    text = re.sub(r"1[3-9]\d[-\s]?\d{4}[-\s]?\d{4}", "", text)

    # 原有过滤规则：系统表情、特殊字符
    text = re.sub(r"#E[-－]?[sS5]?\s*\[[^\]]{0,30}\]", "", text)
    text = re.sub(r"#E[-－]?[sS5]?", "", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # 合并多余空白，收尾
    text = re.sub(r"\s+", " ", text).strip()

    return text

def is_system_like(text: str) -> bool:
    patterns = [r"^顾客通过点击.*信息发送", r"^\[订单编号:", r"^咨询订单号:"]
    return any(re.search(p, text) for p in patterns)

def is_useless_assistant(text: str) -> bool:
    text = clean_text(text)
    if not text: return True
    useless_exact = {"在", "恩", "嗯", "好的", "好", "可以", "您客气了", "没事呢", "[姓名x]"}
    if text in useless_exact: return True
    marketing_patterns = [
        r"很高兴遇到您这么善解人意的客户", r"遇到像您这样宽容的客户",
        r"还辛苦您点击", r"感谢您对京东的支持", r"请问还有其他还可以帮到您的吗"
    ]
    return any(re.search(p, text) for p in marketing_patterns)

def dedup_lines(lines):
    result, seen = [], set()
    for line in lines:
        line = clean_text(line)
        if not line or is_system_like(line): continue
        key = re.sub(r"\s+", "", line)
        if key not in seen:
            seen.add(key)
            result.append(line)
    return result

def dedup_sentences(text: str) -> str:
    text = clean_text(text)
    if not text: return ""
    parts = re.split(r"([。！？!?~])", text)
    sentences, result, seen = [], [], set()
    for i in range(0, len(parts), 2):
        sentence = parts[i].strip()
        punc = parts[i + 1] if i + 1 < len(parts) else ""
        if sentence: sentences.append(sentence + punc)
    for sentence in sentences:
        key = re.sub(r"\s+", "", sentence)
        if key not in seen:
            seen.add(key)
            result.append(sentence)
    return "".join(result).strip()

def flush_buffer(messages, role, buffer):
    lines = dedup_lines(buffer)
    if not lines: return
    content = dedup_sentences("\n".join(lines))
    if not content: return
    if role == "assistant" and is_useless_assistant(content): return
    messages.append({"role": role, "content": content})

def build_messages(group):
    messages, current_role, buffer = [], None, []
    for _, row in group.iterrows():
        role = "assistant" if int(row["waiter_send"]) == 1 else "user"
        text = clean_text(row["content"])
        if not text: continue
        if current_role is None:
            current_role, buffer = role, [text]
        elif role == current_role:
            buffer.append(text)
        else:
            flush_buffer(messages, current_role, buffer)
            current_role, buffer = role, [text]
    if buffer: flush_buffer(messages, current_role, buffer)
    
    while messages and messages[0]["role"] == "assistant": messages.pop(0)
    while messages and messages[-1]["role"] == "user": messages.pop()
    
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] = dedup_sentences(merged[-1]["content"] + "\n" + msg["content"])
        else:
            merged.append(msg)
    return merged

def is_good_conversation(messages):
    if len(messages) < 2: return False
    if messages[0]["role"] != "user" or messages[-1]["role"] != "assistant": return False
    if sum(len(m["content"]) for m in messages) < 10: return False
    return True

# ================= 🔥 全新的主程序：完全不用 pd.read_csv =================
def main():
    input_path = Path(INPUT_PATH)
    
    # 1. 手动读取文件，彻底避免解析错误
    print(f"正在读取: {input_path}")
    raw_data_list = []
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{INPUT_PATH}'")
        print("   请把 txt 文件和脚本放在同一个文件夹，或者修改 INPUT_PATH 为完整路径")
        return

    # 跳过表头，从第2行开始
    for line in lines[1:]:
        line = line.rstrip("\n")
        if not line.strip(): continue
        
        # 🔥 只分割前5个制表符，剩下的全是 content
        parts = line.split("\t", 5)
        
        if len(parts) == 6:
            raw_data_list.append({
                "session_id": parts[0],
                "user_id": parts[1],
                "waiter_send": parts[2],
                "is_transfer": parts[3],
                "is_repeat": parts[4],
                "content": parts[5]
            })

    # 2. 手动构建 DataFrame
    df = pd.DataFrame(raw_data_list)
    print(f"读取到 {len(df)} 条消息")

    # 3. 后续清洗逻辑
    df["waiter_send"] = df["waiter_send"].astype(int)
    df = df[df["is_repeat"].astype(str) != "1"]

    clean_items = []
    for session_id, group in df.groupby("session_id", sort=False):
        messages = build_messages(group)
        if is_good_conversation(messages):
            clean_items.append({"messages": messages})

    # 4. 保存结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in clean_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"原始会话数: {df['session_id'].nunique()}")
    print(f"清洗后会话数: {len(clean_items)}")
    print(f"已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()