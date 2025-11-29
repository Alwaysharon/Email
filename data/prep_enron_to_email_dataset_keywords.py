import re
from pathlib import Path

import pandas as pd
from typing import Optional


INPUT_FILE = Path("data/emails.csv")          # Enron 原始 CSV
OUTPUT_FILE = Path("data/email_dataset.csv")  # 转换后的输出（干净 + 均衡）


# ================== 1. 关键词定义（可以根据需要自己调） ==================

CATEGORY_KEYWORDS = {
    "HR": [
        "job description", "candidate", "hiring", "recruitment",
        "mandatory training", "training session", "employee handbook",
        "performance review", "annual review",
        "new hire", "orientation",
        "timesheet", "payroll", "salary", "compensation",
        "team building", "vacation policy",
        "interview feedback", "employee satisfaction",
        "hr "
    ],
    "Finance": [
        "budget", "expense report", "expense reports", "invoice",
        "year-end financial", "financial statements",
        "petty cash", "credit card", "tax", "variance report",
        "cost estimate", "audit", "audited", "receipts",
        "accounting", "ledger", "profit", "loss", "revenue"
    ],
    "IT": [
        "system maintenance", "maintenance window",
        "software update", "patch", "upgrade",
        "password reset", "account locked",
        "network outage", "server", "router", "switch",
        "security patch", "virus", "malware",
        "workstations", "desktop", "laptop",
        "hardware upgrade", "vpn",
        "backup verification", "restore",
        "printer", "help desk", "it "
    ],
    "Marketing": [
        "campaign", "marketing campaign", "promotion",
        "product launch", "launch materials",
        "social media", "brand guidelines", "branding",
        "customer survey", "trade show", "booth",
        "website analytics", "traffic", "page views",
        "press release", "press statement",
        "competitor analysis", "pricing strategy",
        "content calendar", "newsletter"
    ],
    "Legal": [
        "contract", "agreement", "nda", "non-disclosure",
        "compliance audit", "compliance review",
        "data privacy", "privacy policy",
        "terms of service", "terms and conditions",
        "intellectual property", "ip rights",
        "legal opinion", "litigation", "settlement",
        "employment law", "regulatory", "regulation",
        "risk assessment", "trademark", "patent",
        "compliance checklist"
    ],
    "Operations": [
        "supply chain", "logistics", "warehouse",
        "facility maintenance", "building maintenance",
        "inventory", "stock levels", "office supplies",
        "safety inspection", "health and safety",
        "new vendor", "vendor onboarding",
        "shipping delays", "delivery delays",
        "quality control", "defect rate",
        "equipment calibration", "downtime",
        "process optimization", "efficiency improvement",
        "environmental compliance", "operations"
    ],
}


# ================== 2. 工具函数：清理正文 ==================

HEADER_PREFIXES = (
    "message-id:", "date:", "from:", "to:", "subject:", "cc:", "bcc:",
    "x-", "content-type:", "mime-version:", "content-transfer-encoding:",
)


def strip_headers(raw_text: str) -> str:
    """去掉 Enron 邮件里的头部，只保留正文。"""
    if not isinstance(raw_text, str):
        return ""

    # 有些邮件是用 \r\n 分行的
    lines = raw_text.splitlines()
    body_lines = []
    header = True
    for line in lines:
        low = line.lower().strip()
        if header and any(low.startswith(p) for p in HEADER_PREFIXES):
            continue
        # 空行后基本就进入正文了
        if header and low == "":
            header = False
            continue
        if not header:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()
    # 去掉多余空白
    body = re.sub(r"\s+", " ", body)
    return body


# ================== 3. 工具函数：根据关键词打标签 ==================

def classify_by_keywords(text: str, min_hits: int = 2) -> Optional[str]:
    """
    根据关键词匹配次数打标签：
    - 对每个类别统计命中的关键词数量
    - 选择命中最多的类别，且命中数 >= min_hits 才接受
    - 否则返回 None（不打标签）
    """
    if not isinstance(text, str):
        return None

    t = text.lower()
    if not t.strip():
        return None

    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                scores[cat] += 1

    best_cat = max(scores, key=scores.get)
    if scores[best_cat] >= min_hits:
        return best_cat
    return None


# ================== 4. 主流程：清理 + 打标签 + 均衡采样 ==================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print("原始列名:", list(df.columns))

    if "message" not in df.columns:
        raise ValueError("在 data/emails.csv 中找不到 'message' 列，请确认文件格式。")

    # 1) 清理正文
    print("清理邮件正文（去掉头部信息）...")
    df["email_text"] = df["message"].astype(str).apply(strip_headers)

    # 丢掉太短的邮件
    df = df[df["email_text"].str.len() >= 50]  # 长度阈值可调
    print("清理后样本数:", len(df))

    # 2) 用关键词打标签（更严格：至少命中 2 个关键词）
    print("根据关键词打标签...")
    df["category"] = df["email_text"].apply(lambda x: classify_by_keywords(x, min_hits=2))

    before = len(df)
    df = df.dropna(subset=["category"])
    after = len(df)

    print(f"总共 {before} 封清理后的邮件，其中成功打上六大类标签的有 {after} 封。")
    print("类别分布：")
    print(df["category"].value_counts())

    # 3) 做类别均衡（只下采样，不上采样，保证干净）
    counts = df["category"].value_counts()
    min_count = counts.min()
    # 不要太大，给一个上限，比如每类最多 3000 条
    target_per_class = min(min_count, 3000)

    print(f"\n按每类 {target_per_class} 条做均衡下采样...")
    balanced_list = []
    for cat, group in df.groupby("category"):
        if len(group) >= target_per_class:
            sampled = group.sample(target_per_class, random_state=42)
        else:
            # 理论上不会走到这里，因为 target_per_class <= min_count
            sampled = group
        balanced_list.append(sampled)

    balanced_df = (
        pd.concat(balanced_list)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    print("均衡后类别分布：")
    print(balanced_df["category"].value_counts())

    # 4) 只保留 email_text, category 两列输出
    out = balanced_df[["email_text", "category"]]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n✅ 已保存为: {OUTPUT_FILE}")
    print("示例前 5 行：")
    print(out.head())


if __name__ == "__main__":
    main()
