import argparse
import csv
import os
import re
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import tqdm

# ---------------------------
# 关键字规则（弱标签）
# ---------------------------
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "HR": [
        r"\bbenefit(s)?\b", r"\bleave\b", r"\bhiring\b", r"\brecruit(ment|ing)?\b", r"\borientation\b",
        r"\bpayroll\b", r"\bemployee\b", r"\bvacation\b", r"\bpolicy\b"
    ],
    "FINANCE": [
        r"\binvoice\b", r"\bbudget\b", r"\bexpense(s)?\b", r"\breimburse(ment|)\b", r"\bpayment(s)?\b",
        r"\bbilling\b", r"\bpo\b", r"\bpurchase order\b", r"\baccount(ing)?\b", r"\bforecast\b"
    ],
    "IT": [
        r"\bpassword\b", r"\bvpn\b", r"\bserver\b", r"\boutage\b", r"\bmaintenance\b", r"\bhelpdesk\b",
        r"\bticket\b", r"\bsystem(s)?\b", r"\bnetwork\b", r"\bupgrade\b", r"\breset\b"
    ],
    "MARKETING": [
        r"\bcampaign\b", r"\bnewsletter\b", r"\bpromotion(s)?\b", r"\bbranding\b", r"\bevent(s)?\b",
        r"\bsocial media\b", r"\bpress release\b", r"\badvertis(ing|ement)\b"
    ],
    "LEGAL": [
        r"\bcontract(s)?\b", r"\bnda\b", r"\bnon-?disclosure\b", r"\bcompliance\b", r"\bclause\b",
        r"\blitigation\b", r"\bcounsel\b", r"\bterms?\b", r"\bagreement\b"
    ],
    "OPERATIONS": [
        r"\binventory\b", r"\bshipment(s)?\b", r"\blogistics?\b", r"\bfacilit(y|ies)\b", r"\bsuppl(y|ies)\b",
        r"\bschedule(d|s|ing)?\b", r"\bprocurement\b", r"\bwarehouse\b"
    ],
}
CATEGORIES = ["HR", "FINANCE", "IT", "MARKETING", "LEGAL", "OPERATIONS"]
HEADER_BODY_SPLIT_REGEX = re.compile(r"\r?\n\r?\n", re.MULTILINE)

def sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"(?im)^(from|to|cc|bcc|subject|sent|date):.*?$", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def merge_subject_body(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    parts = HEADER_BODY_SPLIT_REGEX.split(raw, maxsplit=1)
    if len(parts) == 2:
        headers, body = parts
    else:
        headers, body = "", raw
    m = re.search(r"(?im)^subject:\s*(.*)$", headers)
    subject = m.group(1).strip() if m else ""
    return (subject + "\n" + body).strip()

def assign_weak_label(text: str) -> Optional[str]:
    text = text.lower()
    best, hits_best = None, 0
    for lab in CATEGORIES:
        hits = sum(1 for kw in CATEGORY_KEYWORDS[lab] if re.search(kw, text))
        if hits > hits_best:
            best, hits_best = lab, hits
    return best if hits_best >= 1 else None

def process_chunk(df, writer, stats, min_length: int, max_per_label: int = 0):
    if "message" not in df.columns:
        raise ValueError("Column 'message' not found. Are you using Kaggle emails.csv?")
    for raw in df["message"].astype(str).tolist():
        text = sanitize_text(merge_subject_body(raw))
        if len(text) < min_length:
            continue
        label = assign_weak_label(text)
        if label is None:
            continue
        if max_per_label and stats["per_label"].get(label, 0) >= max_per_label:
            continue
        writer.writerow({"email_text": text, "label": label})
        stats["written"] += 1
        stats["per_label"][label] = stats["per_label"].get(label, 0) + 1

def main():
    ap = argparse.ArgumentParser(description="Prepare Enron emails.csv -> labeled CSV (weak labels)")
    ap.add_argument("--input", required=True, help="Path to Kaggle emails.csv (1.43GB)")
    ap.add_argument("--output", default="data/email_dataset.csv", help="Output CSV path")
    ap.add_argument("--chunksize", type=int, default=5000, help="Rows per chunk to read")
    ap.add_argument("--sample", type=int, default=10000, help="Target number of labeled rows to write")
    ap.add_argument("--min_length", type=int, default=30, help="Minimal cleaned text length")
    ap.add_argument("--balance", action="store_true", help="Try to balance classes evenly")
    args = ap.parse_args()
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    f_out = open(args.output, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=["email_text", "label"])
    writer.writeheader()
    stats = {"written": 0, "per_label": {}}
    try:
        reader = pd.read_csv(args.input, chunksize=args.chunksize, encoding="utf-8")
    except UnicodeDecodeError:
        reader = pd.read_csv(args.input, chunksize=args.chunksize, encoding="latin-1")
    max_per_label = 0
    if args.balance and args.sample > 0:
        max_per_label = max(1, args.sample // len(CATEGORIES))
    for chunk in tqdm.tqdm(reader, desc="Reading chunks"):
        process_chunk(chunk, writer, stats, args.min_length, max_per_label)
        if stats["written"] >= args.sample > 0:
            break
    f_out.close()
    removed = stats["written"]
    print(f"\nSaved -> {args.output}")
    print(f"Total labeled rows: {stats['written']}")
    for lab in CATEGORIES:
        print(f"{lab:>10}: {stats['per_label'].get(lab, 0)}")

if __name__ == "__main__":
    main()
