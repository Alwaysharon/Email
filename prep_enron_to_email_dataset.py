import pandas as pd
import pathlib

# ========== 配置区：根据你的 Enron CSV 改这里 ==========
# 1. Enron 数据的位置
# 例如：data/emails.csv  或  data/enron.csv
ENRON_INPUT_FILE = pathlib.Path("data/emails.csv")

# 2. 文本列和标签列在 Enron CSV 里的“原始列名”
#    ❗ 这两个名字要换成你实际看到的列名
TEXT_COLUMN = "text"        # 比如可能叫 "message" / "content" / "body"
LABEL_COLUMN = "category"      # 比如可能叫 "category" / "Class" / "folder"

# 3. 如果你的标签是数字或别的名字，可以在这里做映射（可选）
#    如果本来就是你想要的类别名，可以留空 {}
LABEL_MAPPING = {
    # 例子：
    # 0: "Ham",
    # 1: "Spam",
    # 或者：
    # "finance_dept": "Finance",
    # "hr_team": "HR",
}

# 输出文件（就是你的项目现在在用的那个）
OUTPUT_PATH = pathlib.Path("data/email_dataset.csv")

# =====================================================

def main():
    print(f"读取 Enron 数据：{ENRON_INPUT_FILE}")
    if not ENRON_INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到输入文件：{ENRON_INPUT_FILE}")
    df = pd.read_csv(ENRON_INPUT_FILE)
    print("原始列名：", list(df.columns))
    # 检查指定的列名是否存在
    for column in (TEXT_COLUMN, LABEL_COLUMN):
        if column not in df.columns:
            raise ValueError(f"在输入文件中找不到列: {column}")
    # 只保留需要的两列，并重命名为 email_text / category
    output_df = df[[TEXT_COLUMN, LABEL_COLUMN]].rename(columns={
        TEXT_COLUMN: "email_text",
        LABEL_COLUMN: "category",
    })
    # 类型和空值处理
    output_df["email_text"] = output_df["email_text"].astype(str).str.strip()
    output_df["category"] = output_df["category"].astype(str).str.strip()
    # 如果有标签映射（数字 -> 文本），在这里应用
    if LABEL_MAPPING:
        print("应用标签映射 LABEL_MAPPING ...")
        output_df["category"] = output_df["category"].map(LABEL_MAPPING).fillna(output_df["category"])
    # 删掉空文本或空标签
    initial_count = len(output_df)
    output_df = output_df.dropna(subset=["email_text", "category"])
    output_df = output_df[output_df["email_text"].str.len() > 0]
    final_count = len(output_df)
    removed_count = initial_count - final_count
    print(f"删除空行 {removed_count} 条，最终样本量：{final_count}")
    print("类别分布：")
    print(output_df["category"].value_counts())
    # 保存成你的项目需要的格式
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n✅ 已保存为：{OUTPUT_PATH}")
    print("格式示例：")
    print(output_df.head())

if __name__ == "__main__":
    main()
