import os
import random
import pandas as pd
os.makedirs("data", exist_ok=True)

category_samples = {
    "HR": [
        "New employee orientation will be held next Monday at 9 AM.",
        "Please submit your leave request in the HR portal.",
        "Update your emergency contact details by Friday."
    ],
    "Finance": [
        "Please review the quarterly budget report.",
        "Invoice approval required for vendor payment.",
        "Expense policy updated; attach receipts for reimbursements."
    ],
    "IT": [
        "System maintenance tonight; services may be unavailable.",
        "Reset your account password using the IT self-service tool.",
        "Network upgrade scheduled this weekend."
    ],
    "Marketing": [
        "Approve the final draft of the campaign email.",
        "Social media assets ready for review.",
        "Schedule product launch newsletter for next week."
    ],
    "Legal": [
        "Contract review completed; see comments in section 4.2.",
        "Please sign the updated NDA.",
        "Compliance training due by end of month."
    ],
    "Operations": [
        "Inventory levels are low; reorder office supplies.",
        "Facility inspection scheduled for tomorrow morning.",
        "Logistics update: shipment delayed due to weather."
    ],
}

# 每类150条，合计900条（够训一个demo）
rows = [{"text": random.choice(category_samples[c]), "category": c} for c in category_samples for _ in range(150)]

df = pd.DataFrame(rows)
# 兼容不同脚本的列名习惯：同时写入 category 和 label 两列
df["label"] = df["category"]
df.to_csv("data/email_dataset.csv", index=False, encoding="utf-8")
print("Saved -> data/email_dataset.csv", df["category"].value_counts().to_dict())
