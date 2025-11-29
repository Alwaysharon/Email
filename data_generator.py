import random
import pandas as pd

# Sample email templates for different categories
category_templates = {
    'HR': [
        "Please review the attached job description and let me know your thoughts on the candidate requirements.",
        "Reminder: All employees must complete the mandatory training by end of month.",
        "We are updating our employee handbook. Please review the changes in section 3.2.",
        "Your annual performance review is scheduled for next week. Please prepare your self-assessment.",
        "New hire orientation will be conducted on Monday at 9 AM in conference room A.",
        "Please submit your timesheet by Friday EOD to ensure timely payroll processing.",
        "We're organizing a team building event next month. Please confirm your attendance.",
        "Updated vacation policy is now in effect. Please review the new guidelines.",
        "Interview feedback needed for candidate John Smith who applied for the marketing position.",
        "Employee satisfaction survey results are now available on the company portal."
    ],
    'Finance': [
        "Q3 budget allocation needs your approval before we can proceed with the marketing campaign.",
        "Monthly expense report is overdue. Please submit by EOD today.",
        "Invoice #12345 requires your signature for payment processing.",
        "Year-end financial statements are ready for review. Meeting scheduled for Thursday.",
        "Petty cash reconciliation shows a discrepancy of $150. Please investigate.",
        "New credit card policy requires department head approval for expenses over $500.",
        "Tax documents for 2024 need to be submitted to accounting by March 15th.",
        "Budget variance report shows 15% overspend in office supplies category.",
        "Please provide cost estimates for the new software implementation project.",
        "Audit preparations begin next week. Please gather all receipts from last quarter."
    ],
    'IT': [
        "System maintenance scheduled for this weekend. Email services may be intermittent.",
        "New software update available for download. Please install version 2.1.3.",
        "Password reset request for user account requires manager approval.",
        "Network outage affecting building 2 has been resolved. Services are now restored.",
        "Security patch installation mandatory for all workstations by Friday.",
        "Your laptop needs to be returned for hardware upgrade. IT will contact you to schedule.",
        "VPN access request approved. Login credentials will be sent separately.",
        "Backup verification completed successfully. All data is secure and recoverable.",
        "New printer installed on floor 3. Driver installation guide attached.",
        "Help desk ticket #789 regarding email synchronization has been resolved."
    ],
    'Marketing': [
        "Campaign performance metrics show 25% increase in engagement this quarter.",
        "New product launch materials are ready for review. Feedback needed by Wednesday.",
        "Social media strategy meeting rescheduled to Friday 2 PM in conference room B.",
        "Brand guidelines updated to include new logo specifications and color palette.",
        "Customer survey results indicate high satisfaction with our recent campaigns.",
        "Trade show booth setup begins Monday. Marketing team please arrive by 8 AM.",
        "Website analytics report shows significant traffic increase from mobile devices.",
        "Press release draft attached for approval. Publication scheduled for next week.",
        "Competitor analysis reveals new pricing strategies we should consider.",
        "Content calendar for next month needs input from product development team."
    ],
    'Legal': [
        "Contract review completed. Minor revisions suggested in section 4.2.",
        "Compliance audit scheduled for next month. Please prepare necessary documentation.",
        "New data privacy regulations require updates to our terms of service.",
        "Intellectual property registration successful. Certificate attached for records.",
        "Legal opinion requested regarding the new vendor agreement terms.",
        "Litigation settlement reached. Details will be shared in executive meeting.",
        "Employment law changes affect our hiring practices. HR briefing scheduled.",
        "Risk assessment completed for the merger proposal. Report attached.",
        "Trademark application approved. Registration number is TM2024-5678.",
        "Regulatory compliance checklist updated to reflect recent law changes."
    ],
    'Operations': [
        "Supply chain disruption expected next week. Alternative vendors identified.",
        "Facility maintenance scheduled for Saturday morning. Access will be restricted.",
        "Inventory levels are running low for office supplies. Reorder requested.",
        "Safety inspection passed with no violations. Certificate valid for one year.",
        "New vendor onboarding process requires operations team approval.",
        "Shipping delays affecting customer deliveries. Communication strategy needed.",
        "Quality control metrics show improvement in defect rates this quarter.",
        "Equipment calibration due for manufacturing line 3. Downtime scheduled.",
        "Process optimization resulted in 12% efficiency improvement last month.",
        "Environmental compliance report submitted to regulatory authorities."
    ]
}

# 每类150条，合计900条（够训一个demo）
rows = [{"text": random.choice(category_templates[c]), "category": c} for c in category_templates for _ in range(150)]

df = pd.DataFrame({'email_text': rows, 'category': [category for category, templates in category_templates.items() for _ in range(150)]}).sample(frac=1).reset_index(drop=True)
df.to_csv('data/email_dataset.csv', index=False)
print("Dataset generated with", len(df), "emails")
print("Categories:", df['category'].value_counts().to_dict())
print("Sample emails:")
for i in range(3):
    print(f"Category: {df.iloc[i]['category']}")
    print(f"Email: {df.iloc[i]['email_text'][:100]}...")
    print("-" * 50)
