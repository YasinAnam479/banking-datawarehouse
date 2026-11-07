import sqlite3

import pandas as pd

DB_FILE = "bank_dw.db"
XLSX_FILE = "bank_dw.xlsx"

SHEETS = {
    "Branches": "SELECT * FROM branch",
    "Customers": """SELECT c.customer_id, c.age, o.occupation_name AS occupation, c.marital, c.education,
                           c.credit_default, c.housing_loan, c.personal_loan, c.avg_balance
                    FROM customer c JOIN occupation o ON o.occupation_id = c.occupation_id""",
    "Accounts": "SELECT * FROM account",
    "Transactions": """SELECT t.txn_id, t.account_no, tt.type_name AS txn_type, m.merchant_name AS merchant,
                              t.txn_time, t.amount, t.channel
                       FROM transactions t
                       JOIN transaction_type tt ON tt.type_id = t.type_id
                       JOIN merchant m ON m.merchant_id = t.merchant_id""",
    "Txn_Risk_Flags": "SELECT * FROM risk_flag",
    "Customer_Risk_Flags": "SELECT * FROM customer_risk_flag",
    "Customer_Risk_Summary": "SELECT * FROM v_customer_risk ORDER BY total_risk DESC",
}

conn = sqlite3.connect(DB_FILE)
with pd.ExcelWriter(XLSX_FILE, engine="openpyxl") as writer:
    for name, query in SHEETS.items():
        df = pd.read_sql(query, conn)
        df.to_excel(writer, sheet_name=name, index=False)
        ws = writer.sheets[name]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(width + 3, 35)
        print(f"Sheet '{name}': {len(df)} rows")
conn.close()
print(f"\nDone. Excel file saved as {XLSX_FILE}")
