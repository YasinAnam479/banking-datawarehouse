"""
ML anomaly detection on the bank data warehouse (Isolation Forest)

Reads transactions from bank_dw.db, trains an Isolation Forest on 80% of them,
tests it on the 20% held-out set, then scores every transaction and saves the
results back into the warehouse (tables: ml_anomaly, ml_metrics).

Run AFTER bank_etl_v2.py.   Run:  python ml_risk.py
Needs: pip install scikit-learn pandas numpy
"""
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DB_FILE = "bank_dw.db"
CONTAMINATION = 0.006     # expected share of anomalies (0.6%)
FEATURES = ["log_amount", "is_night", "txns_last_hour", "amount_vs_median", "log_gap"]

conn = sqlite3.connect(DB_FILE)

# ---------------------------------------------------------------
# 1. Load transactions from the warehouse
# ---------------------------------------------------------------
txn = pd.read_sql("SELECT txn_id, account_no, txn_time, amount FROM transactions",
                  conn, parse_dates=["txn_time"])
rules = pd.read_sql("SELECT txn_id, rule FROM risk_flag", conn)
print(f"[1] Loaded {len(txn)} transactions from the warehouse")

# ---------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------
txn = txn.sort_values(["account_no", "txn_time"]).reset_index(drop=True)
txn["log_amount"] = np.log1p(txn["amount"])
txn["hour"] = txn["txn_time"].dt.hour
txn["is_night"] = (txn["hour"] < 5).astype(int)

# how many transactions did this account make in the last hour (including this one)?
counts = [g.set_index("txn_time")["amount"].rolling("1h").count().values
          for _, g in txn.groupby("account_no", sort=False)]
txn["txns_last_hour"] = np.concatenate(counts)

# how large is this amount compared to the account's usual (median) amount?
median = txn.groupby("account_no")["amount"].transform("median").replace(0, np.nan)
txn["amount_vs_median"] = (txn["amount"] / median).fillna(1.0)
# minutes since this account's previous transaction (very small = rapid-fire activity)
gap = txn.groupby("account_no")["txn_time"].diff().dt.total_seconds().div(60).fillna(1e5).clip(upper=1e4)
txn["log_gap"] = np.log1p(gap)
print(f"[2] Features built: {FEATURES}")

# ---------------------------------------------------------------
# 3. Train / test split (80% train, 20% held-out test)
# ---------------------------------------------------------------
X = txn[FEATURES]
train_idx, test_idx = train_test_split(txn.index, test_size=0.2, random_state=42)
model = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=42)
model.fit(X.loc[train_idx])
print(f"[3] Trained Isolation Forest on {len(train_idx)} transactions, testing on {len(test_idx)} held-out")

# ---------------------------------------------------------------
# 4. Evaluate on the held-out test set
#    The Isolation Forest is unsupervised, so there is no true "fraud" label.
#    We use the serious rule-based alerts (HIGH_VALUE, VELOCITY) as a proxy label.
# ---------------------------------------------------------------
serious_ids = set(rules.loc[rules["rule"].isin(["HIGH_VALUE", "VELOCITY"]), "txn_id"])
txn["rule_serious"] = txn["txn_id"].isin(serious_ids).astype(int)

test = txn.loc[test_idx]
pred_test = (model.predict(X.loc[test_idx]) == -1).astype(int)
prec = precision_score(test["rule_serious"], pred_test, zero_division=0)
rec = recall_score(test["rule_serious"], pred_test, zero_division=0)
f1 = f1_score(test["rule_serious"], pred_test, zero_division=0)
print(f"[4] Held-out test results vs rule-based alerts: precision={prec:.2f}  recall={rec:.2f}  F1={f1:.2f}")

# ---------------------------------------------------------------
# 5. Score ALL transactions and save into the warehouse
# ---------------------------------------------------------------
txn["anomaly_score"] = -model.score_samples(X)          # higher = more unusual
txn["is_anomaly"] = (model.predict(X) == -1).astype(int)
txn["rule_any"] = txn["txn_id"].isin(set(rules["txn_id"])).astype(int)

txn[["txn_id", "anomaly_score", "is_anomaly"]].to_sql("ml_anomaly", conn, if_exists="replace", index=False)
pd.DataFrame([{"model": "Isolation Forest", "contamination": CONTAMINATION,
               "train_rows": len(train_idx), "test_rows": len(test_idx),
               "precision": prec, "recall": rec, "f1": f1}]).to_sql("ml_metrics", conn, if_exists="replace", index=False)
conn.commit()
conn.close()

n_ml = int(txn["is_anomaly"].sum())
both = int(((txn["is_anomaly"] == 1) & (txn["rule_serious"] == 1)).sum())
ml_only = int(((txn["is_anomaly"] == 1) & (txn["rule_any"] == 0)).sum())
print(f"[5] Saved scores. Model flagged {n_ml} transactions:")
print(f"      {both} also caught by serious rules (HIGH_VALUE / VELOCITY)")
print(f"      {ml_only} are NEW anomalies that no rule flagged")
print("\nDone. Tables 'ml_anomaly' and 'ml_metrics' added to bank_dw.db")
