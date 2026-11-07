"""
Bank Transaction Data Warehouse - ETL Pipeline (v2, hybrid)

Sources
  1. test.csv                (Kaggle / UCI Bank Marketing) -> REAL customer profiles
  2. raw_branches.csv        (generated)                   -> branch master data
  3. raw_accounts.csv        (generated)                   -> core-banking accounts
  4. raw_transactions.csv    (generated)                   -> transaction feed

Run:  python bank_etl_v2.py        (keep test.csv in the same folder)
Needs: pip install pandas
"""
import os
import random
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

random.seed(42)

CUSTOMER_CSV = "test.csv"
RAW_BRANCH = "raw_branches.csv"
RAW_ACCOUNT = "raw_accounts.csv"
RAW_TXN = "raw_transactions.csv"
DB_FILE = "bank_dw.db"

HIGH_VALUE_LIMIT = 5000   # a single transaction at or above this amount is flagged
VELOCITY_COUNT = 5        # this many txns by one account within 1 hour is flagged
ODD_HOUR_END = 5          # transactions between 00:00 and 04:59 are flagged


# ===============================================================
# STEP 1: EXTRACT the real customer file
# ===============================================================
def extract_customers():
    df = pd.read_csv(CUSTOMER_CSV, sep=";")          # the Kaggle file is semicolon separated
    df.columns = [c.strip().lower() for c in df.columns]
    df.insert(0, "customer_id", [f"C{i:05d}" for i in range(1, len(df) + 1)])
    print(f"[1] Extract  : {len(df)} customers read from {CUSTOMER_CSV}")
    return df


# ===============================================================
# STEP 0: Generate the other source systems (accounts, txns)
# ===============================================================
def generate_source_systems(cust):
    branches = pd.DataFrame([
        ("BR01", "Eluru Main", "Eluru"), ("BR02", "Benz Circle", "Vijayawada"),
        ("BR03", "Brodipet", "Guntur"), ("BR04", "Madhapur", "Hyderabad"),
        ("BR05", "MVP Colony", "Visakhapatnam"), ("BR06", "Power Pet", "Eluru"),
    ], columns=["branch_code", "branch_name", "branch_city"])
    branches.to_csv(RAW_BRANCH, index=False)

    # accounts: every customer has 1 account, about 12% have a 2nd one
    accs = []
    n = 0
    for cid in cust["customer_id"]:
        for _ in range(2 if random.random() < 0.12 else 1):
            n += 1
            accs.append({"account_no": f"AC{10000000 + n}", "customer_id": cid,
                         "branch_code": random.choice(branches["branch_code"].tolist()),
                         "account_type": random.choice(["Savings", "Current", "Salary"])})
    accounts = pd.DataFrame(accs)
    accounts.to_csv(RAW_ACCOUNT, index=False)

    # transactions
    types = ["Deposit", "Withdrawal", "Transfer", "Bill Payment", "POS Purchase", "ATM Withdrawal"]
    merchants = ["Reliance Smart", "Amazon", "Flipkart", "BigBazaar", "IRCTC", "Swiggy",
                 "Zomato", "Apollo Pharmacy", "HP Petrol Pump", "Jio Recharge"]
    channels = ["Branch", "ATM", "Mobile App", "Net Banking", "POS"]
    hour_w = [1, 1, 1, 1, 1, 2, 3, 5, 8, 10, 10, 10, 10, 10, 10, 10, 9, 8, 6, 4, 3, 2, 2, 1]
    start = datetime(2026, 7, 1)
    acc_list = accounts["account_no"].tolist()
    rows, counter = [], 0

    def new_row(acc, ts, ttype, amount):
        nonlocal counter
        counter += 1
        merchant = random.choice(merchants) if ttype in ("POS Purchase", "Bill Payment") else None
        ch = {"ATM Withdrawal": "ATM", "POS Purchase": "POS"}.get(ttype, random.choice(channels))
        return {"txn_id": f"T{counter:07d}", "account_no": acc,
                "txn_time": ts.strftime("%Y-%m-%d %H:%M:%S"), "txn_type": ttype,
                "merchant": merchant, "amount": amount, "channel": ch}

    for acc in acc_list:                                   # normal traffic
        for _ in range(random.randint(3, 12)):
            day = start + timedelta(days=random.randint(0, 89))
            hour = random.choices(range(24), weights=hour_w)[0]
            ts = day.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
            amount = round(min(random.lognormvariate(5.5, 1.1), 4000), 2)
            rows.append(new_row(acc, ts, random.choice(types), amount))

    for acc in random.sample(acc_list, 120):               # injected: high value
        ts = start + timedelta(days=random.randint(0, 89), hours=random.randint(9, 18))
        rows.append(new_row(acc, ts, random.choice(["Transfer", "Withdrawal", "Deposit"]),
                            round(random.uniform(5000, 30000), 2)))

    for acc in random.sample(acc_list, 25):                # injected: velocity bursts
        base = start + timedelta(days=random.randint(0, 89), hours=random.randint(10, 20))
        for k in range(random.randint(6, 8)):
            rows.append(new_row(acc, base + timedelta(minutes=k * 5), "Transfer",
                                round(random.uniform(200, 900), 2)))

    df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)

    # make the feed dirty on purpose so the ETL has real cleaning to do
    df.loc[df.sample(frac=0.01, random_state=2).index, "amount"] = None
    df["txn_type"] = df["txn_type"].apply(
        lambda t: random.choice([t.upper(), t.lower(), f"  {t} "]) if random.random() < 0.05 else t)
    df = pd.concat([df, df.sample(frac=0.02, random_state=3)], ignore_index=True)
    df.to_csv(RAW_TXN, index=False)
    print(f"[0] Generated: {len(branches)} branches, {len(accounts)} accounts, {len(df)} raw transactions")


# ===============================================================
# Target schema: normalized to 3NF
# ===============================================================
SCHEMA = """
DROP VIEW  IF EXISTS v_customer_risk;
DROP TABLE IF EXISTS customer_risk_flag;
DROP TABLE IF EXISTS risk_flag;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS account;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS occupation;
DROP TABLE IF EXISTS branch;
DROP TABLE IF EXISTS transaction_type;
DROP TABLE IF EXISTS merchant;

CREATE TABLE branch (
    branch_code TEXT PRIMARY KEY,
    branch_name TEXT NOT NULL,
    branch_city TEXT NOT NULL
);
CREATE TABLE occupation (
    occupation_id   INTEGER PRIMARY KEY,
    occupation_name TEXT UNIQUE NOT NULL
);
CREATE TABLE customer (
    customer_id   TEXT PRIMARY KEY,
    age           INTEGER,
    occupation_id INTEGER NOT NULL REFERENCES occupation(occupation_id),
    marital       TEXT,
    education     TEXT,
    credit_default INTEGER NOT NULL,   -- 1 = has defaulted on credit
    housing_loan   INTEGER NOT NULL,
    personal_loan  INTEGER NOT NULL,
    avg_balance    REAL NOT NULL       -- average yearly balance
);
CREATE TABLE account (
    account_no   TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customer(customer_id),
    branch_code  TEXT NOT NULL REFERENCES branch(branch_code),
    account_type TEXT NOT NULL
);
CREATE TABLE transaction_type (
    type_id   INTEGER PRIMARY KEY,
    type_name TEXT UNIQUE NOT NULL
);
CREATE TABLE merchant (
    merchant_id   INTEGER PRIMARY KEY,
    merchant_name TEXT UNIQUE NOT NULL
);
CREATE TABLE transactions (
    txn_id      TEXT PRIMARY KEY,
    account_no  TEXT NOT NULL REFERENCES account(account_no),
    type_id     INTEGER NOT NULL REFERENCES transaction_type(type_id),
    merchant_id INTEGER NOT NULL REFERENCES merchant(merchant_id),
    txn_time    TEXT NOT NULL,
    amount      REAL NOT NULL CHECK (amount >= 0),
    channel     TEXT
);
CREATE TABLE risk_flag (                 -- transaction-level risk
    flag_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id   TEXT NOT NULL REFERENCES transactions(txn_id),
    rule     TEXT NOT NULL,
    severity INTEGER NOT NULL
);
CREATE TABLE customer_risk_flag (        -- customer-profile risk
    flag_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES customer(customer_id),
    rule        TEXT NOT NULL,
    severity    INTEGER NOT NULL
);
CREATE INDEX idx_txn_time    ON transactions(txn_time);
CREATE INDEX idx_txn_account ON transactions(account_no);

CREATE VIEW v_customer_risk AS
SELECT *, txn_risk + profile_risk AS total_risk FROM (
    SELECT c.customer_id, c.age, o.occupation_name AS occupation, c.avg_balance,
           COALESCE((SELECT SUM(r.severity) FROM risk_flag r
                     JOIN transactions t ON t.txn_id = r.txn_id
                     JOIN account a ON a.account_no = t.account_no
                     WHERE a.customer_id = c.customer_id), 0) AS txn_risk,
           COALESCE((SELECT SUM(f.severity) FROM customer_risk_flag f
                     WHERE f.customer_id = c.customer_id), 0) AS profile_risk
    FROM customer c JOIN occupation o ON o.occupation_id = c.occupation_id
);
"""


# ===============================================================
# STEP 2: TRANSFORM
# ===============================================================
def transform_customers(df):
    df = df.rename(columns={"job": "occupation", "default": "credit_default", "housing": "housing_loan",
                            "loan": "personal_loan", "balance": "avg_balance"})
    df = df[["customer_id", "age", "occupation", "marital", "education",
             "credit_default", "housing_loan", "personal_loan", "avg_balance"]].copy()
    for c in ["occupation", "marital", "education"]:
        df[c] = (df[c].astype(str).str.strip().str.replace(".", "", regex=False)
                 .str.replace("-", " ", regex=False).str.title())
    for c in ["credit_default", "housing_loan", "personal_loan"]:
        df[c] = df[c].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)
    df = df.dropna(subset=["age", "avg_balance"])
    print(f"[2] Transform: customers cleaned -> {len(df)} rows (yes/no -> 1/0, text standardized)")
    return df


def transform_transactions(df, valid_accounts):
    before = len(df)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()
    df["txn_type"] = df["txn_type"].str.title()
    df = df.drop_duplicates(subset="txn_id")
    n_dupes = before - len(df)
    n1 = len(df)
    df = df.dropna(subset=["amount"])
    n_nulls = n1 - len(df)
    n2 = len(df)
    df = df[df["account_no"].isin(valid_accounts)]            # referential integrity check
    n_orphans = n2 - len(df)
    df["amount"] = df["amount"].abs()
    df["txn_time"] = pd.to_datetime(df["txn_time"])
    df["merchant"] = df["merchant"].fillna("Not Applicable")
    print(f"[2] Transform: transactions {before} -> {len(df)} "
          f"(duplicates {n_dupes}, missing amount {n_nulls}, orphan accounts {n_orphans} removed)")
    return df


# ===============================================================
# STEP 3: LOAD
# ===============================================================
def load(conn, branches, cust, accounts, txn):
    conn.executescript(SCHEMA)

    occ = pd.DataFrame({"occupation_name": sorted(cust["occupation"].unique())})
    occ["occupation_id"] = range(1, len(occ) + 1)
    cust_db = cust.merge(occ, left_on="occupation", right_on="occupation_name")
    cust_db = cust_db[["customer_id", "age", "occupation_id", "marital", "education",
                       "credit_default", "housing_loan", "personal_loan", "avg_balance"]]

    ttype = pd.DataFrame({"type_name": sorted(txn["txn_type"].unique())})
    ttype["type_id"] = range(1, len(ttype) + 1)
    merch = pd.DataFrame({"merchant_name": sorted(txn["merchant"].unique())})
    merch["merchant_id"] = range(1, len(merch) + 1)

    fact = (txn.merge(ttype, left_on="txn_type", right_on="type_name")
               .merge(merch, left_on="merchant", right_on="merchant_name"))
    fact = fact[["txn_id", "account_no", "type_id", "merchant_id", "txn_time", "amount", "channel"]]
    fact["txn_time"] = fact["txn_time"].astype(str)

    branches.to_sql("branch", conn, if_exists="append", index=False)
    occ.to_sql("occupation", conn, if_exists="append", index=False)
    cust_db.to_sql("customer", conn, if_exists="append", index=False)
    accounts.to_sql("account", conn, if_exists="append", index=False)
    ttype.to_sql("transaction_type", conn, if_exists="append", index=False)
    merch.to_sql("merchant", conn, if_exists="append", index=False)
    fact.to_sql("transactions", conn, if_exists="append", index=False, chunksize=5000)
    print(f"[3] Load     : {len(branches)} branches, {len(occ)} occupations, {len(cust_db)} customers, "
          f"{len(accounts)} accounts, {len(fact)} transactions")


# ===============================================================
# STEP 4: RISK DETECTION
# ===============================================================
def detect_risk(conn, cust, txn):
    # --- transaction-level rules ---
    flags = [(t, "HIGH_VALUE", 3) for t in txn.loc[txn["amount"] >= HIGH_VALUE_LIMIT, "txn_id"]]
    for _, g in txn.groupby("account_no"):
        s = g.sort_values("txn_time").set_index("txn_time")
        cnt = s["amount"].rolling("1h").count()
        flags += [(t, "VELOCITY", 2) for t in s.loc[cnt >= VELOCITY_COUNT, "txn_id"]]
    flags += [(t, "ODD_HOUR", 1) for t in txn.loc[txn["txn_time"].dt.hour < ODD_HOUR_END, "txn_id"]]
    conn.executemany("INSERT INTO risk_flag (txn_id, rule, severity) VALUES (?, ?, ?)", flags)

    # --- customer-profile rules (from the Kaggle data) ---
    cflags = [(c, "CREDIT_DEFAULT", 3) for c in cust.loc[cust["credit_default"] == 1, "customer_id"]]
    cflags += [(c, "NEGATIVE_BALANCE", 2) for c in cust.loc[cust["avg_balance"] < 0, "customer_id"]]
    both = (cust["housing_loan"] == 1) & (cust["personal_loan"] == 1)
    cflags += [(c, "MULTIPLE_LOANS", 1) for c in cust.loc[both, "customer_id"]]
    conn.executemany("INSERT INTO customer_risk_flag (customer_id, rule, severity) VALUES (?, ?, ?)", cflags)
    conn.commit()

    print(f"[4] Risk     : {len(flags)} transaction flags {pd.Series([f[1] for f in flags]).value_counts().to_dict()}")
    print(f"               {len(cflags)} customer flags    {pd.Series([f[1] for f in cflags]).value_counts().to_dict()}")


def main():
    cust_raw = extract_customers()
    if not all(os.path.exists(f) for f in (RAW_BRANCH, RAW_ACCOUNT, RAW_TXN)):
        generate_source_systems(cust_raw)

    branches = pd.read_csv(RAW_BRANCH)
    accounts = pd.read_csv(RAW_ACCOUNT)
    txn_raw = pd.read_csv(RAW_TXN)
    print(f"[1] Extract  : {len(branches)} branches, {len(accounts)} accounts, {len(txn_raw)} transactions")

    cust = transform_customers(cust_raw)
    accounts = accounts[accounts["customer_id"].isin(cust["customer_id"])]
    txn = transform_transactions(txn_raw, set(accounts["account_no"]))

    conn = sqlite3.connect(DB_FILE)
    load(conn, branches, cust, accounts, txn)
    detect_risk(conn, cust, txn)

    print("\nSample check 1 - risk by branch (transaction flags):")
    print(pd.read_sql("""
        SELECT b.branch_name, COUNT(*) AS flags, SUM(r.severity) AS risk_score
        FROM risk_flag r JOIN transactions t ON t.txn_id = r.txn_id
        JOIN account a ON a.account_no = t.account_no
        JOIN branch b ON b.branch_code = a.branch_code
        GROUP BY b.branch_name ORDER BY risk_score DESC""", conn).to_string(index=False))

    print("\nSample check 2 - top 5 riskiest customers:")
    print(pd.read_sql("""SELECT customer_id, occupation, avg_balance, txn_risk, profile_risk, total_risk
                         FROM v_customer_risk ORDER BY total_risk DESC LIMIT 5""", conn).to_string(index=False))
    conn.close()
    print(f"\nDone. Warehouse saved as {DB_FILE}")


if __name__ == "__main__":
    main()
