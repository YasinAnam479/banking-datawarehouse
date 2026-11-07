"""
Bank Transaction Risk Dashboard (Streamlit)
Reads the warehouse built by bank_etl_v2.py  ->  bank_dw.db

Run:  streamlit run dashboard.py
Needs: pip install streamlit plotly pandas
"""
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

DB_FILE = "bank_dw.db"
HIGH_RISK_SCORE = 6   # customers with total_risk >= this are called "high risk"

st.set_page_config(page_title="Bank Risk Dashboard", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------
# Load data from the warehouse
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_FILE)
    txn = pd.read_sql("""
        SELECT t.txn_id, t.account_no, a.customer_id, b.branch_name, b.branch_city,
               tt.type_name AS txn_type, t.txn_time, t.amount, t.channel
        FROM transactions t
        JOIN account a           ON a.account_no  = t.account_no
        JOIN branch b            ON b.branch_code = a.branch_code
        JOIN transaction_type tt ON tt.type_id    = t.type_id
    """, conn, parse_dates=["txn_time"])
    flags = pd.read_sql("SELECT txn_id, rule, severity FROM risk_flag", conn)
    cust = pd.read_sql("SELECT * FROM v_customer_risk", conn)
    cflags = pd.read_sql("SELECT customer_id, rule, severity FROM customer_risk_flag", conn)
    has_ml = len(pd.read_sql("SELECT name FROM sqlite_master WHERE name = 'ml_anomaly'", conn)) > 0
    ml = pd.read_sql("SELECT * FROM ml_anomaly", conn) if has_ml else pd.DataFrame()
    ml_metrics = pd.read_sql("SELECT * FROM ml_metrics", conn) if has_ml else pd.DataFrame()
    conn.close()
    txn["txn_type"] = txn["txn_type"].replace({"Atm Withdrawal": "ATM Withdrawal", "Pos Purchase": "POS Purchase"})
    return txn, flags, cust, cflags, ml, ml_metrics


txn, flags, cust, cflags, ml, ml_metrics = load_data()

# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
st.sidebar.header("Filters")
all_branches = sorted(txn["branch_name"].unique())
sel_branches = st.sidebar.multiselect("Branch", all_branches, default=all_branches)

min_d, max_d = txn["txn_time"].dt.date.min(), txn["txn_time"].dt.date.max()
date_pick = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(date_pick, (tuple, list)) and len(date_pick) == 2:
    d_from, d_to = date_pick
else:
    d_from, d_to = min_d, max_d

f = txn[(txn["branch_name"].isin(sel_branches)) &
        (txn["txn_time"].dt.date >= d_from) & (txn["txn_time"].dt.date <= d_to)].copy()
ff = flags.merge(f, on="txn_id")            # flags that belong to the filtered transactions
st.sidebar.caption("Filters apply to the transaction views. Customer-profile views always show all customers.")

# ---------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------
st.title("🏦 Bank Transaction Risk Dashboard")
st.caption("Source: normalized data warehouse (SQLite) built by an ETL pipeline. "
           "Customer profiles from UCI Bank Marketing data; transactions are synthetic.")

flagged_ids = ff["txn_id"].nunique()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Transactions", f"{len(f):,}")
k2.metric("Total value", f"{f['amount'].sum():,.0f}")
k3.metric("Flagged transactions", f"{flagged_ids:,}", f"{(flagged_ids / max(len(f), 1)) * 100:.1f}% of total", delta_color="off")
k4.metric("High-value alerts", f"{(ff['rule'] == 'HIGH_VALUE').sum():,}")
k5.metric("High-risk customers", f"{(cust['total_risk'] >= HIGH_RISK_SCORE).sum():,}",
          f"score >= {HIGH_RISK_SCORE}", delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚨 Transaction Risk", "👤 Customer Risk", "🤖 ML Anomalies"])

# ---------------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------------
with tab1:
    c1, c2 = st.columns(2)

    daily = f.groupby(f["txn_time"].dt.date).size().rename("Transactions")
    daily_flag = ff.groupby(ff["txn_time"].dt.date)["txn_id"].nunique().rename("Flagged")
    d = pd.concat([daily, daily_flag], axis=1).fillna(0).reset_index().rename(columns={"txn_time": "Date"})
    fig = px.line(d, x="Date", y=["Transactions", "Flagged"], title="Daily transactions vs flagged")
    fig.update_layout(legend_title_text="", yaxis_title="Count")
    c1.plotly_chart(fig, width="stretch")

    monthly = f.groupby(f["txn_time"].dt.to_period("M").astype(str))["amount"].sum().reset_index()
    monthly.columns = ["Month", "Total value"]
    fig = px.bar(monthly, x="Month", y="Total value", title="Monthly transaction value")
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    by_type = f.groupby("txn_type").size().reset_index(name="Count").sort_values("Count", ascending=False)
    fig = px.bar(by_type, x="txn_type", y="Count", title="Transactions by type")
    fig.update_layout(xaxis_title="")
    c3.plotly_chart(fig, width="stretch")

    by_ch = f.groupby("channel").size().reset_index(name="Count")
    fig = px.pie(by_ch, names="channel", values="Count", title="Transactions by channel", hole=0.4)
    c4.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------
# TAB 2: Transaction risk
# ---------------------------------------------------------------
with tab2:
    c1, c2 = st.columns(2)

    by_branch = ff.groupby(["branch_name", "rule"]).size().reset_index(name="Flags")
    fig = px.bar(by_branch, x="branch_name", y="Flags", color="rule", title="Risk flags by branch",
                 color_discrete_map={"HIGH_VALUE": "#d62728", "VELOCITY": "#ff7f0e", "ODD_HOUR": "#1f77b4"})
    fig.update_layout(xaxis_title="", legend_title_text="Rule")
    c1.plotly_chart(fig, width="stretch")

    score = ff.groupby("branch_name")["severity"].sum().reset_index(name="Risk score").sort_values("Risk score")
    fig = px.bar(score, x="Risk score", y="branch_name", orientation="h", title="Branch risk score (sum of severity)")
    fig.update_layout(yaxis_title="")
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    hours = f["txn_time"].dt.hour.value_counts().sort_index().reset_index()
    hours.columns = ["Hour", "Transactions"]
    hours["Period"] = hours["Hour"].apply(lambda h: "Odd hours (00-04)" if h < 5 else "Normal hours")
    fig = px.bar(hours, x="Hour", y="Transactions", color="Period", title="Transactions by hour of day",
                 color_discrete_map={"Odd hours (00-04)": "#d62728", "Normal hours": "#4c78a8"})
    c3.plotly_chart(fig, width="stretch")

    rules = ff["rule"].value_counts().reset_index()
    rules.columns = ["Rule", "Flags"]
    fig = px.pie(rules, names="Rule", values="Flags", title="Share of flags by rule", hole=0.4)
    c4.plotly_chart(fig, width="stretch")

    st.subheader("Top 10 high-value transactions")
    hv = (ff[ff["rule"] == "HIGH_VALUE"].sort_values("amount", ascending=False).head(10)
          [["txn_id", "account_no", "branch_name", "txn_type", "txn_time", "amount", "channel"]])
    st.dataframe(hv, width="stretch", hide_index=True)

    st.subheader("Velocity alerts (accounts with rapid bursts of transactions)")
    vel = ff[ff["rule"] == "VELOCITY"]
    if len(vel):
        v = (vel.groupby(["account_no", "branch_name"])
                .agg(flagged_txns=("txn_id", "count"), first_seen=("txn_time", "min"), total_amount=("amount", "sum"))
                .reset_index().sort_values("flagged_txns", ascending=False).head(10))
        st.dataframe(v, width="stretch", hide_index=True)
    else:
        st.info("No velocity alerts for the selected filters.")

# ---------------------------------------------------------------
# TAB 3: Customer risk
# ---------------------------------------------------------------
with tab3:
    c1, c2 = st.columns(2)

    pr = cflags["rule"].value_counts().reset_index()
    pr.columns = ["Rule", "Customers"]
    fig = px.bar(pr, x="Rule", y="Customers", title="Customer-profile risk flags")
    fig.update_layout(xaxis_title="")
    c1.plotly_chart(fig, width="stretch")

    occ = cust.groupby("occupation")["total_risk"].mean().reset_index(name="Avg risk score").sort_values("Avg risk score")
    fig = px.bar(occ, x="Avg risk score", y="occupation", orientation="h", title="Average risk score by occupation")
    fig.update_layout(yaxis_title="")
    c2.plotly_chart(fig, width="stretch")

    dist = cust["total_risk"].value_counts().sort_index().reset_index()
    dist.columns = ["Total risk score", "Customers"]
    fig = px.bar(dist, x="Total risk score", y="Customers", title="How many customers at each risk score")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Top 15 riskiest customers")
    top = cust.sort_values("total_risk", ascending=False).head(15)
    st.dataframe(top, width="stretch", hide_index=True)

# ---------------------------------------------------------------
# TAB 4: ML anomalies (Isolation Forest)
# ---------------------------------------------------------------
with tab4:
    if ml.empty:
        st.info("ML results not found. Run  python ml_risk.py  once, then refresh this page.")
    else:
        m = f.merge(ml, on="txn_id")
        m["caught_by_rules"] = m["txn_id"].isin(set(flags["txn_id"]))
        an = m[m["is_anomaly"] == 1]
        new_an = an[~an["caught_by_rules"]]
        met = ml_metrics.iloc[0]

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Anomalies flagged by model", f"{len(an):,}")
        a2.metric("Also caught by rules", f"{int(an['caught_by_rules'].sum()):,}")
        a3.metric("New anomalies (no rule flagged)", f"{len(new_an):,}")
        a4.metric("Held-out F1 score", f"{met['f1']:.2f}",
                  f"precision {met['precision']:.2f} | recall {met['recall']:.2f}", delta_color="off")

        c1, c2 = st.columns(2)
        m["Status"] = m["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
        fig = px.histogram(m, x="anomaly_score", color="Status", nbins=50, log_y=True,
                           title="Anomaly score distribution (higher = more unusual)",
                           color_discrete_map={"Normal": "#4c78a8", "Anomaly": "#d62728"})
        c1.plotly_chart(fig, width="stretch")

        sample = pd.concat([m[m["is_anomaly"] == 0].sample(min(3000, int((m["is_anomaly"] == 0).sum())), random_state=1),
                            m[m["is_anomaly"] == 1]])
        fig = px.scatter(sample, x="txn_time", y="amount", color="Status", log_y=True, opacity=0.7,
                         title="Transactions over time (anomalies highlighted)",
                         color_discrete_map={"Normal": "#9ecae1", "Anomaly": "#d62728"})
        c2.plotly_chart(fig, width="stretch")

        st.subheader("Top 15 new anomalies found by the model (not flagged by any rule)")
        cols = ["txn_id", "account_no", "branch_name", "txn_type", "txn_time", "amount", "anomaly_score"]
        st.dataframe(new_an.sort_values("anomaly_score", ascending=False).head(15)[cols],
                     width="stretch", hide_index=True)
        st.caption(f"Model: {met['model']} | trained on {int(met['train_rows']):,} transactions, "
                   f"tested on {int(met['test_rows']):,} held-out transactions | contamination {met['contamination']}. "
                   "No true fraud labels exist, so the serious rule alerts (HIGH_VALUE, VELOCITY) are used as a proxy label.")

st.divider()
st.caption("Risk rules: HIGH_VALUE (amount >= 5,000, severity 3) | VELOCITY (5+ txns in 1 hour, severity 2) | "
           "ODD_HOUR (00:00-04:59, severity 1) | CREDIT_DEFAULT (3) | NEGATIVE_BALANCE (2) | MULTIPLE_LOANS (1)")
