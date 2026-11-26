from pathlib import Path
import pandas as pd

REQUIRED = {"invoice_id", "customer", "region", "invoice_date", "revenue", "outstanding_balance"}

def load_transactions(path):
    df = pd.read_csv(path, parse_dates=["invoice_date"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for col in ["revenue", "outstanding_balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def revenue_by_month(df, month):
    period = pd.Period(month, freq="M")
    return round(float(df.loc[df.invoice_date.dt.to_period("M") == period, "revenue"].sum()), 2)

def top_outstanding_customer(df):
    totals = df.groupby("customer", as_index=False).outstanding_balance.sum()
    if totals.empty:
        return {"customer": None, "outstanding_balance": 0.0}
    row = totals.sort_values("outstanding_balance", ascending=False).iloc[0]
    return {"customer": str(row.customer), "outstanding_balance": round(float(row.outstanding_balance), 2)}

def revenue_by_region(df):
    return {str(k): round(float(v), 2) for k, v in df.groupby("region").revenue.sum().sort_values(ascending=False).items()}
