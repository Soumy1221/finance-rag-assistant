from pathlib import Path
from rag.analytics import load_transactions, revenue_by_month, top_outstanding_customer, revenue_by_region

DATA = Path(__file__).resolve().parents[1] / "data" / "finance_transactions.csv"

def test_november_revenue():
    assert revenue_by_month(load_transactions(DATA), "2025-11") == 63200.0

def test_top_outstanding_customer():
    result = top_outstanding_customer(load_transactions(DATA))
    assert result["customer"] == "Blue Peak Ltd"
    assert result["outstanding_balance"] == 8100.0

def test_region_revenue():
    assert revenue_by_region(load_transactions(DATA))["North"] == 31900.0
