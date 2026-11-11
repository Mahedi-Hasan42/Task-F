import csv
import os
from datetime import datetime
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "db"
STOCK_CSV = DB_DIR / "stock.csv"
LOG_CSV = DB_DIR / "scan_log.csv"

LOG_FIELDS = ["time", "source", "predicted", "confidence", "status", "final_label"]


SUCCESS_STATUS = "CONFIRMED"


def init_db():
    DB_DIR.mkdir(exist_ok=True)
    if not STOCK_CSV.exists():
        _write_stock({})
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_FIELDS)


def _read_stock():
    stock = {}
    with open(STOCK_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stock[row["product"]] = int(row["quantity"])
    return stock


def _write_stock(stock):
    tmp = STOCK_CSV.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["product", "quantity"])
        for product, qty in sorted(stock.items()):
            w.writerow([product, qty])
    os.replace(tmp, STOCK_CSV)


def add_stock(product, delta=1):
    stock = _read_stock()
    stock[product] = stock.get(product, 0) + delta
    _write_stock(stock)
    return stock[product]


def get_quantity(product):
    return _read_stock().get(product, 0)


def log_scan(source, predicted, confidence, status, final_label=""):
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            source, predicted, f"{confidence:.4f}", status, final_label,
        ])


def stats():
    stock = {k: v for k, v in _read_stock().items() if v > 0}
    rows = sorted(stock.items(), key=lambda kv: -kv[1])
    return rows, len(rows), sum(stock.values())


def log_summary():
    counts = {}
    with open(LOG_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["status"]] = counts.get(row["status"], 0) + 1
    total = sum(counts.values())
    ok = counts.get(SUCCESS_STATUS, 0)
    return {"total": total, "successful": ok, "failed": total - ok, "by_status": counts}