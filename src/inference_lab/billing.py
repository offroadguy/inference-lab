"""Import a resource-attributed statement; never imply independent provider verification."""
import hashlib
import json
import math
from pathlib import Path
import time


def reconcile(store, run, path):
    raw = Path(path).read_bytes()
    bill = json.loads(raw)
    if run["plan"]["provider"] != "modal" or not run.get("resource_id"):
        raise ValueError("Billing reconciliation requires a real Modal resource")
    if bill.get("run_id") != run["id"] or bill.get("resource_id") != run["resource_id"]:
        raise ValueError("Statement must identify this exact run and resource")
    amount = bill.get("total_usd")
    if bill.get("currency") != "USD" or type(amount) not in {int, float} or not math.isfinite(amount) or amount < 0:
        raise ValueError("Statement needs nonnegative finite total_usd and currency USD")
    if bill.get("source") not in {"modal_export", "modal_invoice"}:
        raise ValueError("Statement source must be modal_export or modal_invoice")
    for key in ("period_start", "period_end"):
        if type(bill.get(key)) not in {int, float} or not math.isfinite(bill[key]):
            raise ValueError("Billing period uses finite Unix timestamps")
    if not run.get("stopped_at") or bill["period_start"] > run["started_at"] or bill["period_end"] < run["stopped_at"]:
        raise ValueError("Statement period must cover the complete stopped run")
    if bill.get("allocation") != "exclusive_resource":
        raise ValueError("Account-wide bills cannot be treated as exact run cost")
    run["billing"] = {"total_usd": amount, "currency": "USD", "source": bill["source"],
                      "source_sha256": hashlib.sha256(raw).hexdigest(), "imported_at": time.time(),
                      "verification": "user_supplied_provider_statement_not_independently_verified",
                      "period_start": bill["period_start"], "period_end": bill["period_end"]}
    directory = store.root / run["id"]
    directory.mkdir(exist_ok=True)
    (directory / "billing-evidence.json").write_bytes(raw)
    store.save(run)
    return run["billing"]
