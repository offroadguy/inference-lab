"""Sequential, aggregate-approved sweeps with durable child run IDs."""
import json
import math
from statistics import median
import time
import uuid

from .config import Plan
from .datasets import create_run, digest, validate_manifest
from .models import profile_config


def save(store, campaign):
    path = store.root / f"campaign-{campaign['id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(campaign, indent=2))
    temporary.replace(path)
    return path


def load(store, campaign_id):
    if len(campaign_id) != 12 or any(c not in "0123456789abcdef" for c in campaign_id):
        raise ValueError("Invalid campaign ID")
    return json.loads((store.root / f"campaign-{campaign_id}.json").read_text())


def fingerprint(store, campaign):
    for rid in campaign["run_ids"]:
        validate_manifest(store.get(rid)["manifest"])
    return digest({"max_spend_usd": campaign["max_spend_usd"], "runs": [
        {"id": rid, "plan": store.get(rid)["plan"], "manifest": store.get(rid)["manifest"]["sha256"]}
        for rid in campaign["run_ids"]]})


def create(store, profiles, rates, repetitions, config, max_spend):
    if not math.isfinite(max_spend) or max_spend <= 0:
        raise ValueError("Campaign maximum spend must be positive and finite")
    if not 2 <= repetitions <= 10 or not rates or len(rates) > 10 or len(profiles) > 4:
        raise ValueError("Sweep supports 2–10 repetitions, 1–10 load points, 1–4 profiles")
    if len(set(rates)) != len(rates) or len(set(profiles)) != len(profiles):
        raise ValueError("Profiles and rates must be unique")
    plans = []
    for key in profiles:
        for rate in rates:
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Rates must be positive and finite")
            for rep in range(1, repetitions + 1):
                values = {**config, **profile_config(key), "rate": rate, "repetition": rep}
                # Explicit profile controls resource envelope; config controls approved time budgets.
                plan = Plan(**values)
                plan.validate()
                plans.append(plan)
    upper = sum(p.estimate()["upper_usd"] for p in plans)
    lifetime = sum(p.runtime_s for p in plans)
    if upper > max_spend or lifetime > 86400:
        raise ValueError("Aggregate estimate exceeds campaign spend or 24-hour planned runtime limit")
    campaign = {"id": uuid.uuid4().hex[:12], "run_ids": [], "max_spend_usd": max_spend,
                "estimated_upper_usd": upper, "planned_runtime_s": lifetime,
                "approval": None, "status": "preparing", "created_at": time.time()}
    save(store, campaign)
    for plan in plans:
        run = create_run(store, plan)
        run["campaign_id"] = campaign["id"]
        store.save(run)
        campaign["run_ids"].append(run["id"])
        save(store, campaign)
    campaign["status"] = "planned"
    save(store, campaign)
    return campaign


def approve(store, campaign, confirmation):
    if campaign["status"] == "preparing":
        raise ValueError("Preparation incomplete; create a new campaign")
    phrase = f"APPROVE SWEEP {campaign['id']} USD {campaign['max_spend_usd']:.2f}"
    if confirmation != phrase:
        raise ValueError("Campaign spending not approved")
    campaign["approval"] = {"fingerprint": fingerprint(store, campaign), "approved_at": time.time(),
                            "expires_at": time.time() + 86400,
                            "kind": "simulation_only" if all(store.get(rid)["plan"]["provider"] == "simulated"
                                                               for rid in campaign["run_ids"]) else "typed_spending_approval"}
    save(store, campaign)


def execute_campaign(store, campaign):
    from .cli import launch
    approval = campaign.get("approval")
    if not approval or approval["fingerprint"] != fingerprint(store, campaign) or time.time() > approval["expires_at"]:
        raise ValueError("Campaign needs fresh exact-plan approval")
    campaign["status"] = "running"
    save(store, campaign)
    try:
        for rid in campaign["run_ids"]:
            with store.lock(rid):
                run = store.get(rid)
                if run["status"] == "complete" and run["shutdown"] == "verified":
                    continue
                if time.time() > approval["expires_at"]:
                    raise ValueError("Campaign approval expired; no new resources started")
                plan = Plan(**run["plan"])
                run["approval"] = {"fingerprint": plan.fingerprint(), "manifest_sha256": run["manifest"]["sha256"],
                                   "expires_at": approval["expires_at"], "approved_at": approval["approved_at"],
                                   "max_spend_usd": plan.max_spend_usd, "runtime_s": plan.runtime_s,
                                   "campaign_id": campaign["id"]}
                store.save(run)
                launch(store, run, "resume")
                if store.get(rid)["shutdown"] != "verified":
                    raise ValueError("Campaign stopped: cleanup unverified")
        campaign["status"] = "complete"
    except BaseException:
        campaign["status"] = "partial"
        raise
    finally:
        save(store, campaign)
        write_report(store, campaign)


def write_report(store, campaign):
    from .report import summarize
    groups = {}
    for rid in campaign["run_ids"]:
        run = store.get(rid)
        p = Plan(**run["plan"])
        summary = summarize(run, store.requests(rid))
        groups.setdefault((p.model, p.rate), []).append((run, summary))
    points = []
    for (model, rate), items in groups.items():
        successful_rps = [s["successful_rps"] for r, s in items if r["status"] == "complete"
                          and s["successful_rps"] is not None]
        all_qualified = all(s.get("capacity_evidence_eligible", False) for r, s in items)
        points.append({"model": model, "offered_rps": rate, "repetitions_planned": len(items),
                       "repetitions_completed": len(successful_rps), "rps_by_repetition": successful_rps,
                       "rps_median": median(successful_rps) if successful_rps else None,
                       "rps_min": min(successful_rps) if successful_rps else None,
                       "rps_max": max(successful_rps) if successful_rps else None,
                       "capacity_qualified": all_qualified and len(items) >= 3,
                       "run_ids": [r["id"] for r, _ in items]})
    capacity = {}
    for point in points:
        if point["capacity_qualified"]:
            capacity[point["model"]] = max(capacity.get(point["model"], 0), point["offered_rps"])
    report = {"campaign": campaign, "points": points, "highest_tested_qualified_offered_rps": capacity,
              "limitations": ["A tested operating point is not maximum hardware capacity.",
                               "Each repetition uses a fresh sandbox; cost includes repeated startup.",
                               "Simulation never qualifies as GPU capacity evidence.",
                               "At least three qualifying repetitions are required; ranges expose run-to-run variation."]}
    (store.root / f"campaign-{campaign['id']}-report.json").write_text(json.dumps(report, indent=2))
    return report
