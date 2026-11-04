import csv
import json
import math
from .statistics import bootstrap_interval, wilson
from .telemetry import histogram_deltas


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summarize(run, records):
    rows = [r for r in records if r["phase"] == "steady"]
    ok = [r for r in rows if r["status"] == "success"]
    duration = sum(s["ended_at"] - s["started_at"] for s in run["segments"] if s["ended_at"])
    complete_window = all(s["ended_at"] is not None for s in run["segments"])
    tokens = sum(r["output_tokens"] for r in ok) if ok and all(
        r.get("output_tokens") is not None for r in ok) else None
    from .config import Plan
    plan = Plan(**run["plan"])
    estimate = plan.estimate()
    elapsed = (run.get("stopped_at", run.get("deadline", run["created_at"])) -
               run.get("started_at", run["created_at"])) if run.get("started_at") else 0
    cost = max(0, elapsed) * estimate["resource_usd_s"]
    if run["shutdown"] == "unverified":
        cost = None
    latency = [r["latency_s"] for r in ok]
    err = (len(rows) - len(ok)) / len(rows) if rows else None
    p95 = percentile(latency, .95)
    quality = [r["quality_pass"] for r in ok if r.get("quality_pass") is not None]
    good = [r for r in ok if r.get("latency_s", float("inf")) <= plan.target_p95_s
            and r.get("ttft_s") is not None and r["ttft_s"] <= plan.target_ttft_s
            and (r.get("output_tokens") == 1 or (r.get("tpot_s") is not None and r["tpot_s"] <= plan.target_tpot_s))
            and r.get("length_conformant") is not False]
    errors = len(rows) - len(ok)
    error_interval = wilson(errors, len(rows))
    lags = [r["schedule_lag_s"] for r in rows if r.get("schedule_lag_s") is not None]
    lag_p95 = percentile(lags, .95)
    categories = {}
    for record in rows:
        if record.get("quality_category"):
            categories.setdefault(record["quality_category"], {}).setdefault(record.get("case_id"), []).append(
                bool(record.get("quality_pass")))
    category_scores = {category: {"unique_cases": len(cases),
                                 "case_mean_accuracy": sum(sum(v)/len(v) for v in cases.values())/len(cases)}
                       for category, cases in categories.items()}
    controlled = plan.workload == "tokens"
    conformity = all(r.get("length_conformant") is True for r in ok) if controlled and ok else None
    quality_cases = {r.get("case_id") for r in ok if r.get("quality_category")}
    expected_cases = {r["id"] for r in run.get("manifest", {}).get("records", []) if r.get("quality_case")}
    reasons = []
    from .datasets import validate_manifest
    try:
        validate_manifest(run.get("manifest", {}))
        manifest_valid = bool(run.get("manifest"))
    except ValueError:
        manifest_valid = False
    if not manifest_valid:
        reasons.append("prepared workload integrity missing or invalid")
    if (run.get("approval") or {}).get("fingerprint") != plan.fingerprint():
        reasons.append("current plan does not match recorded approval")
    if plan.provider != "modal":
        reasons.append("simulated evidence")
    if run.get("status") != "complete" or run.get("shutdown") != "verified":
        reasons.append("run or cleanup incomplete")
    if not run.get("server_identity", {}).get("verified"):
        reasons.append("remote server identity not verified")
    if len(ok) < plan.min_capacity_samples:
        reasons.append("insufficient successful samples")
    if duration < 60 or not complete_window:
        reasons.append("less than 60s of complete steady-state observation")
    if not controlled or not conformity:
        reasons.append("token-length control absent or mismatched")
    if p95 is None or p95 > plan.target_p95_s:
        reasons.append("latency target not met")
    for field, target, name in [("ttft_s", plan.target_ttft_s, "first-content"), ("tpot_s", plan.target_tpot_s, "time-per-output-token")]:
        values = [r[field] for r in ok if r.get(field) is not None]
        observed = percentile(values, .95)
        if len(values) != len(ok) or observed is None or observed > target:
            reasons.append(f"{name} p95 target not met or samples missing")
    if error_interval is None or error_interval[1] > plan.target_error_rate:
        reasons.append("error-rate upper bound exceeds target")
    if lag_p95 is None or lag_p95 > max(.01, .1/plan.rate):
        reasons.append("client scheduling lag too high or unknown")
    if len(run["segments"]) != 1:
        reasons.append("resumed or absent observation window")
    if len(rows) != plan.requests:
        reasons.append("not all requests offered")
    billing = run.get("billing", {})
    actual_cost = billing.get("total_usd")
    return {
        "manifest_integrity_verified": manifest_valid,
        "plan_fingerprint": plan.fingerprint(),
        "goodput_rps": len(good)/duration if duration and complete_window else None,
        "good_requests": len(good), "error_rate_wilson_95": error_interval,
        "latency_p95_bootstrap_95_s": run.get("analysis", {}).get("latency_p95_bootstrap_95_s"),
        "schedule_lag_p95_s": lag_p95,
        "client_persistence_total_s": sum(r.get("persistence_s", 0) for r in rows),
        "length_conformant": conformity,
        "input_token_distribution": sorted({r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None}),
        "output_token_distribution": sorted({r["output_tokens"] for r in ok if r.get("output_tokens") is not None}),
        "capacity_evidence_eligible": not reasons, "capacity_exclusions": reasons,
        "quality_categories": category_scores, "quality_unique_cases": len(quality_cases),
        "quality_coverage": len(quality_cases)/len(expected_cases) if expected_cases else None,
        "server_measurements": histogram_deltas(run.get("telemetry", [])),
        "billing_verification": billing.get("verification", "unavailable"),
        "reconciled_cost_per_success_usd": actual_cost/len(ok) if actual_cost is not None and ok else None,
        "reconciled_cost_per_million_tokens_usd": actual_cost/tokens*1e6 if actual_cost is not None and tokens else None,
        "reference_status": run.get("reference", {}).get("status", "not_run"),
        "evidence": "SIMULATED — not GPU evidence" if plan.provider == "simulated" else "remote_client_measurements",
        "offered": len(rows), "successful": len(ok), "planned": plan.requests,
        "unattempted": plan.requests - len(rows), "error_rate_including_client_rejections": err,
        "sample_count": len(ok), "window_s": duration if complete_window else None,
        "latency_p50_s": percentile(latency, .5), "latency_p95_s": p95,
        "latency_p99_s": percentile(latency, .99),
        "ttft_p95_s": percentile([r["ttft_s"] for r in ok if r.get("ttft_s") is not None], .95),
        "tpot_p95_s": percentile([r["tpot_s"] for r in ok if r.get("tpot_s") is not None], .95),
        "successful_rps": len(ok) / duration if duration and complete_window else None,
        "output_tokens_s": tokens / duration if duration and tokens is not None and complete_window else None,
        "offered_rate_target": plan.rate, "output_tokens": tokens,
        "within_targets": bool(p95 is not None and p95 <= plan.target_p95_s and err <= plan.target_error_rate
                               and len(rows) == plan.requests and complete_window and "benchmark" in run["stages"]),
        "resource_cost_estimate_usd": cost, "provider_reported_cost_usd": actual_cost,
        "cost_per_success_estimate_usd": cost / len(ok) if ok and cost is not None else None,
        "cost_per_million_output_tokens_estimate_usd": cost / tokens * 1e6 if tokens and cost is not None else None,
        "quality_exact_match": sum(quality) / len(quality) if quality else None,
        "quality_sample_count": len(quality),
        "limitations": ["Client TTFT is first nonempty content chunk, not exact server token time.",
                       "SSE chunks may contain multiple tokens: individual token gaps are unavailable.",
                       "TPOT uses first-to-last content chunk divided by output_tokens-1.",
                       "Percentiles interpolate observed samples; small samples do not establish tail SLOs.",
                       "Capacity applies only to this workload, network path, load and bounded client.",
                       "Cost is whole-run resource time, excludes non-compute charges; billing remains unknown.",
                       "Readiness includes provisioning and loading; model-only loading is unavailable.",
                       "Device utilization/memory come only from NVIDIA samples; absent samples remain unavailable.",
                       "Quality suite is a versioned task regression set, not a broad model-capability benchmark.",
                       "Bootstrap intervals assume independent request samples; use repeated runs for temporal variation."],
    }


def export(store, run):
    rows = store.requests(run["id"])
    latency = [r["latency_s"] for r in rows if r["phase"] == "steady" and r["status"] == "success"]
    run["analysis"] = {"latency_p95_bootstrap_95_s": bootstrap_interval(latency)}
    store.save(run)
    summary = summarize(run, rows)
    if run.get("reference", {}).get("metrics"):
        from .reference import crosscheck
        summary["reference_crosscheck"] = crosscheck(summary, run["reference"]["metrics"])
    folder = store.root / run["id"]
    folder.mkdir(exist_ok=True)
    payload = {"run": run, "summary": summary, "requests": rows, "events": store.events(run["id"])}
    (folder / "report.json").write_text(json.dumps(payload, indent=2))
    fields = ["id", "phase", "status", "error", "case_id", "quality_category", "input_tokens_target",
              "output_tokens_target", "length_conformant", "schedule_lag_s", "prompt_chars", "output_chars", "prompt_tokens", "output_tokens",
              "ttft_s", "tpot_s", "latency_s", "quality_pass"]
    with (folder / "requests.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (folder / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    text = (f"# Inference run {run['id']}\n\n{summary['evidence']}\n\n"
            f"Status: {run['status']}. Shutdown: {run['shutdown']}.\n\n"
            f"Hypothesis: {run['plan']['hypothesis']}\n\n"
            f"```json\n{json.dumps(summary, indent=2)}\n```\n\n"
            "## Next steps\n\n"
            "If shutdown is unverified, use teardown and inspect the Modal workspace. "
            "Preserve incomplete requests; resume never replays recorded attempts. "
            "Do not infer credit exhaustion or OOM from a generic network/provider error. "
            "Consult provider logs locally for a supported diagnosis.\n")
    (folder / "report.md").write_text(text)
    return folder


def compare(a, b):
    from .config import Plan
    pa, pb = Plan(**a["run"]["plan"]).to_dict(), Plan(**b["run"]["plan"]).to_dict()
    allowed = {"hypothesis", "baseline_id", "repetition"}
    changes = {k: (pa[k], pb[k]) for k in pa
               if pa[k] != pb[k] and k not in allowed}
    caveats = []
    for field, description in [("source_tree_sha256", "source code"), ("dependency_lock_sha256", "dependency lock")]:
        left = a["run"].get("provenance", {}).get(field)
        right = b["run"].get("provenance", {}).get(field)
        if not left or not right:
            caveats.append(f"Comparable {description} provenance is unavailable")
        elif left != right:
            caveats.append(f"{description.title()} changed between runs")
    if a['run'].get('environment') != b['run'].get('environment'):
        caveats.append("Recorded client environment changed")
    return {"comparability_caveats": caveats, "controlled_changes": changes, "single_variable": len(changes) == 1,
            "baseline": a["summary"], "candidate": b["summary"],
            "warning": "No improvement claim without repetitions, comparable environment and quality review."}
