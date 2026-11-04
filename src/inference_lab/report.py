import csv
import json
import math


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
    return {
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
        "resource_cost_estimate_usd": cost, "provider_reported_cost_usd": None,
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
                       "GPU utilization and allocated memory unavailable; KV occupancy is a different metric.",
                       "Versioned four-item quality set is an exact-match smoke test, not a quality benchmark."],
    }


def export(store, run):
    rows = store.requests(run["id"])
    summary = summarize(run, rows)
    folder = store.root / run["id"]
    folder.mkdir(exist_ok=True)
    payload = {"run": run, "summary": summary, "requests": rows, "events": store.events(run["id"])}
    (folder / "report.json").write_text(json.dumps(payload, indent=2))
    fields = ["id", "phase", "status", "error", "prompt_chars", "output_chars", "prompt_tokens", "output_tokens",
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
    allowed = {"hypothesis", "baseline_id", "repetition"}
    changes = {k: (a["run"]["plan"][k], b["run"]["plan"][k]) for k in a["run"]["plan"]
               if a["run"]["plan"][k] != b["run"]["plan"][k] and k not in allowed}
    return {"controlled_changes": changes, "single_variable": len(changes) == 1,
            "baseline": a["summary"], "candidate": b["summary"],
            "warning": "No improvement claim without repetitions, comparable environment and quality review."}
