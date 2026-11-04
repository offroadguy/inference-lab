"""Retain labelled server series and compute deltas, never mix with client timing."""
import math
from prometheus_client.parser import text_string_to_metric_families

PREFIXES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc", "vllm:time_to_first_token_seconds", "vllm:inter_token_latency_seconds",
            "vllm:time_per_output_token_seconds", "vllm:request_queue_time_seconds",
            "vllm:request_prefill_time_seconds", "vllm:request_decode_time_seconds",
            "vllm:e2e_request_latency_seconds", "vllm:prompt_tokens", "vllm:generation_tokens",
            "vllm:num_preemptions", "vllm:request_success")


def parse_metrics(text):
    series = []
    for family in text_string_to_metric_families(text):
        for metric in family.samples:
            if any(metric.name == p or metric.name.startswith(p + "_") for p in PREFIXES):
                if math.isfinite(metric.value):
                    labels = {k: v for k, v in metric.labels.items()
                              if k in {"model_name", "engine", "le", "finished_reason"}}
                    series.append({"name": metric.name, "labels": labels, "value": metric.value})
    return series


def histogram_deltas(samples):
    available = [s for s in samples if s.get("series")]
    if len(available) < 2:
        return {"status": "unavailable", "histograms": {}}
    def totals(sample):
        out = {}
        for item in sample["series"]:
            name = item["name"]
            if name.endswith(("_count", "_sum")):
                out[name] = out.get(name, 0) + item["value"]
        return out
    before, after = totals(available[0]), totals(available[-1])
    hist = {}
    for key, value in after.items():
        if key.endswith("_count") and key in before:
            name = key[:-6]
            count = value - before[key]
            skey = name + "_sum"
            total = after.get(skey, 0) - before.get(skey, 0)
            hist[name] = {"count": count if count >= 0 else None,
                          "mean_s": total/count if count > 0 and total >= 0 else None,
                          "counter_reset": count < 0 or total < 0}
    return {"status": "available", "histograms": hist,
            "scope": "first-to-last server snapshot; includes concurrent traffic if endpoint is shared"}


def parse_gpu_csv(text):
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            raise ValueError("Unexpected NVIDIA telemetry columns")
        index, name, driver, used, total, utilization = parts
        rows.append({"index": int(index), "name": name, "driver": driver,
                     "memory_used_bytes": float(used) * 2**20, "memory_total_bytes": float(total) * 2**20,
                     "utilization_percent": float(utilization)})
    return rows
