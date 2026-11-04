import asyncio
import json
import time
import httpx
from prometheus_client.parser import text_string_to_metric_families
from .workloads import workload, quality_score


def error_code(exc):
    # Never persist exception strings: URLs, credentials and provider bodies may contain secrets.
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    return type(exc).__name__


async def measure(client, endpoint, plan, prompt, request_id, store, run_id, phase="steady"):
    start = time.perf_counter()
    record = {"id": request_id, "phase": phase, "status": "running", "started_at": time.time(),
              "prompt_chars": len(prompt), "output_chars": 0, "prompt_tokens": None,
              "output_tokens": None, "chunk_times_s": [], "inter_chunk_gaps_s": [],
              "individual_inter_token_gaps_s": None, "ttft_s": None, "tpot_s": None,
              "latency_s": None, "error": None, "output": "", "quality_pass": None}
    store.request(run_id, record)
    body = {"model": plan.model, "messages": [{"role": "user", "content": prompt}],
            "stream": True, "stream_options": {"include_usage": True}, "max_tokens": plan.max_tokens,
            "temperature": plan.temperature, "seed": plan.seed}
    done = False
    try:
        async with asyncio.timeout(plan.request_timeout_s):
            async with client.stream("POST", endpoint.url + "/v1/chat/completions",
                                     headers=endpoint.headers, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    data = json.loads(payload)
                    if "error" in data:
                        raise RuntimeError("stream_error")
                    usage = data.get("usage")
                    if usage:
                        record["prompt_tokens"] = usage.get("prompt_tokens")
                        record["output_tokens"] = usage.get("completion_tokens")
                    for choice in data.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if content:
                            record["chunk_times_s"].append(time.perf_counter() - start)
                            record["output"] += content
                    store.request(run_id, record)
        if not done:
            raise ValueError("incomplete_stream")
        record["status"] = "success"
    except asyncio.CancelledError as exc:
        record.update(status="incomplete", error=error_code(exc))
        raise
    except Exception as exc:
        record.update(status="error", error=error_code(exc))
    finally:
        record["ended_at"] = time.time()
        record["latency_s"] = time.perf_counter() - start
        times = record["chunk_times_s"]
        record["output_chars"] = len(record["output"])
        if times:
            record["ttft_s"] = times[0]
            record["inter_chunk_gaps_s"] = [b - a for a, b in zip(times, times[1:])]
            tokens = record["output_tokens"]
            if tokens and tokens > 1:
                record["tpot_s"] = (times[-1] - times[0]) / (tokens - 1)
        if record["status"] == "success":
            record["quality_pass"] = quality_score(prompt, record["output"])
        store.request(run_id, record)
    return record


async def health(client, endpoint, deadline):
    last = "not_ready"
    while time.time() < deadline:
        try:
            response = await client.get(endpoint.url + "/health", headers=endpoint.headers, timeout=3)
            response.raise_for_status()
            return
        except httpx.HTTPError as exc:
            last = error_code(exc)
            await asyncio.sleep(1)
    raise TimeoutError(f"readiness_{last}")


async def telemetry(client, endpoint, run, store):
    sample = {"at": time.time(), "source": "server_prometheus", "values": {}}
    try:
        response = await client.get(endpoint.url + "/metrics", headers=endpoint.headers, timeout=3)
        response.raise_for_status()
        allowed = {"vllm:num_requests_running", "vllm:num_requests_waiting",
                   "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"}
        for family in text_string_to_metric_families(response.text):
            for metric in family.samples:
                if metric.name in allowed:
                    sample["values"][metric.name] = metric.value
        sample["status"] = "available" if sample["values"] else "unavailable"
    except Exception as exc:
        sample.update(status="unavailable", error=error_code(exc))
    # No NVIDIA metrics are synthesized from KV-cache occupancy.
    sample.update(gpu_memory_bytes=None, gpu_utilization=None)
    run["telemetry"].append(sample)
    store.save(run)


async def execute(run, plan, store, endpoint):
    async with httpx.AsyncClient(timeout=plan.request_timeout_s, limits=httpx.Limits(
        max_connections=plan.concurrency + 2, max_keepalive_connections=plan.concurrency + 2
    )) as client:
        ready_start = time.perf_counter()
        await health(client, endpoint, min(time.time() + plan.loading_s, run["deadline"] - 30))
        if "validated" not in run["stages"]:
            run["readiness_wait_s"] = time.perf_counter() - ready_start
            run["cold_start_to_ready_s"] = time.time() - run["started_at"]
            run["model_loading_s"] = None
            run["stages"].append("validated")
            store.save(run)
        previous = {r["id"]: r for r in store.requests(run["id"])}
        if "warmup" not in run["stages"]:
            async with asyncio.timeout(min(plan.warmup_s, max(0.01, run["deadline"] - time.time() - 30))):
                for i in range(plan.warmup_requests):
                    rid = f"warmup-{i:05}"
                    if rid in previous:
                        if previous[rid]["status"] != "success":
                            raise ValueError("Warm-up previously failed; start a fresh run")
                        continue
                    r = await measure(client, endpoint, plan, workload(plan.workload, i), rid,
                                      store, run["id"], "warmup")
                    if r["status"] != "success":
                        raise ValueError("Warm-up failed; inspect request evidence")
            run["stages"].append("warmup")
            store.save(run)
        await telemetry(client, endpoint, run, store)
        active = set()
        remaining = [i for i in range(plan.requests) if f"steady-{i:05}" not in previous]
        segment = {"started_at": time.time(), "ended_at": None, "offered": 0}
        run["segments"].append(segment)
        store.save(run)
        async def sample_loop():
            while True:
                await asyncio.sleep(5)
                await telemetry(client, endpoint, run, store)

        sampler = asyncio.create_task(sample_loop())
        start = time.perf_counter()
        budget = min(plan.benchmark_s, run["deadline"] - time.time() - 30)
        try:
            async with asyncio.timeout(max(0.01, budget)):
                for offset, index in enumerate(remaining):
                    await asyncio.sleep(max(0, start + offset / plan.rate - time.perf_counter()))
                    active = {task for task in active if not task.done()}
                    rid = f"steady-{index:05}"
                    segment["offered"] += 1
                    if len(active) >= plan.concurrency:
                        store.request(run["id"], {"id": rid, "phase": "steady", "status": "rejected",
                                                  "error": "client_concurrency_limit", "started_at": time.time()})
                    else:
                        active.add(asyncio.create_task(measure(client, endpoint, plan,
                            workload(plan.workload, index), rid, store, run["id"])))
                if active:
                    await asyncio.gather(*active)
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
            for task in active:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            segment["ended_at"] = time.time()
            store.save(run)
            await telemetry(client, endpoint, run, store)
        run["stages"].append("benchmark")
        store.save(run)
