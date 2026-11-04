import asyncio
import json
import time
import random
import httpx
from .telemetry import parse_metrics
from .evaluation import score_case
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


async def measure(client, endpoint, plan, prompt, request_id, store, run_id, phase="steady", spec=None, scheduled_at=None):
    start = time.perf_counter()
    record = {"id": request_id, "phase": phase, "status": "running", "started_at": time.time(),
              "prompt_chars": len(prompt) if isinstance(prompt, str) else None, "output_chars": 0, "prompt_tokens": None,
              "output_tokens": None, "chunk_times_s": [], "inter_chunk_gaps_s": [],
              "individual_inter_token_gaps_s": None, "ttft_s": None, "tpot_s": None,
              "latency_s": None, "error": None, "output": "", "quality_pass": None}
    spec = spec or {}
    record.update(case_id=spec.get("id"), quality_category=spec.get("quality_case", {}).get("category"),
                  input_tokens_target=spec.get("input_tokens_target"),
                  output_tokens_target=spec.get("output_tokens_target", plan.max_tokens),
                  token_count_source="synthetic_fixture" if plan.provider == "simulated" else "server_usage",
                  schedule_lag_s=max(0, time.time() - scheduled_at) if scheduled_at else 0,
                  persistence_s=0.0, length_conformant=None)
    store.request(run_id, record)
    body = {"model": plan.model, "messages": [{"role": "user", "content": prompt}],
            "stream": True, "stream_options": {"include_usage": True}, "max_tokens": plan.max_tokens,
            "temperature": plan.temperature, "seed": plan.seed}
    completion = spec.get("endpoint") == "completions"
    body["max_tokens"] = spec.get("output_tokens_target", plan.max_tokens)
    if completion:
        body.pop("messages")
        body.update(prompt=prompt, ignore_eos=True, min_tokens=body["max_tokens"])
    done = False
    try:
        async with asyncio.timeout(plan.request_timeout_s):
            async with client.stream("POST", endpoint.url + ("/v1/completions" if completion else "/v1/chat/completions"),
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
                        content = choice.get("text") if completion else choice.get("delta", {}).get("content")
                        if content:
                            record["chunk_times_s"].append(time.perf_counter() - start)
                            record["output"] += content
                    write_start = time.perf_counter()
                    store.request(run_id, record)
                    record["persistence_s"] += time.perf_counter() - write_start
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
            case = spec.get("quality_case")
            if case:
                result = score_case(case, record["output"])
                record.update(quality_pass=result["pass"], quality_reason=result["reason"])
            elif isinstance(prompt, str):
                record["quality_pass"] = quality_score(prompt, record["output"])
            if completion:
                record["length_conformant"] = (record["prompt_tokens"] == spec["input_tokens_target"] and
                                              record["output_tokens"] == spec["output_tokens_target"])
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


async def telemetry(client, endpoint, run, store, provider=None):
    sample = {"at": time.time(), "source": "server_prometheus", "values": {}}
    try:
        response = await client.get(endpoint.url + "/metrics", headers=endpoint.headers, timeout=3)
        response.raise_for_status()
        sample["series"] = parse_metrics(response.text)
        for series in sample["series"]:
            if not series["name"].endswith(("_bucket", "_count", "_sum", "_created", "_total")):
                label = series["name"] + (" [engine=" + series["labels"]["engine"] + "]" if "engine" in series["labels"] else "")
                sample["values"][label] = series["value"]
        sample["status"] = "available" if sample["values"] else "unavailable"
    except Exception as exc:
        sample.update(status="unavailable", error=error_code(exc))
    sample.update(gpu_memory_bytes=None, gpu_utilization=None)
    if provider is not None and hasattr(provider, "gpu_telemetry"):
        try:
            async with asyncio.timeout(8):
                sample["devices"] = await provider.gpu_telemetry()
            if sample["devices"]:
                sample["gpu_memory_bytes"] = sum(d["memory_used_bytes"] for d in sample["devices"])
                sample["gpu_utilization"] = sum(d["utilization_percent"] for d in sample["devices"]) / len(sample["devices"])
        except Exception as exc:
            sample["gpu_error"] = error_code(exc)
    run["telemetry"].append(sample)
    store.save(run)


async def execute(run, plan, store, endpoint, provider=None):
    async with httpx.AsyncClient(timeout=plan.request_timeout_s, limits=httpx.Limits(
        max_connections=plan.concurrency + 2, max_keepalive_connections=plan.concurrency + 2
    )) as client:
        ready_start = time.perf_counter()
        await health(client, endpoint, min(time.time() + plan.loading_s, run["deadline"] - 30))
        if plan.provider == "modal":
            version_response = await client.get(endpoint.url + "/version", headers=endpoint.headers, timeout=5)
            version_response.raise_for_status()
            model_response = await client.get(endpoint.url + "/v1/models", headers=endpoint.headers, timeout=5)
            model_response.raise_for_status()
            version = version_response.json().get("version")
            model_ids = [m["id"] for m in model_response.json().get("data", [])]
            verified = version == "0.11.0" and plan.model in model_ids
            run["server_identity"] = {"version": version, "models": model_ids, "verified": verified}
            store.save(run)
            if not verified:
                raise ValueError("Server version or model identity does not match the approved plan")
        if "validated" not in run["stages"]:
            run["readiness_wait_s"] = time.perf_counter() - ready_start
            run["cold_start_to_ready_s"] = time.time() - run["started_at"]
            run["model_loading_s"] = None
            run["stages"].append("validated")
            store.save(run)
        manifest = run.get("manifest", {})
        specs = manifest.get("records", [])
        def request_spec(i):
            return specs[i] if specs else {"prompt": workload(plan.workload, i)}
        previous = {r["id"]: r for r in store.requests(run["id"])}
        if "warmup" not in run["stages"]:
            async with asyncio.timeout(min(plan.warmup_s, max(0.01, run["deadline"] - time.time() - 30))):
                for i in range(plan.warmup_requests):
                    rid = f"warmup-{i:05}"
                    if rid in previous:
                        if previous[rid]["status"] != "success":
                            raise ValueError("Warm-up previously failed; start a fresh run")
                        continue
                    r = await measure(client, endpoint, plan, request_spec(i % plan.requests)["prompt"], rid,
                                      store, run["id"], "warmup", request_spec(i % plan.requests))
                    if r["status"] != "success":
                        raise ValueError("Warm-up failed; inspect request evidence")
            run["stages"].append("warmup")
            store.save(run)
        await telemetry(client, endpoint, run, store, provider)
        active = set()
        remaining = [i for i in range(plan.requests) if f"steady-{i:05}" not in previous]
        segment = {"started_at": time.time(), "ended_at": None, "offered": 0}
        run["segments"].append(segment)
        store.save(run)
        async def sample_loop():
            while True:
                await asyncio.sleep(5)
                await telemetry(client, endpoint, run, store, provider)

        sampler = asyncio.create_task(sample_loop())
        start = time.perf_counter()
        wall_start = time.time()
        rng = random.Random(plan.seed + len(run["segments"]) - 1)
        arrival_offset = 0.0
        budget = min(plan.benchmark_s, run["deadline"] - time.time() - 30)
        try:
            async with asyncio.timeout(max(0.01, budget)):
                for offset, index in enumerate(remaining):
                    if offset:
                        arrival_offset += rng.expovariate(plan.rate) if plan.arrival == "poisson" else 1 / plan.rate
                    await asyncio.sleep(max(0, start + arrival_offset - time.perf_counter()))
                    active = {task for task in active if not task.done()}
                    rid = f"steady-{index:05}"
                    segment["offered"] += 1
                    if len(active) >= plan.concurrency:
                        store.request(run["id"], {"id": rid, "phase": "steady", "status": "rejected",
                                                  "error": "client_concurrency_limit", "started_at": time.time()})
                    else:
                        active.add(asyncio.create_task(measure(client, endpoint, plan,
                            request_spec(index)["prompt"], rid, store, run["id"], spec=request_spec(index),
                            scheduled_at=wall_start + arrival_offset)))
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
            await telemetry(client, endpoint, run, store, provider)
        run["stages"].append("benchmark")
        store.save(run)
