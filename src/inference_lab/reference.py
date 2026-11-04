"""Optional upstream vLLM benchmark in a local CPU client container, within approved lifetime."""
import asyncio
import json
import math
import os
import time
import subprocess
from .datasets import digest


def command(plan, endpoint, directory):
    if plan.workload != "tokens" or len(set(plan.input_lengths)) != 1 or len(set(plan.output_lengths)) != 1:
        raise ValueError("Reference cross-check needs one controlled input length and one output length")
    if plan.arrival != "poisson" or plan.prefix_caching:
        raise ValueError("Reference cross-check needs Poisson arrivals and prefix caching disabled")
    return ["docker", "run", "--rm", "--platform", "linux/amd64", "--name", "inference-reference-" + directory.parent.name,
            "-e", "OPENAI_API_KEY", "-v", f"{directory.resolve()}:/evidence", "--entrypoint", "vllm", plan.image,
            "bench", "serve", "--backend", "vllm", "--base-url", endpoint.url,
            "--endpoint", "/v1/completions", "--model", plan.model, "--tokenizer", "/evidence/tokenizer",
            "--dataset-name", "random", "--random-input-len", str(plan.input_lengths[0]),
            "--random-output-len", str(plan.output_lengths[0]), "--random-range-ratio", "1.0",
            "--num-prompts", str(plan.requests), "--request-rate", str(plan.rate),
            "--max-concurrency", str(plan.concurrency), "--seed", str(plan.seed), "--ignore-eos",
            "--save-result", "--save-detailed", "--result-dir", "/evidence", "--result-filename", "upstream.json",
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99"]


def normalize(raw):
    mapping = {"successful_rps": "request_throughput", "output_tokens_s": "output_throughput",
               "latency_p95_s": "p95_e2el_ms", "ttft_p95_s": "p95_ttft_ms", "tpot_p95_s": "p95_tpot_ms"}
    result = {}
    for target, key in mapping.items():
        value = raw.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid upstream measurement")
        result[target] = value / 1000 if value is not None and key.endswith("_ms") else value
    result["successful"] = raw.get("completed")
    return result


def crosscheck(summary, reference):
    differences = {}
    for key in ("successful_rps", "output_tokens_s", "latency_p95_s", "ttft_p95_s", "tpot_p95_s"):
        a, b = summary.get(key), reference.get(key)
        differences[key] = {"lab": a, "upstream": b,
                            "relative_difference": (a-b)/b if a is not None and b else None}
    return {"metrics": differences, "equivalence_claim": False,
            "limitations": ["Different prompt contents and client scheduling/admission semantics.",
                             "Upstream retokenizes random text; inspect actual input/output lengths.",
                             "Sequential runs on the same resource, not simultaneous paired requests.",
                             "Investigate discrepancies; matching numbers alone do not prove correctness."]}


async def run_reference(plan, run, store, endpoint):
    directory = store.root / run["id"] / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    # Tokenizer assets are prepared before the billable deployment by prepare_reference().
    if not (directory / "tokenizer" / "tokenizer.json").exists():
        raise ValueError("Reference tokenizer was not prepared before deployment")
    argv = command(plan, endpoint, directory)
    env = {**os.environ, "OPENAI_API_KEY": endpoint.headers.get("Authorization", "").removeprefix("Bearer ")}
    deadline = min(plan.reference_s, run["deadline"] - time.time() - 30)
    if deadline <= 0:
        raise TimeoutError("No approved lifetime remains for reference benchmark")
    process = await asyncio.create_subprocess_exec(*argv, env=env, stdout=asyncio.subprocess.DEVNULL,
                                                   stderr=asyncio.subprocess.DEVNULL)
    try:
        async with asyncio.timeout(deadline):
            code = await process.wait()
        if code != 0:
            raise RuntimeError("Upstream benchmark failed")
        raw = json.loads((directory / "upstream.json").read_text())
        run["reference"] = {"status": "completed", "engine_version": "vllm-0.11.0",
                            "source_sha256": digest(raw), "metrics": normalize(raw),
                            "scope": "local Docker client, sequential random-text cross-check"}
        run["stages"].append("reference")
        store.save(run)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        # Killing docker's CLI alone does not guarantee the reference client container stops.
        cleanup = await asyncio.create_subprocess_exec("docker", "rm", "-f", "inference-reference-" + directory.parent.name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        try:
            async with asyncio.timeout(10):
                await cleanup.wait()
        except TimeoutError:
            cleanup.kill()
            await cleanup.wait()
            run["reference_client_cleanup"] = "unverified"
            store.save(run)


def prepare_reference(plan, run, store):
    if not plan.reference_s:
        return
    from huggingface_hub import snapshot_download
    directory = store.root / run["id"] / "reference" / "tokenizer"
    snapshot_download(plan.model, revision=plan.revision, local_dir=directory,
                      allow_patterns=["tokenizer.json", "tokenizer_config.json", "config.json", "vocab.json",
                                      "merges.txt", "special_tokens_map.json"])


def preflight(plan):
    if not plan.reference_s:
        return
    try:
        result = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "--pull=never",
                                 "--entrypoint", "vllm", plan.image, "bench", "serve", "--help"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("Prepare the local Docker reference client before provisioning") from None
    if result.returncode:
        raise ValueError("Local upstream client unavailable; pull the pinned image and verify vllm bench serve --help")
