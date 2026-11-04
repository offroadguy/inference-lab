"""Prepare and hash workloads before approving cloud spending."""
import hashlib
import json
import os
from pathlib import Path
import random
from .evaluation import VERSION, builtin_cases, validate_cases
from .workloads import workload

CORPUS_VERSION = "engineering-passages-v1"
PASSAGES = [
    "A serving system batches compatible requests. Queueing can improve throughput but increases latency. "
    "Measure admission rejection separately from server failure. Preserve request outcomes and timeouts.",
    "Transformer inference stores keys and values for previous tokens. Memory depends on context length, "
    "batch size, layers and KV heads. Weight precision and temporary allocations also affect the budget.",
    "A service has three replicas and a bounded queue. During a spike, the queue fills and arrivals are rejected. "
    "An operator compares timestamps, server metrics and client latency before attributing a bottleneck.",
    "An experiment changes one setting while keeping workload and output lengths fixed. Repetitions reveal "
    "variation. Quality checks prevent a faster but incorrect response from being called an improvement.",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(plan, tokenizer=None):
    rng = random.Random(plan.seed)
    records = []
    if plan.workload == "tokens":
        if tokenizer is None:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_pretrained(plan.model, revision=plan.revision, token=os.getenv("HF_TOKEN") or None)
        encoded = [tokenizer.encode(p, add_special_tokens=False).ids for p in PASSAGES]
        if not all(encoded):
            raise ValueError("Tokenizer produced an empty corpus")
        for i in range(plan.requests):
            n = plan.input_lengths[i % len(plan.input_lengths)]
            output = plan.output_lengths[i % len(plan.output_lengths)]
            ids = []
            while len(ids) < n:
                ids.extend(rng.choice(encoded))
            records.append({"id": f"case-{i:05}", "prompt": ids[:n], "input_tokens_target": n,
                            "output_tokens_target": output, "endpoint": "completions"})
        version = CORPUS_VERSION
    elif plan.workload == "quality":
        cases = (validate_cases([json.loads(line) for line in Path(plan.evaluation_file).read_text().splitlines()
                                 if line.strip()]) if plan.evaluation_file else builtin_cases())
        if plan.requests < len(cases):
            raise ValueError(f"Quality run needs at least {len(cases)} requests to cover the full evaluation set")
        for i in range(plan.requests):
            case = cases[i % len(cases)]
            records.append({"id": case["id"], "prompt": case["prompt"], "quality_case": case,
                            "output_tokens_target": plan.max_tokens, "endpoint": "chat"})
        version = "custom-quality-v1" if plan.evaluation_file else VERSION
    else:
        records = [{"id": f"case-{i:05}", "prompt": workload(plan.workload, i),
                    "output_tokens_target": plan.max_tokens, "endpoint": "chat"} for i in range(plan.requests)]
        version = "legacy-prompts-v1"
    manifest = {"version": version, "model": plan.model, "tokenizer_revision": plan.revision,
                "seed": plan.seed, "records": records}
    return {**manifest, "sha256": digest(manifest)}


def validate_manifest(manifest):
    if manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Prepared workload changed; create a fresh plan and approval")


def create_run(store, plan, manifest=None):
    plan.validate()
    manifest = manifest or prepare(plan)
    validate_manifest(manifest)
    if manifest.get("model") != plan.model or manifest.get("tokenizer_revision") != plan.revision or len(manifest.get("records", [])) != plan.requests:
        raise ValueError("Prepared workload does not match plan model, revision or request count")
    run = store.create(plan)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    run["provenance"] = {"source_files_sha256": sources, "source_tree_sha256": digest(sources),
                         "plan_fingerprint": plan.fingerprint(), "workload_sha256": manifest["sha256"]}
    from importlib.metadata import version
    run["provenance"]["client_versions"] = {name: version(name) for name in ["httpx", "modal", "tokenizers", "huggingface-hub"]}
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    run["provenance"]["dependency_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    run["manifest"] = manifest
    store.save(run)
    from .reference import prepare_reference
    prepare_reference(plan, run, store)
    return run
