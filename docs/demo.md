# Demonstrations and interview walkthrough

## Five-minute local demo

1. **0:00–0:45** — Explain the architecture and the boundary between local control and remote serving. State that the saved fixture is simulated.
2. **0:45–1:30** — Show `plan` with resources, cost assumptions, credit unknowns and exact typed approval. Do not approve or launch a cloud run during a free demo.
3. **1:30–2:30** — Run `uv run python -m inference_lab.demo` or open `examples/simulated-demo.json` as the saved-results fallback. Show raw streaming samples, latency distributions and errors in the dashboard.
4. **2:30–3:30** — Compare the synthetic concurrency experiment. Show what changed and distinguish software behavior from GPU optimization evidence.
5. **3:30–4:30** — Inspect the controlled 503 experiment and subsequent healthy run. Explain durable requests, named resource reconciliation, partial results and unverified shutdown.
6. **4:30–5:00** — State limitations and the next paid validation step. No real GPU capacity or cost savings have been measured yet.

## 1. Performance improvement protocol — real run pending

Establish three fresh baseline runs of the Modal chat or summary profile. Inspect error rate, TTFT/TPOT, throughput and available queue/KV metrics. Form a bottleneck hypothesis supported by those observations. Change **one** factor (for example, concurrency 1→2 under identical offered load, or prefix caching for explicitly reused prefixes). Repeat three times. Compare reports:

```sh
uv run inference-lab compare data/BASELINE/report.json data/CANDIDATE/report.json
```

Report p50/p95/p99, token throughput, sample counts, cost estimate and unavailable GPU memory honestly. Run the identical quality workload before/after if the change can affect answers. A failed or inconclusive improvement is valid evidence when explained. The local concurrency fixture demonstrates the analysis pipeline only.

## 2. Capacity and cost protocol — real run pending

Hold workload, output length, caching and hardware fixed. Create fresh approved plans at 1, 2, 4 and 8 offered requests/sec, with enough bounded concurrency and runtime to avoid accidental client bottlenecks. Repeat each point and find the highest **tested** rate meeting latency/error targets. If client rejection dominates, describe the client bound rather than calling it a GPU capacity limit.

Use the measured successful RPS in the headroom formula in methodology.md. Display measured versus extrapolated values separately. Provider cost remains unknown until reconciled with account billing. Do not use synthetic fixture throughput for hardware sizing.

## 3. Failure investigation — local controlled demo verified

The demo's fourth run uses an explicit HTTP 503 fixture; its fifth restores healthy behavior. Inspect `data/simulated-demo.json` for symptoms, cause, recovery and prevention, and individual run reports for request-level evidence. The tests also cover a stalled response, interrupted stream preservation, missing approval, ambiguous creation guard and cleanup failure.

This is a **simulated failure/recovery investigation**, not a production incident. GPU OOM, quota errors, credit exhaustion and provider outages have not been deliberately triggered or tested against a real account. They produce bounded local failure handling; establishing their cause requires provider evidence.

## Explain it in your own words

Practice one question at a time, attempting an answer before reading implementation details:

- Why is a provider lifetime stronger than local cleanup alone?
- How does a pending deployment intent prevent duplicate paid work?
- What does TTFT include when measured by this client?
- Why are stream chunks not necessarily tokens?
- Which denominator should successful requests/sec use?
- How do weights and KV-cache memory scale differently?
- What evidence would support an OOM diagnosis?
- What would make the comparison confounded?

## Evidence-grounded resume drafts (review after personally running/explaining)

- Built, with AI coding assistance, a Python inference experiment harness with SQLite checkpoints, streamed request measurements, a local dashboard, and explicit spending approval.
- Implemented and locally tested interrupted-stream preservation, bounded request timeouts, cleanup uncertainty reporting and controlled HTTP-failure recovery.

After real cloud validation, replace implementation-only wording with actual observed model/hardware/results. Do not claim vLLM authorship, measured GPU optimization, production ownership or cost reductions without evidence. These are drafts; no resume was edited.
