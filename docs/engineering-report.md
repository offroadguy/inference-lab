# Engineering report — 2026-11-03

## Verified scope

Implemented a local Python CLI, SQLite run/request journal, Streamlit dashboard, JSON/CSV/JSONL/Markdown exports, a bounded HTTP/SSE workload generator, explicit spending approval and a Modal GPU sandbox adapter. Versions are pinned in pyproject/uv.lock; model and tokenizer use a fixed Hugging Face revision.

**No GPU deployment was performed and no cloud money was spent by this implementation session.** The Modal adapter's interfaces were checked against installed SDK 1.6.3 and official documentation; this is not an end-to-end cloud validation. Both native Python and the built Docker image were checked locally.

## Checks actually executed

- `uv sync` resolved and installed the environment; the lockfile is committed.
- `uv run ruff check .` passed.
- `uv run pytest -q`: **24 passed**. Coverage includes unapproved spending never reaching provider creation, approval expiry/configuration changes, invalid profiles/budgets, ambiguous-create guard, cleanup failure redaction, stream truncation, cancellation/partial preservation, recovery without replay, percentile and throughput calculations, real local HTTP integration, controlled HTTP 503 and bounded timeout.
- Five synthetic experiments completed with verified local child-process shutdown.
- Streamlit's `AppTest` rendered the dashboard with the saved demo runs and reported **no exceptions**.
- `docker compose config --quiet` and `docker compose build dashboard` passed; the built image completed a three-request simulation and a dashboard AppTest without exceptions.
- All three supplied Modal JSON profiles passed local compatibility/budget validation. No credentials were read to perform cloud provisioning.

## Observed local fixture results

These counts come from the saved `examples/simulated-demo.json`. They measure local harness behavior, **not GPU/model capacity or quality**. Each experiment offered 24 steady-state requests. The fixture emits a short synthetic stream with intentional delay.

| Experiment | Offered rate | Client concurrency | Successful | Error/rejection fraction | Shutdown |
|---|---:|---:|---:|---:|---|
| Baseline | 12/s | 1 | 12/24 | 50% | Verified |
| Controlled concurrency change | 12/s | 2 | 24/24 | 0% | Verified |
| Higher load | 24/s | 1 | 8/24 | 66.7% | Verified |
| Injected HTTP 503 | 12/s | 1 | 0/24 | 100% | Verified |
| Healthy recovery | 12/s | 1 | 12/24 | 50% | Verified |

The one-variable synthetic concurrency change eliminated client admission rejection for this short fixture. That validates the experiment comparison mechanism; it does not establish a vLLM bottleneck or optimization. The healthy recovery run restored the baseline behavior, including its intentionally restrictive concurrency limit.

## Controlled incident investigation

**Symptoms:** all 24 requests in the failure run returned HTTP 503; the report preserved errors and verified simulator cleanup.

**Evidence:** the run configuration explicitly enabled `simulated_fault=http503`; request errors were classified as `http_503`. The subsequent healthy fixture restored 12 successful requests, matching the baseline's admission behavior.

**Cause:** deliberately injected HTTP response failure in the local fixture. There is no evidence of credit exhaustion, GPU OOM or provider outage.

**Recovery:** completed the fault run, terminated its local process, and started a fresh healthy run. No recorded request was replayed automatically.

**Prevention/containment:** bounded admission, timeouts, durable records, exact-plan approval, verified/unverified cleanup states and provider lifetime for future cloud runs. Provider failure scenarios remain to be validated.

## Limitations and next evidence

1. Complete one explicitly approved Modal deployment before calling the remote path verified. Check image/GPU compatibility, actual startup, authentication, stream usage, telemetry and teardown.
2. Performance and capacity demonstrations are protocols, not completed real experiments. Collect repeated baseline/candidate runs and inspect quality outputs.
3. GPU device memory/utilization and isolated model-load duration are unavailable. Exact token gaps cannot be derived from SSE chunks.
4. Costs are resource-time estimates; provider-reported billing/credit data is not integrated. Estimates are not hard dollar caps.
5. Container image uses a version tag rather than immutable digest. Local dependencies have a transitive lockfile; the upstream image controls server dependencies.
6. Fixed regular arrivals and tiny quality fixtures are learning tools, not full production load/quality benchmarks. SQLite per-chunk persistence can affect high-rate client measurements.
7. Simulator standalone teardown after losing its parent handle cannot verify shutdown; its own lifetime still bounds it. Use the simulator through `run`/the demo. Cloud teardown reconciles the durable sandbox name/ID.

## Ownership and assistance

This is an independent learning/portfolio implementation for Venkata Ganji, authored with OpenAI Codex assistance. The orchestration, approval flow, measurement logic, persistence, reports, tests and dashboard are project code. Modal supplies compute and sandbox lifecycle; vLLM supplies model serving; Qwen supplies model weights/tokenizer; Streamlit and Plotly supply presentation components. Real operation and personal technical understanding require separate demonstration. No employer materials, production incidents or confidential implementation details were used.
