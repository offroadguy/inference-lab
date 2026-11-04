# Inference Lab

A local-first inference engineering lab: choose one of four pinned model profiles, review and approve a spending plan, run streamed workloads on a remote Modal GPU, and inspect persistent results in a local Streamlit dashboard.

**Status:** the local simulator, four pinned tokenizer pipelines, measurement logic, approval guardrails and recovery tests are verified. The Modal/vLLM adapter is implemented and checked against the installed SDK, but **has not been GPU-deployed or benchmarked**. No real performance, capacity, cost savings or model-quality claims are made. See [verified results](docs/engineering-report.md).

## Multi-model evaluation

**Four configured checkpoints:** Qwen2.5-Instruct 0.5B, 3B, 7B and 32B. All are below 100B; none is claimed GPU-validated yet. The dashboard now distinguishes profile configuration, tokenizer preparation and verified remote smoke runs.

- Hashed token-length-controlled workloads, regular/Poisson arrivals and repeated load campaigns.
- Aggregate campaign spending approval, resumable child runs and explicit capacity evidence gates.
- Goodput, scheduling lag, confidence intervals and separate server/device telemetry.
- A 40-case versioned quality suite plus custom JSONL evaluation sets.
- Optional local upstream vLLM cross-check and resource-attributed billing reconciliation.

See the [complete multi-model workflow and validation boundaries](docs/multi-model-evaluation.md).

```sh
uv run inference-lab models
uv run inference-lab plan --profile qwen-3b --config examples/sweep-simulated.json
```

## Run the free local demo

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --frozen
uv run pytest -q
uv run python -m inference_lab.demo
uv run streamlit run src/inference_lab/dashboard.py --server.address=127.0.0.1
```

Open http://localhost:8501. The demo runs five **synthetic** experiments: baseline, a concurrency change, higher offered load, injected HTTP 503, and a healthy recovery run. It generates `data/simulated-demo.json` and a report directory per run. This tests the software; its timings and placeholder token counts are **not model or GPU evidence**.

For a single run:

```sh
uv run inference-lab plan --provider simulated --workload chat
uv run inference-lab run RUN_ID
uv run inference-lab report RUN_ID
```

Use the `run_id` printed by `plan`. State and raw measurements stay in ignored `data/`. `LAB_DATA` or `--data` changes the CLI directory; set `LAB_DATA` for the dashboard too.

### Docker Compose

```sh
docker compose up --build -d dashboard
docker compose run --rm --entrypoint python cli -m inference_lab.demo
# Interactive CLI example:
docker compose run --rm cli plan --provider simulated
```

Dashboard port binds to localhost. Only the CLI container reads `.env`. The Docker image build, containerized simulator and dashboard smoke check passed. Native Python was also verified.

## A cloud run (billable, requires your typed approval)

1. Copy `.env.example` to `.env` and set your Modal token locally, or use Modal's existing local credential store. Do not commit credentials. No Hugging Face token is required for the selected public model.
2. Copy `examples/modal-chat.json` to a local config. Verify current [Modal pricing](https://modal.com/pricing), update the rates and `pricing_checked` (maximum age seven days), and choose your limits. Credits and expiry remain unknown unless you independently verify them in your account. Sandbox CPU/memory pricing differs from Function pricing.
3. Review and approve:

```sh
uv run inference-lab plan --config examples/modal-chat.json
uv run inference-lab approve RUN_ID
# Type the exact phrase shown, including run ID and maximum USD amount.
uv run inference-lab run RUN_ID
```

`run` deploys, validates readiness, warms up, benchmarks, exports JSON/CSV/Markdown, and attempts teardown even on errors. Approval covers one exact plan and expires after one hour if deployment has not started. Any configuration change requires a new plan/approval. There is no automatic redeployment or paid retry.

Separate commands are also available:

```sh
uv run inference-lab deploy RUN_ID
uv run inference-lab benchmark RUN_ID
uv run inference-lab resume RUN_ID
uv run inference-lab teardown RUN_ID
uv run inference-lab list
```

**`deploy` leaves a billable sandbox running**, bounded by its configured lifetime. `resume` reconciles the named sandbox/ID and preserves recorded attempts; it never replays successful, failed or incomplete requests. If the sandbox has terminated, create a fresh plan. If remote creation was ambiguous, reconciliation must succeed before work can continue. Check the provider workspace if shutdown is unverified. A local process crash does not stop remote billing.

The estimate uses the selected profile’s GPU count, CPU cores and RAM, a 25% compute margin and a separate other-charge allowance. CPU/memory limits and a provider-enforced lifetime bound the resource envelope. It is **not an exact dollar billing cap**: image builds, network, taxes, fees, account rates and credits can differ. No persistent model volume or auto-restarting deployment is created. A sandbox remains billable while idle until terminated or timed out; idle time is included in the plan.

## What is measured

- First content chunk latency (client TTFT), streamed chunk timestamps/gaps, average time per output token, end-to-end latency, interpolated p50/p95/p99 and sample counts.
- Server-reported prompt/output token usage when supplied; unavailable stays null.
- Offered requests, successful requests/sec and output tokens/sec over measured windows, admission rejection, timeouts and errors.
- Readiness/cold-start envelope, separate warm-up requests, periodic available vLLM queue/KV-cache metrics.
- Whole-run resource-cost estimate, per-success and per-million-output-token estimates; provider billing is explicitly unknown.

SSE chunks are not necessarily tokens. Individual server token timestamps and isolated model-loading time remain unavailable. NVIDIA device samples and vLLM request-phase histograms are collected when exposed by a real approved run. The dashboard does not invent them. Raw JSONL preserves measurements; CSV contains request summaries. [Methodology](docs/methodology.md) explains limits and interpretation.

## Scope and architecture

```mermaid
flowchart LR
    CLI[Local Python CLI: plan and approval] --> DB[(Local SQLite)]
    CLI --> Adapter[Provider adapter]
    Adapter --> GPU[Modal bounded GPU sandbox: vLLM]
    Client[Local streaming benchmark client] -->|Authenticated HTTPS| GPU
    Client --> DB
    DB --> Dashboard[Local Streamlit dashboard]
    DB --> Files[JSONL / JSON / CSV / Markdown]
```

Supported profiles use Qwen2.5-Instruct 0.5B/3B/7B/32B at fixed Hugging Face commits, vLLM 0.11.0, FP16 and no quantization. L4 and A100 hardware envelopes are reviewed per profile; 32B uses two 80GB A100s with tensor parallelism. Unsupported combinations fail before provisioning. Local dependencies are locked in `uv.lock`; the server image is version-tagged. More families, arbitrary sub-100B models and Superlinked remain extensions requiring separate validation.

## Engineering and interview evidence

- [Architecture, memory and cost reasoning](docs/architecture.md)
- [Experiment and measurement discipline](docs/methodology.md)
- [Three demonstration protocols and five-minute walkthrough](docs/demo.md)
- [Verified engineering report, limitations and attribution](docs/engineering-report.md)

The implementation was authored with OpenAI Codex assistance for Venkata Ganji. Modal supplies remote compute/lifecycle, vLLM supplies inference, and the Qwen team supplies the model. Personal understanding, operation and real benchmark results must be demonstrated separately. No employer source, internal project material or credentials are included.
