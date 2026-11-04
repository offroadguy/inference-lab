# Multi-model evaluation platform

## What is implemented, and what is validated

The platform now has four reviewed, pinned **Qwen2.5-Instruct checkpoints**: 0.5B, 3B, 7B and 32B. Keeping the first set within one architecture family makes size comparisons easier to interpret. These are four checkpoints, not four unrelated model families. They are all below 100B; the registry does **not** imply compatibility with every sub-100B model.

| Profile | GPU envelope | CPU / RAM | Initial evidence level |
|---|---|---|---|
| qwen-0.5b | 1 × L4 24GB | 2 cores / 16 GiB | Pinned tokenizer + simulated pipeline |
| qwen-3b | 1 × L4 24GB | 4 cores / 32 GiB | Pinned tokenizer + simulated pipeline |
| qwen-7b | 1 × A100 40GB | 4 cores / 48 GiB | Pinned tokenizer + simulated pipeline |
| qwen-32b | 2 × A100 80GB | 8 cores / 128 GiB | Pinned tokenizer + simulated pipeline |

**No model weights were loaded on GPUs during this implementation. No real performance or capacity is established.** Model revisions and layer/KV dimensions were retrieved from Hugging Face. Each tokenizer was actually downloaded and used to build controlled requests. The simulated HTTP fixture then exercised the full measurement/export path for each profile. Model-specific licenses still apply.

The registry validates exact model/revision/hardware/precision combinations and estimates FP16 weights + full-context KV cache + an overhead allowance. It rejects combinations outside the reviewed profiles. vLLM uses tensor parallelism equal to GPU count. The server sequence cap defaults to 32 and is separate from client concurrency, so changing the client admission bound does not silently change server batching capacity. An analytical fit is not runtime validation; fragmentation, kernels, drivers, activation memory and provider availability remain to be checked remotely. See `inference-lab models` for complete budgets and fixed revisions.

## Controlled token workloads

```sh
uv run inference-lab plan --profile qwen-3b --config examples/sweep-simulated.json
uv run inference-lab run RUN_ID
```

`workload=tokens` prepares token-ID arrays with the pinned tokenizer and sends them to `/v1/completions`. Input lengths cycle through `input_lengths`; output lengths cycle through `output_lengths`. A seeded assembly of four original engineering passages creates the inputs. Token IDs avoid hidden chat-template overhead or text retokenization length drift. This is a **controlled synthetic performance workload**, not a representative sample of all user traffic.

Output uses `min_tokens=max_tokens` and `ignore_eos=true`. Every successful response is checked against the server-reported prompt/completion counts; a mismatch is visible and disqualifies capacity evidence. Use natural `chat`/`summary` and dedicated `quality` runs to assess conversational behavior and answer quality—fixed-length generation is not a quality test.

The complete prepared manifest, its version, tokenizer revision and SHA-256 are persisted before approval. Approval binds both the plan and manifest. The manifest is verified before execution; changes require fresh approval. Raw token usage in simulation is explicitly marked `synthetic_fixture`.

## Repeated load sweeps

A free demonstration across all four profiles:

```sh
uv run inference-lab sweep-plan \
  --profiles qwen-0.5b,qwen-3b,qwen-7b,qwen-32b \
  --rates 10 --repetitions 2 \
  --config examples/sweep-simulated.json --max-spend-usd 1
uv run inference-lab sweep-approve CAMPAIGN_ID
uv run inference-lab sweep-run CAMPAIGN_ID
uv run inference-lab sweep-report CAMPAIGN_ID
```

The displayed estimated cloud cost for simulation is $0. Campaign approval is still exercised to demonstrate the flow. It is marked simulation-only.

For real evidence, use `examples/sweep-modal.json` after verifying rates. Start with **one small profile and one load point**, not all four at once. A larger reviewed campaign can use `--rates 1,2,4 --repetitions 3`. The example budgets are configuration limits, **not authorization**. The CLI prints every child plan and the aggregate estimate before requiring `APPROVE SWEEP <id> USD <limit>`.

Sweeps run sequentially. Each repetition gets a fresh bounded sandbox, its own approval-bound manifest and its own cold-start record. The campaign reserves the sum of upper estimates against the aggregate maximum. It stops on execution failure or unverifiable cleanup; rerunning resumes existing children and skips completed, verified ones. It never automatically replaces a stopped paid resource. Approval expires after 24 hours, and planned lifetimes cannot exceed 24 hours. Individual standalone approval still expires after one hour before starting.

Read `campaign-<id>-report.json` for per-repetition throughput and min/median/max. No pooling across different models/load settings is performed. A qualified operating point requires at least three qualifying repetitions. Qualification currently requires real Modal evidence, verified model/server identity and cleanup, full offered workload, controlled conformant lengths, ≥200 successful samples by default, ≥60 seconds of observation, acceptable scheduling lag, latency target, and an error-rate Wilson upper bound within target. This establishes a **tested operating point**, not maximum capacity or production readiness. Reference cross-checks and quality review remain separate required engineering judgments.

## Measurement confidence

- Request goodput counts only successful requests within configured end-to-end, TTFT and TPOT targets, excluding token-length mismatches.
- Error-rate Wilson 95% intervals prevent “0 errors in a tiny sample” from being treated as a guarantee.
- Reports compute a deterministic 400-resample bootstrap interval for p95 when ≥20 successful samples exist. It assumes independent samples and is not a substitute for repeated runs.
- Scheduled-arrival lag and time spent writing request state expose client-side overhead. Regular and seeded Poisson arrivals are supported.
- A deterministic streamed-response test checks known TTFT, end-to-end, chunk-gap and TPOT values independently of real-time fixture variability.
- SSE content chunks remain distinct from tokens. Exact individual server token timestamps are **not available**. vLLM's server inter-token histogram, when exposed, provides aggregate server-side observations without pretending client chunks are individual tokens.

## Server telemetry and identity

The Modal path verifies `/version` reports 0.11.0 and `/v1/models` includes the approved model. It records local Python/package/platform metadata. Every approximately five seconds, the collector retains recognized labelled Prometheus queue, KV-cache, TTFT, inter-token, prefill, decode, end-to-end, token-count and preemption series. Labels are restricted; raw credentials and provider exception bodies are not stored.

Server histogram count/sum deltas yield means over the snapshot interval. Counter resets are flagged. Raw buckets remain in the journal; they are not presented as exact per-request timings. The collector also executes a bounded `nvidia-smi` query inside the approved sandbox for device memory, utilization and driver identity. Missing or failed measurements remain unavailable. Model-only loading duration remains unavailable; cold-start-to-ready includes provisioning and startup.

## Upstream vLLM cross-check

Use `examples/modal-reference.json` with a small reviewed profile. It sets a separate `reference_s` budget, Poisson arrivals, one input length, one output length and no prefix caching. The normal run performs the lab benchmark, then launches **vLLM 0.11.0's upstream `bench serve` locally in Docker** against the same still-approved remote resource. Reference time is included in the runtime budget. The client cannot extend the sandbox lifetime.

Prepare the local reference image before cloud spending:

```sh
docker pull --platform linux/amd64 vllm/vllm-openai:v0.11.0
docker run --rm --platform linux/amd64 --entrypoint vllm \
  vllm/vllm-openai:v0.11.0 bench serve --help
uv run inference-lab plan --profile qwen-0.5b --config examples/modal-reference.json
uv run inference-lab approve RUN_ID
uv run inference-lab run RUN_ID
```

The image is large. Preflight uses `--pull=never` and fails before provisioning if the local reference client cannot run. The tokenizer is revision-pinned and prepared locally. Authentication is passed through an environment variable, never embedded in a generated command or report. The local benchmark container is removed after completion/interruption; uncertain client cleanup is reported.

Upstream JSON is retained with a hash and normalized milliseconds-to-seconds metrics. The dashboard shows relative discrepancies. **This is a cross-check, not an equivalence proof:** the upstream random-text workload and its bounded-concurrency scheduling differ from this lab's assembled token IDs and immediate client admission rejection. Review actual token counts, errors, sample sizes, sequential thermal/cache effects and load semantics before interpreting differences. The upstream Docker/GPU combination has not yet been executed end to end; local schema/command/units tests do not replace it.

## Quality evaluation

The built-in `infra-quality-v2` suite has 40 distinct cases across arithmetic, structured extraction, code reasoning and grounded reading. Exact-normalized and structural JSON scorers are supported; generated code is never executed. JSON booleans and numbers are distinct, extra keys fail, malformed JSON fails, and scoring results retain reasons. Category scores average within unique case IDs so repetitions do not masquerade as additional independent questions. Coverage and successful unique cases are reported separately.

```sh
uv run inference-lab plan --profile qwen-7b --config examples/modal-quality.json
```

The built-in suite requires at least 40 requests. Supply `evaluation_file` in the plan for a custom JSONL suite; every row needs `id`, `category`, `prompt`, `scorer` (`exact` or `json`) and `expected`. The contents are copied into the approval-bound manifest. Use non-confidential evaluation material. This regression suite is more substantial than four toy questions but is **not MMLU, HumanEval, a representative domain benchmark or a broad model ranking**. Human review and workload-relevant test sets remain necessary.

## Cost reconciliation

The estimate accounts for every GPU and profile-specific CPU/RAM. Billing imports require an exact stopped Modal run/resource, a USD amount, a covering billing period and explicit exclusive-resource attribution. Account-wide totals cannot silently become per-run cost.

```sh
uv run inference-lab reconcile-cost RUN_ID path/to/statement.json
```

Statement schema:

```json
{
  "run_id": "YOUR_RUN_ID", "resource_id": "EXACT_MODAL_SANDBOX_ID",
  "currency": "USD", "total_usd": 1.25,
  "source": "modal_export", "allocation": "exclusive_resource",
  "period_start": 0, "period_end": 1
}
```

Replace the illustrative amount and Unix timestamps with actual provider statement data. An import is labelled **user-supplied provider statement, not independently verified**. Its hash and original file are retained locally in ignored data/. Cost per successful request and per million output tokens are computed separately for estimated and reconciled cost. Credits, discounts and invoice provenance cannot be inferred from resource runtime.

## Sources and limitations

- [vLLM 0.11.0 reference benchmark](https://docs.vllm.ai/en/v0.11.0/cli/bench/serve.html)
- [vLLM 0.11.0 server metrics](https://docs.vllm.ai/en/v0.11.0/usage/metrics.html)
- [Modal resource pricing](https://modal.com/pricing), checked 2026-11-03
- [Qwen2.5 model collection](https://huggingface.co/Qwen)

Superlinked is still not an adapter in this implementation. This change expands the reviewed Modal model profiles and benchmark discipline. Real hardware validation, provider statement reconciliation and upstream cross-check execution require actual approved runs. The dashboard makes those pending steps visible instead of converting local test coverage into deployment claims.
