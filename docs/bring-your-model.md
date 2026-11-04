# Reproducible model experiments

## Start from a clone

Install Python 3.12 and uv, then follow the README's `git clone`, `uv sync --frozen`, test, simulator and Streamlit commands. The simulator requires no Modal account, Hugging Face token or GPU. Dependency installation requires network access. A fresh dashboard has no experiments; its Overview emphasizes remote evidence, so select Experiments → Simulated to inspect the free demo.

`data/` is ignored. Your plans, model metadata, prompts, responses, local database and reports remain on your machine. Set `LAB_DATA` consistently for CLI and dashboard when using another directory. Never commit `.env` or tokens. Public reports must be reviewed before sharing, especially for private models or custom datasets.

## Inspect a model without starting compute

```sh
uv run inference-lab onboard HuggingFaceTB/SmolLM2-135M-Instruct --output data/my-model.json
uv run inference-lab preflight --config data/my-model.json
uv run inference-lab plan --config data/my-model.json
```

`onboard` reads Hugging Face metadata, resolves the requested revision to a commit SHA, and writes an editable configuration containing a hashed metadata snapshot. It downloads configuration JSON, not model weights, and never executes repository Python. `preflight` validates the configuration and prints the exact serving command, analytical memory budget, plan fingerprint and cost assumptions. It does not check your Modal account or guarantee GPU availability. `plan` downloads the pinned tokenizer and prepares the hashed request manifest. None of these commands starts a GPU.

Inspect the generated JSON before approval: it contains example resource/time limits, not a recommendation for your model. Verify current Modal rates and update `pricing_checked`; prices older than seven days are rejected for cloud plans. Larger models may need different CPU/RAM/GPU count. `onboard --gpu A100-80GB --gpu-count 2` selects explicit hardware; no automatic fallback changes the experiment.

Supported initial scope: dense `LlamaForCausalLM`, `MistralForCausalLM`, `Qwen2ForCausalLM`, and `Qwen3ForCausalLM`, safetensors parameter metadata below 100B, and `tokenizer.json`. Architecture checks are necessary but not sufficient for successful serving. No MoE, multimodal, quantized-weight or custom-Python onboarding. Check the model license yourself. Missing or unsupported metadata fails before spending.

For gated/private models, obtain access yourself. Set `HF_TOKEN` locally for metadata/tokenizer downloads and supply `--hf-secret-name YOUR_MODAL_SECRET`; that existing Modal Secret must contain `HF_TOKEN` for remote weight downloads. Only the secret name is persisted, never its value. This private-model path is implemented and unit-tested; it has not been validated against a private remote deployment.

## Explicit paid execution

Configure your own Modal credentials (`uv run modal token new` or local environment credentials). Review the emitted run ID and plan, then separately run:

```sh
uv run inference-lab approve RUN_ID
uv run inference-lab run RUN_ID
uv run inference-lab report RUN_ID
```

Approval binds the complete plan and workload hash. Changing model revision, hardware or an engine control requires a fresh plan and approval. Sandbox lifetime limits and teardown bound execution; estimates are not hard billing caps. Existing historical approvals may need new plans when the plan schema changes.

## Serving experiments

Editable JSON controls include `dtype` (`half` or `bfloat16`), `gpu_memory_utilization` (0.5–0.9), `enforce_eager`, `max_num_batched_tokens` (512–32768 or null), `chunked_prefill` (true/false/null), prefix caching, context (512–32768 within model/memory limits), and sequence count. Null preserves the pinned engine's default; it is not an experimentally measured value. Memory arithmetic uses 16-bit weights/KV plus a coarse overhead allowance and conservative full-context occupancy; actual peak memory requires a GPU run. CUDA graphs can add memory beyond this allowance.

The engine remains pinned to vLLM 0.11.0. Its container uses a version tag, not an immutable registry digest; this is a remaining reproducibility limitation. Quantization, FP8 KV cache, speculative decoding, alternate engines and arbitrary server flags remain unsupported.

## Repetitions and load sweeps

After a smoke test, edit workload sizes, request count, observation budget, offered rate, concurrency and targets for your intended use case. Do not describe a low-rate smoke test as maximum throughput.

```sh
uv run inference-lab sweep-plan --profiles custom --rates 1,2,4 --repetitions 3 --config data/my-model.json --max-spend-usd 10
```

This only prepares a campaign and may reject it if conservative aggregate costs exceed the supplied budget. Choose a justified budget yourself. Inspect the child plans before `sweep-approve CAMPAIGN_ID` and `sweep-run CAMPAIGN_ID`. The custom sweep keeps the onboarded model/hardware and varies rates and repetition indices. To compare engine settings, prepare separate configurations and change one setting at a time.

## What counts as evidence

Every newly prepared run records hashes of the local Python source files, plan and workload. These detect changes; they are not signatures or independent attestations. Reports preserve failed and incomplete requests. Capacity qualification requires intact workload evidence, an approved matching plan, verified remote identity/cleanup, sufficient samples, a complete observation window, token conformity, acceptable scheduling lag, and configured latency/error checks. At least three qualifying repetitions are required for a campaign operating point. Neither a qualifying point nor a clean confidence interval proves maximum capacity or production reliability.

Quality workloads and upstream benchmark cross-checks remain separate steps. Synthetic outputs never establish model quality. Client first-content timing is not server TTFT; SSE chunk gaps are not individual-token gaps. Costs remain estimates until a user-supplied billing statement is attached; that import is not independent billing verification.

## Honest self-critique

The useful direction is reproducible experiment orchestration over serving infrastructure. Modal already hosts models; another hosting wrapper alone offers little engineering differentiation. The project's strongest features are traceable configurations, bounded spending, failure retention, and explicit evidence limits.

Remaining gaps: validate a newly onboarded family on GPU; repeat representative workload/load sweeps; correlate client and server timing; run model quality checks; use immutable image digests; measure client saturation and telemetry overhead; and test gated-model access end-to-end. Model onboarding is CLI-based, not a dashboard form. This is an engineering portfolio and experimental harness, not a production certification system or universal benchmark suite.
