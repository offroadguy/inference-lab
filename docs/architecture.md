# Architecture decisions

## Local control, remote execution

SQLite is the durable journal; a per-run file lock prevents two CLI processes from owning the same run. The CLI persists deployment intent before a remote create call and saves the resource ID immediately afterward. The dashboard reads state, never approves spending. Each streamed response updates SQLite, allowing interruption recovery without waiting for a batch to finish.

The Modal adapter uses a **named GPU sandbox**, rather than an indefinitely deployed autoscaling service. This supports an absolute provider-enforced lifetime, fixed concurrency/resource envelope and lookup by name after an ambiguous create RPC. There are no public raw ports: requests use short-lived Sandbox Connect credentials held in process memory. Endpoint URLs and bearer credentials are not saved in reports. SDK exceptions are classified without logging raw provider messages.

A stopped resource is not replaced automatically. The user creates a new immutable spending plan. A failed lookup is not proof that provisioning failed; the CLI attempts cleanup and labels unverifiable shutdown. Remote lifetime remains the last-resort protection after a local crash. If image creation stalls before the sandbox starts, the provider's build lifecycle can exceed the local deadline; build costs are uncertain and require checking the provider account.

## Model/hardware compatibility

The profile chooses a small Qwen2 causal language model, an architecture supported by vLLM, FP16 on an L4, no quantization and a 4096-token context. A fixed model/tokenizer revision prevents silent upstream weight changes. Validation rejects changes to this profile until another profile is explicitly implemented and tested. This is **static compatibility validation**, not proof of successful deployment: CUDA, driver/image compatibility, GPU availability and runtime memory must still be validated by a real cloud run.

Approximate budget (not measured allocation): 0.49 billion parameters × 2 bytes ≈ 0.91 GiB of weights. With 24 layers, 2 KV heads, head dimension 64, FP16 keys and values, KV storage is approximately `2 × 24 × 2 × 64 × 2 = 12,288 bytes/token`; 4096 tokens is about 48 MiB per sequence, or 1.5 GiB for 32 full sequences. Framework, temporary tensors, activations and allocator overhead add memory. vLLM is configured to use up to 80% of the L4's nominal 24 GiB. Eager execution simplifies the first deployment at the expense of potential throughput. Validate the assumptions against the pinned model config and actual server metrics before any larger model.

## Cost reasoning

The plan contains the rate source/date, resources, loading/warm-up/benchmark/idle budgets, absolute runtime, maximum authorized spend and allowance. Compute rate is:

`L4 USD/s + 2 × sandbox CPU USD/core/s + 16 × sandbox memory USD/GiB/s`

Upper planning estimate is `rate × runtime × 1.25 + other-charge allowance`. It must fit the authorized amount. This gate is a conservative planning check, **not provider-enforced dollar accounting**. The run-time cost report uses elapsed lifetime and configured resource rates, excluding unknown non-compute charges. With uncertain shutdown, cost is unavailable rather than deceptively precise.

Sandbox CPU/memory rates were checked on 2026-11-03. Modal lists GPU rates separately; the checked L4 rate is $0.000222/second. Re-verify prices before approval. Credit balance/expiry, actual usage billing and image-build charges are not fetched. No claim is made that free credits cover a run. No persistent volume is created, trading repeated download time for a simpler cleanup story.

## Sources

- [Modal sandbox lifecycle and resource API](https://modal.com/docs/reference/modal.Sandbox)
- [Sandbox authenticated HTTP access](https://modal.com/docs/guide/sandbox-networking)
- [Modal pricing](https://modal.com/pricing)
- [Pinned Qwen model configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json)
- [vLLM 0.11.0 supported models](https://docs.vllm.ai/en/v0.11.0/models/supported_models.html)
