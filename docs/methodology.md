# Measurement and experiment discipline

## Workloads and scheduling

`chat` rotates short inference-engineering prompts. `summary` repeats a versioned, synthetic engineering passage into a longer document. `quality` uses four exact-match smoke-test items. Prompt characters and server-reported token counts are stored separately. Default generation is deterministic-temperature sampling (`temperature=0`, seed 42), bounded output, and a request-wide timeout. This does not guarantee bit-for-bit identical GPU results.

The client offers requests at regular intervals (not a Poisson process). Each scheduled arrival either starts immediately or is rejected at the client concurrency bound; there is no unbounded client queue. This exposes overload without hiding latency in a growing client queue. Client rejections are included in the error fraction and are distinct from server HTTP rejection. This workload is useful for reproducible experiments but is not a claim to replicate all production traffic. Local scheduling, SQLite writes and network latency can limit the load generator.

Warm-up is stored separately and excluded from steady-state aggregates. Prefix caching is explicitly disabled in the baseline. Repeated prompts can benefit from cache reuse when testing the prefix-cache setting; document this rather than claiming general gains. Repetitions are separate runs. Record baseline ID, hypothesis and repetition in each JSON plan. Avoid simultaneously changing hardware, workload, output budget, concurrency and engine parameters.

## Timing and throughput

A monotonic clock measures POST-to-first-nonempty-content-chunk latency and POST-to-stream-completion latency. Content chunk timestamps produce individual **inter-chunk** gaps. Exact inter-token gaps cannot be inferred from SSE content chunks and remain unavailable. Average TPOT is `(last content time - first content time)/(reported output tokens - 1)` for more than one token; buffering can distort it. HTTP chunk delivery and TLS are part of client measurements.

End-to-end percentiles use linear interpolation over successful steady-state requests. The report includes successes, errors, unattempted work and sample size; failed-request latencies stay in raw data but not successful-request latency percentiles. Small samples make p99 descriptive, not evidence of a tail guarantee. Run repeated larger samples before making capacity claims.

Throughput is successful completions (or successful output tokens) divided by the sum of complete benchmark segment durations, including response drain. A crash with an unclosed segment makes throughput unavailable. Resume does not replay previously recorded attempts; incomplete requests remain incomplete. A resumed run is not interchangeable with a fresh continuous-load run.

Queue/KV telemetry is sampled at stage boundaries and approximately every five seconds during load where `/metrics` exposes recognized vLLM metrics. Missing metrics remain unavailable. Current code does not collect device utilization or allocated memory; KV-cache occupancy is not either of those. Readiness time combines provisioning, image/model startup and network polling. No isolated model-loading duration is claimed.

## Capacity, cost and quality

A single run meets its configured targets only if all requests were offered, benchmark finished, p95 successful latency is within target and the total error/rejection fraction is within target. Report the highest **tested** load passing this check, with repetitions and sample sizes. This is a measured operating point, not proof of maximum hardware capacity.

For planning only, if a verified deployment sustains `r` successful requests/sec, a target `R` requests/sec at 30% headroom needs approximately `ceil(R/(0.7*r))` equivalent replicas. Multiply by hourly resource cost. This extrapolation assumes comparable workload and linear scaling; it omits network, routing, interference and discount effects. Do not extrapolate from the simulator.

`compare` lists all changed configuration fields and flags multi-variable comparisons. It does not prove statistical significance. For changes affecting outputs, run the same four-item quality set before/after and review saved outputs. Exact matching is brittle and too small to establish broad model quality; it is a regression smoke test only.

## Failure evidence

Timeout, HTTP status and exception type are recorded without raw exception bodies. A 503 does not prove OOM, a network failure does not prove credit exhaustion, and a local interruption does not prove remote shutdown. Use private provider logs/account evidence to diagnose those cases; do not publish raw logs without review. Current automatic reports preserve facts but intentionally do not invent provider root causes.
