from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
IMAGE = "vllm/vllm-openai:v0.11.0"


@dataclass(frozen=True)
class Plan:
    provider: str = "simulated"
    model: str = MODEL
    revision: str = REVISION
    image: str = IMAGE
    gpu: str = "L4"
    gpu_count: int = 1
    dtype: str = "half"
    quantization: str = "none"
    workload: str = "chat"
    requests: int = 30
    concurrency: int = 2
    rate: float = 2.0
    max_tokens: int = 128
    temperature: float = 0.0
    seed: int = 42
    request_timeout_s: float = 60.0
    loading_s: int = 420
    warmup_requests: int = 2
    warmup_s: int = 120
    benchmark_s: int = 180
    idle_s: int = 30
    runtime_s: int = 900
    max_spend_usd: float = 2.0
    gpu_usd_s: float = 0.000222
    cpu_usd_core_s: float = 0.00003942
    memory_usd_gib_s: float = 0.00000667
    pricing_checked: str = "2026-11-03"
    pricing_source: str = "https://modal.com/pricing"
    other_charge_allowance_usd: float = 0.25
    target_p95_s: float = 15.0
    target_error_rate: float = 0.05
    prefix_caching: bool = False
    hypothesis: str = "Establish the baseline"
    baseline_id: str | None = None
    repetition: int = 1
    simulated_fault: str = "none"

    def validate(self):
        if type(self.prefix_caching) is not bool:
            raise ValueError("prefix_caching must be a boolean")
        if (self.model, self.revision, self.image, self.gpu, self.gpu_count, self.dtype, self.quantization) != (
            MODEL, REVISION, IMAGE, "L4", 1, "half", "none"
        ):
            raise ValueError("Only the pinned Qwen 0.5B / vLLM 0.11.0 / 1x L4 FP16 profile is supported")
        if self.provider not in {"modal", "simulated"} or self.workload not in {"chat", "summary", "quality"}:
            raise ValueError("Unsupported provider or workload")
        for key in ("requests", "concurrency", "max_tokens", "loading_s", "warmup_s", "benchmark_s",
                    "idle_s", "runtime_s", "warmup_requests", "repetition", "seed", "gpu_count"):
            v = getattr(self, key)
            if type(v) is not int or v < (0 if key in {"warmup_requests", "seed"} else 1):
                raise ValueError(f"Invalid integer: {key}")
        for key in ("rate", "request_timeout_s", "max_spend_usd", "gpu_usd_s", "cpu_usd_core_s",
                    "memory_usd_gib_s", "other_charge_allowance_usd", "target_p95_s"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"Invalid positive number: {key}")
        if not 0 <= self.temperature <= 2 or not 0 <= self.target_error_rate <= 1:
            raise ValueError("Invalid temperature or error target")
        if self.concurrency > 32 or self.requests > 10000 or self.max_tokens > 1024:
            raise ValueError("Profile limits: concurrency <=32, requests <=10000, output <=1024")
        if not 120 <= self.runtime_s <= 3600:
            raise ValueError("Runtime must be 120–3600 seconds")
        if self.loading_s + self.warmup_s + self.benchmark_s + self.idle_s > self.runtime_s - 30:
            raise ValueError("Stage budgets must leave 30 seconds for cleanup")
        if self.requests / self.rate > self.benchmark_s:
            raise ValueError("Offered workload does not fit benchmark budget")
        if self.simulated_fault not in {"none", "http503", "timeout"}:
            raise ValueError("Unknown simulated fault")
        if self.provider == "modal" and self.simulated_fault != "none":
            raise ValueError("Fault injection is simulator-only")
        if self.provider == "modal":
            age = (date.today() - date.fromisoformat(self.pricing_checked)).days
            if age < 0 or age > 7:
                raise ValueError("Verify current provider rates and update pricing_checked (maximum age 7 days)")
            if self.estimate()["upper_usd"] > self.max_spend_usd:
                raise ValueError("Conservative estimate exceeds maximum authorized spend")

    def estimate(self):
        # One GPU, 2 physical CPU cores (4 vCPU equivalent), 16 GiB RAM, hard resource limits.
        rate = self.gpu_usd_s + 2 * self.cpu_usd_core_s + 16 * self.memory_usd_gib_s
        low_s = self.loading_s + self.warmup_s + self.benchmark_s
        return {
            "lower_usd": 0 if self.provider == "simulated" else round(rate * low_s, 4),
            "upper_usd": 0 if self.provider == "simulated" else round(
                rate * self.runtime_s * 1.25 + self.other_charge_allowance_usd, 4),
            "resource_usd_s": rate if self.provider == "modal" else 0,
            "credit_balance": "unknown", "credit_expiry": "unknown",
            "provider_reported_cost": None,
            "assumptions": "1 L4, 2 physical cores, 16 GiB RAM; 25% compute margin plus other-charge allowance",
            "other_charges": "Image build, network egress, taxes and plan fees may apply; allowance is not a cap",
            "shutdown": "Provider-enforced sandbox lifetime; client cleanup starts 30s early; no auto-replacement",
            "warning": "Estimate is not an exact billing cap. Credits may not cover all charges.",
        }

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
