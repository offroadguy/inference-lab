from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math
from .models import profile_for

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
    cpu_cores: int = 2
    ram_gib: int = 16
    max_model_len: int = 4096
    server_max_sequences: int = 32
    input_lengths: tuple = (128, 512, 2048)
    output_lengths: tuple = (64, 128, 256)
    arrival: str = "regular"
    evaluation_file: str | None = None
    target_ttft_s: float = 2.0
    target_tpot_s: float = 0.1
    min_capacity_samples: int = 200
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
    reference_s: int = 0
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
    model_metadata: dict | None = None
    hf_secret_name: str | None = None
    gpu_memory_utilization: float = 0.80
    enforce_eager: bool = True
    max_num_batched_tokens: int | None = None
    chunked_prefill: bool | None = None
    prefix_caching: bool = False
    hypothesis: str = "Establish the baseline"
    baseline_id: str | None = None
    repetition: int = 1
    simulated_fault: str = "none"

    def validate(self):
        if type(self.prefix_caching) is not bool:
            raise ValueError("prefix_caching must be a boolean")
        import re
        if self.hf_secret_name is not None and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', self.hf_secret_name):
            raise ValueError("Use a Modal secret name, never a token value")
        if self.hf_secret_name and re.fullmatch(r'hf_[A-Za-z0-9]{20,}', self.hf_secret_name):
            raise ValueError("A Hugging Face token is not a Modal secret name")
        if type(self.enforce_eager) is not bool or (self.chunked_prefill is not None and type(self.chunked_prefill) is not bool):
            raise ValueError("Engine switches must be booleans")
        if isinstance(self.gpu_memory_utilization, bool) or not isinstance(self.gpu_memory_utilization, (int, float)) or not .5 <= self.gpu_memory_utilization <= .9:
            raise ValueError("GPU memory utilization must be 0.5–0.9")
        if self.max_num_batched_tokens is not None:
            if type(self.max_num_batched_tokens) is not int or not 512 <= self.max_num_batched_tokens <= 32768:
                raise ValueError("Batched token budget must be 512–32768")
            if self.chunked_prefill is not True and self.max_num_batched_tokens < self.max_model_len:
                raise ValueError("A batched token budget below context requires explicit chunked prefill")
        if self.dtype not in {"half", "bfloat16"}:
            raise ValueError("Only FP16 and BF16 serving are supported")
        if self.model_metadata is not None:
            from .onboarding import custom_profile
            profile = custom_profile(self)
        else:
            profile = profile_for(self.model, self.gpu)
        if (self.revision, self.image, self.gpu, self.gpu_count, self.dtype, self.quantization,
            self.cpu_cores, self.ram_gib) != (
            profile.revision, IMAGE, profile.gpu, profile.gpu_count, self.dtype, "none",
            profile.cpu_cores, profile.ram_gib
        ):
            raise ValueError("Model, revision and hardware must match a reviewed profile")
        if type(self.max_model_len) is not int or not 512 <= self.max_model_len <= 32768:
            raise ValueError("Context must be 512–32768 tokens")
        if type(self.server_max_sequences) is not int or not 1 <= self.server_max_sequences <= 32:
            raise ValueError("Server sequence cap must be 1–32")
        budget = profile.memory_budget(self.max_model_len, self.server_max_sequences)
        budget["engine_budget_gib"] = profile.gpu_gib * profile.gpu_count * self.gpu_memory_utilization
        if budget["estimated_total_gib"] > budget["engine_budget_gib"]:
            raise ValueError("Analytical weights/KV/overhead budget exceeds engine memory envelope")
        if self.arrival not in {"regular", "poisson"}:
            raise ValueError("Arrival must be regular or poisson")
        for lengths in (self.input_lengths, self.output_lengths):
            if not isinstance(lengths, (list, tuple)) or not lengths or len(lengths) > 16:
                raise ValueError("Token length distributions need 1–16 positive integer lengths")
            if any(type(n) is not int or n <= 0 for n in lengths):
                raise ValueError("Token lengths must be positive integers")
        if max(self.input_lengths) + max(self.output_lengths) > self.max_model_len:
            raise ValueError("Input + output length exceeds model context")
        if max(self.output_lengths) > 1024:
            raise ValueError("Controlled output length exceeds 1024")
        if self.provider not in {"modal", "simulated"} or self.workload not in {"chat", "summary", "quality", "tokens"}:
            raise ValueError("Unsupported provider or workload")
        for key in ("requests", "concurrency", "max_tokens", "loading_s", "warmup_s", "benchmark_s",
                    "idle_s", "runtime_s", "warmup_requests", "reference_s", "repetition", "seed", "gpu_count", "cpu_cores", "ram_gib", "min_capacity_samples"):
            v = getattr(self, key)
            if type(v) is not int or v < (0 if key in {"warmup_requests", "seed", "reference_s"} else 1):
                raise ValueError(f"Invalid integer: {key}")
        for key in ("rate", "request_timeout_s", "max_spend_usd", "gpu_usd_s", "cpu_usd_core_s",
                    "memory_usd_gib_s", "other_charge_allowance_usd", "target_p95_s", "target_ttft_s", "target_tpot_s"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"Invalid positive number: {key}")
        if not 0 <= self.temperature <= 2 or not 0 <= self.target_error_rate <= 1:
            raise ValueError("Invalid temperature or error target")
        if self.concurrency > 32 or self.requests > 10000 or self.max_tokens > 1024:
            raise ValueError("Profile limits: concurrency <=32, requests <=10000, output <=1024")
        if not 120 <= self.runtime_s <= 3600:
            raise ValueError("Runtime must be 120–3600 seconds")
        if self.loading_s + self.warmup_s + self.benchmark_s + self.idle_s + self.reference_s > self.runtime_s - 30:
            raise ValueError("Stage budgets must leave 30 seconds for cleanup")
        if self.requests / self.rate > self.benchmark_s:
            raise ValueError("Offered workload does not fit benchmark budget")
        if self.reference_s:
            if self.provider != "modal" or self.workload != "tokens" or self.arrival != "poisson" or self.prefix_caching:
                raise ValueError("Upstream reference requires Modal token workload, Poisson arrivals and no prefix cache")
            if len(set(self.input_lengths)) != 1 or len(set(self.output_lengths)) != 1:
                raise ValueError("Upstream reference requires single input/output token lengths")
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
        rate = self.gpu_count * self.gpu_usd_s + self.cpu_cores * self.cpu_usd_core_s + self.ram_gib * self.memory_usd_gib_s
        low_s = self.loading_s + self.warmup_s + self.benchmark_s
        return {
            "lower_usd": 0 if self.provider == "simulated" else round(rate * low_s, 4),
            "upper_usd": 0 if self.provider == "simulated" else round(
                rate * self.runtime_s * 1.25 + self.other_charge_allowance_usd, 4),
            "resource_usd_s": rate if self.provider == "modal" else 0,
            "credit_balance": "unknown", "credit_expiry": "unknown",
            "provider_reported_cost": None,
            "assumptions": f"{self.gpu_count} {self.gpu}, {self.cpu_cores} physical cores, {self.ram_gib} GiB RAM; 25% compute margin plus allowance",
            "other_charges": "Image build, network egress, taxes and plan fees may apply; allowance is not a cap",
            "shutdown": "Provider-enforced sandbox lifetime; client cleanup starts 30s early; no auto-replacement",
            "warning": "Estimate is not an exact billing cap. Credits may not cover all charges.",
        }

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
