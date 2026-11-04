"""Reviewed model profiles, not claims of successful GPU validation."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelProfile:
    key: str
    model: str
    revision: str
    parameters_b: float
    gpu: str
    gpu_count: int
    gpu_gib: int
    cpu_cores: int
    ram_gib: int
    gpu_usd_s: float
    layers: int
    kv_heads: int
    head_dim: int

    def memory_budget(self, context=4096, sequences=32):
        weights = self.parameters_b * 1e9 * 2 / 2**30
        kv = 2 * self.layers * self.kv_heads * self.head_dim * 2 * context * sequences / 2**30
        overhead = 4 * self.gpu_count
        return {"weights_gib": weights, "kv_cache_full_context_gib": kv,
                "overhead_allowance_gib": overhead, "estimated_total_gib": weights + kv + overhead,
                "engine_budget_gib": self.gpu_gib * self.gpu_count * .8,
                "note": "Analytical FP16 estimate, not measured allocation; TP sharding and runtime overhead require validation"}

    def to_dict(self):
        return {**asdict(self), "status": "configured_not_gpu_validated", "memory_budget": self.memory_budget()}


PROFILES = {
    "qwen-0.5b": ModelProfile("qwen-0.5b", "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775", .49, "L4", 1, 24, 2, 16, .000222, 24, 2, 64),
    "qwen-3b": ModelProfile("qwen-3b", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", 3.09, "L4", 1, 24, 4, 32, .000222, 36, 2, 128),
    "qwen-7b": ModelProfile("qwen-7b", "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28", 7.61, "A100-40GB", 1, 40, 4, 48, .000583, 28, 4, 128),
    "qwen-32b": ModelProfile("qwen-32b", "Qwen/Qwen2.5-32B-Instruct",
        "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd", 32.5, "A100-80GB", 2, 80, 8, 128, .000694, 64, 8, 128),
}


def profile_for(model):
    for profile in PROFILES.values():
        if profile.model == model:
            return profile
    raise ValueError("Unknown model; select a reviewed profile with the models command")


def profile_config(key):
    p = PROFILES[key]
    return {"model": p.model, "revision": p.revision, "gpu": p.gpu, "gpu_count": p.gpu_count,
            "cpu_cores": p.cpu_cores, "ram_gib": p.ram_gib, "gpu_usd_s": p.gpu_usd_s}
