"""Intervals are descriptive and explicitly conditional on observed samples."""
import math
import random


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bootstrap_interval(values, q=.95, draws=400, seed=42):
    if len(values) < 20:
        return None
    rng = random.Random(seed)
    stats = [quantile(rng.choices(values, k=len(values)), q) for _ in range(draws)]
    return [quantile(stats, .025), quantile(stats, .975)]


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    mid = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0, mid-half), min(1, mid+half)]
