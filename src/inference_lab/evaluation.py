"""Versioned, deterministic task scoring. Never executes generated code."""
import json
import re

VERSION = "infra-quality-v2"


def builtin_cases():
    cases = []
    for i in range(10):
        a, b = 17 + i * 7, 25 + i * 3
        cases.append({"id": f"arithmetic-{i}", "category": "arithmetic", "scorer": "exact",
                      "prompt": f"Compute {a} + {b}. Reply only with the integer.", "expected": str(a + b)})
        cases.append({"id": f"extract-{i}", "category": "structured_extraction", "scorer": "json",
                      "prompt": f'Service worker-{i} has {i + 2} replicas in zone west. '
                                'Return only JSON with keys "service" (string) and "replicas" (integer).',
                      "expected": {"service": f"worker-{i}", "replicas": i + 2}})
        nums = [i + 8, i + 2, i + 5]
        cases.append({"id": f"python-{i}", "category": "code_reasoning", "scorer": "json",
                      "prompt": f"What does sorted({nums}) return in Python? Reply only with a JSON array.",
                      "expected": sorted(nums)})
        cases.append({"id": f"grounded-{i}", "category": "grounded_reading", "scorer": "exact",
                      "prompt": f"Notes: API timeout is {i + 10}s. Queue limit is {i + 20}. "
                                "Nothing specifies the GPU model. What GPU model is used? "
                                "If the notes do not say, reply only UNKNOWN.", "expected": "UNKNOWN"})
    return cases


def validate_cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 10000:
        raise ValueError("Evaluation file must contain 1–10000 JSONL cases")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not {"id", "category", "prompt", "scorer", "expected"} <= case.keys():
            raise ValueError("Quality cases require id, category, prompt, scorer and expected")
        if not all(isinstance(case[k], str) and case[k] for k in ("id", "category", "prompt", "scorer")):
            raise ValueError("Invalid quality case strings")
        if case["id"] in seen or case["scorer"] not in {"exact", "json"}:
            raise ValueError("Case IDs must be unique; scorers are exact or json")
        seen.add(case["id"])
    return cases


def score_case(case, output):
    if case["scorer"] == "json":
        try:
            actual = json.loads(output.strip())
            # Canonical JSON keeps booleans distinct from numbers; extra keys fail.
            passed = json.dumps(actual, sort_keys=True) == json.dumps(case["expected"], sort_keys=True)
            return {"pass": passed, "reason": "match" if passed else "json_mismatch"}
        except (ValueError, TypeError):
            return {"pass": False, "reason": "invalid_json"}
    def normalize(x):
        return re.sub(r"\s+", " ", str(x).strip()).casefold()
    passed = normalize(output) == normalize(case["expected"])
    return {"pass": passed, "reason": "match" if passed else "exact_mismatch"}
