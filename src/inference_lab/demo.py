"""Free local integration demonstrations; does not provision any cloud resource."""
from dataclasses import replace
import json
from .config import Plan
from .store import Store
from .cli import launch
from .report import summarize, compare


def main():
    store = Store()
    baseline = Plan(requests=24, rate=12, concurrency=1, warmup_requests=1,
                    hypothesis="Synthetic baseline: test measurement and persistence")
    experiments = [baseline,
                   replace(baseline, concurrency=2,
                           hypothesis="Allowing two requests should reduce client admission rejection"),
                   replace(baseline, rate=24, hypothesis="A higher offered rate should expose admission limits"),
                   replace(baseline, simulated_fault="http503", warmup_requests=0,
                           hypothesis="Controlled HTTP 503 fixture should retain errors and verify cleanup"),
                   replace(baseline, hypothesis="Recovery: healthy fixture after controlled failure")]
    results = []
    for p in experiments:
        run = store.create(p)
        launch(store, run, "run")
        run = store.get(run["id"])
        results.append({"run": run, "summary": summarize(run, store.requests(run["id"]))})
    report = {"evidence": "SIMULATED SOFTWARE DEMONSTRATION ONLY",
              "comparison": compare(results[0], results[1]), "runs": results,
              "incident": {"symptom": "Every request received HTTP 503 in the fault run",
                           "evidence": results[3]["summary"],
                           "cause": "Explicit SIM_FAULT=http503 fixture, not a provider incident",
                           "recovery": results[4]["summary"],
                           "prevention": "Bound concurrency, retain error evidence, verify cleanup",
                           "limitations": "Does not test real quota, credit exhaustion, OOM or GPU recovery"}}
    path = store.root / "simulated-demo.json"
    path.write_text(json.dumps(report, indent=2))
    print(path)


if __name__ == "__main__":
    main()
