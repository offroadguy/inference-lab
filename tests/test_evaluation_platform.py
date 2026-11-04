from dataclasses import replace
from datetime import date
import json
import time
from unittest.mock import Mock

import pytest

from inference_lab.billing import reconcile
from inference_lab.campaigns import create, approve, execute_campaign, write_report
from inference_lab.cli import launch, require_approval
from inference_lab.config import Plan
from inference_lab.datasets import create_run, prepare, validate_manifest
from inference_lab.evaluation import builtin_cases, score_case, validate_cases
from inference_lab.models import PROFILES, profile_config
from inference_lab.reference import command, crosscheck, normalize
from inference_lab.providers import Endpoint
from inference_lab.report import summarize
from inference_lab.statistics import bootstrap_interval, wilson
from inference_lab.store import Store
from inference_lab.telemetry import parse_metrics, histogram_deltas, parse_gpu_csv


class FixtureTokenizer:
    def encode(self, text, add_special_tokens=False):
        return type("Encoding", (), {"ids": list(range(100, 100 + len(text.split())))})()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def test_four_profiles_have_pinned_revisions_and_feasible_budget():
    assert len(PROFILES) == 4
    for key, profile in PROFILES.items():
        plan = Plan(**profile_config(key))
        plan.validate()
        assert len(profile.revision) == 40
        assert profile.parameters_b < 100
        assert profile.memory_budget()["estimated_total_gib"] < profile.memory_budget()["engine_budget_gib"]
        with pytest.raises(ValueError, match="reviewed profile"):
            replace(plan, gpu_count=plan.gpu_count + 1).validate()


def test_multigpu_cost_includes_each_gpu():
    p = Plan(provider="modal", **profile_config("qwen-32b"), max_spend_usd=10,
             pricing_checked=date.today().isoformat())
    p.validate()
    assert p.estimate()["resource_usd_s"] == pytest.approx(2*p.gpu_usd_s + 8*p.cpu_usd_core_s + 128*p.memory_usd_gib_s)


def test_manifest_exact_lengths_and_tamper_detection():
    p = Plan(workload="tokens", requests=6, input_lengths=(16, 32), output_lengths=(4, 8))
    a = prepare(p, FixtureTokenizer())
    b = prepare(p, FixtureTokenizer())
    assert a == b
    assert [len(r["prompt"]) for r in a["records"]] == [16, 32]*3
    validate_manifest(a)
    a["records"][0]["prompt"].append(999)
    with pytest.raises(ValueError, match="changed"):
        validate_manifest(a)


def test_controlled_token_run_and_actual_counts(store):
    p = Plan(workload="tokens", requests=6, rate=10, input_lengths=(16, 32), output_lengths=(4, 8),
             warmup_requests=1, arrival="poisson")
    r = create_run(store, p, prepare(p, FixtureTokenizer()))
    launch(store, r, "run")
    records = [x for x in store.requests(r["id"]) if x["phase"] == "steady"]
    assert len(records) == 6
    assert all(x["length_conformant"] for x in records if x["status"] == "success")
    summary = summarize(store.get(r["id"]), store.requests(r["id"]))
    assert summary["goodput_rps"] is not None
    assert not summary["capacity_evidence_eligible"]
    assert "simulated evidence" in summary["capacity_exclusions"]


def test_quality_set_and_structural_scoring():
    cases = validate_cases(builtin_cases())
    assert len(cases) == 40
    assert len({c["category"] for c in cases}) == 4
    for case in cases:
        output = json.dumps(case["expected"]) if case["scorer"] == "json" else case["expected"]
        assert score_case(case, output)["pass"]
        assert not score_case(case, "irrelevant response")["pass"]
    case = {"scorer": "json", "expected": {"a": 1}}
    assert not score_case(case, '{"a": true}')["pass"]
    assert not score_case(case, '{"a": 1, "extra": 2}')["pass"]
    with pytest.raises(ValueError):
        validate_cases([cases[0], cases[0]])


def test_quality_requires_coverage():
    with pytest.raises(ValueError, match="at least 40"):
        prepare(Plan(workload="quality", requests=4))
    assert len(prepare(Plan(workload="quality", requests=40))["records"]) == 40


def test_statistical_intervals_are_not_zero_error_claims():
    low, high = wilson(0, 20)
    assert low == pytest.approx(0)
    assert high > .1
    assert bootstrap_interval([1]*10) is None
    assert bootstrap_interval([1]*30) == [1, 1]


def test_server_histogram_deltas_and_counter_reset():
    text = '''# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_count{model_name="qwen",engine="0"} 4
vllm:request_queue_time_seconds_sum{model_name="qwen",engine="0"} 2
vllm:request_queue_time_seconds_bucket{model_name="qwen",le="+Inf"} 4
'''
    before = {"series": parse_metrics(text)}
    after = {"series": parse_metrics(text.replace('} 4', '} 6').replace('} 2', '} 3'))}
    result = histogram_deltas([before, after])["histograms"]["vllm:request_queue_time_seconds"]
    assert result["count"] == 2
    assert result["mean_s"] == .5
    assert histogram_deltas([after, before])["histograms"]["vllm:request_queue_time_seconds"]["counter_reset"]
    assert parse_gpu_csv('0, NVIDIA L4, 550.1, 1024, 24576, 80')[0]["memory_used_bytes"] == 2**30


def test_manifest_bound_to_approval(store):
    p = Plan(provider="modal", pricing_checked=date.today().isoformat())
    r = create_run(store, p)
    r["approval"] = {"fingerprint": p.fingerprint(), "expires_at": time.time()+60}
    with pytest.raises(ValueError, match="workload is not covered"):
        require_approval(r, p)
    r["approval"]["manifest_sha256"] = r["manifest"]["sha256"]
    require_approval(r, p)


def test_campaign_aggregate_approval_and_no_duplicate_completed_work(store, monkeypatch):
    campaign = create(store, ["qwen-0.5b"], [10], 2,
                      {"provider": "simulated", "requests": 2, "warmup_requests": 0}, 1)
    fake_launch = Mock()
    monkeypatch.setattr("inference_lab.cli.launch", fake_launch)
    with pytest.raises(ValueError, match="approval"):
        execute_campaign(store, campaign)
    fake_launch.assert_not_called()
    phrase = f"APPROVE SWEEP {campaign['id']} USD 1.00"
    with pytest.raises(ValueError):
        approve(store, campaign, "yes")
    approve(store, campaign, phrase)
    def finish(store, run, command):
        run.update(status="complete", shutdown="verified")
        store.save(run)
    fake_launch.side_effect = finish
    execute_campaign(store, campaign)
    assert fake_launch.call_count == 2
    execute_campaign(store, campaign)
    assert fake_launch.call_count == 2
    assert not write_report(store, campaign)["highest_tested_qualified_offered_rps"]


def test_campaign_stops_before_next_run_when_cleanup_unverified(store, monkeypatch):
    campaign = create(store, ["qwen-0.5b"], [10], 2, {"provider": "simulated", "requests": 2}, 1)
    approve(store, campaign, f"APPROVE SWEEP {campaign['id']} USD 1.00")
    def fail_cleanup(store, run, command):
        run.update(status="complete", shutdown="unverified")
        store.save(run)
    mock = Mock(side_effect=fail_cleanup)
    monkeypatch.setattr("inference_lab.cli.launch", mock)
    with pytest.raises(ValueError, match="cleanup"):
        execute_campaign(store, campaign)
    assert mock.call_count == 1
    assert campaign["status"] == "partial"


def test_billing_requires_exact_resource_and_provenance(store, tmp_path):
    p = Plan(provider="modal")
    r = store.create(p)
    r.update(resource_id="sb-fixture", started_at=10, stopped_at=20)
    bill = {"run_id": r["id"], "resource_id": "wrong", "total_usd": 1.5, "currency": "USD",
            "source": "modal_export", "allocation": "exclusive_resource", "period_start": 0, "period_end": 30}
    path = tmp_path / "bill.json"
    path.write_text(json.dumps(bill))
    with pytest.raises(ValueError, match="exact run"):
        reconcile(store, r, path)
    bill["resource_id"] = r["resource_id"]
    path.write_text(json.dumps(bill))
    result = reconcile(store, r, path)
    assert result["total_usd"] == 1.5
    assert "not_independently_verified" in result["verification"]
    assert summarize(r, [])["provider_reported_cost_usd"] == 1.5


def test_reference_units_and_no_false_equivalence(tmp_path):
    raw = {"request_throughput": 2, "p95_e2el_ms": 100, "p95_ttft_ms": 20, "completed": 10}
    result = normalize(raw)
    assert result["latency_p95_s"] == .1
    assert not crosscheck(result, result)["equivalence_claim"]
    p = Plan(provider="modal", workload="tokens", input_lengths=(128,), output_lengths=(64,), arrival="poisson")
    args = command(p, Endpoint("https://test", {"Authorization": "Bearer PRIVATE"}), tmp_path / "reference")
    assert "PRIVATE" not in str(args)
    assert "OPENAI_API_KEY" in args
    with pytest.raises(ValueError):
        command(replace(p, prefix_caching=True), Endpoint("https://test", {}), tmp_path)


def test_measurement_against_known_stream_timeline(store, monkeypatch):
    import httpx
    from inference_lab.benchmark import measure
    import inference_lab.benchmark as benchmark
    clock = [0.0]
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    class TimedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for at, word in [(1.0, "a"), (1.2, "b"), (1.6, "c")]:
                clock[0] = at
                yield ('data: ' + json.dumps({"choices": [{"delta": {"content": word}}]}) + '\n\n').encode()
            clock[0] = 1.7
            yield b'data: {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 3}}\n\n'
            clock[0] = 1.8
            yield b'data: [DONE]\n\n'
    async def run():
        r = store.create(Plan())
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=TimedStream()))) as client:
            return await measure(client, Endpoint("http://fixture", {}), Plan(), "prompt", "1", store, r["id"])
    import asyncio
    result = asyncio.run(run())
    assert result["ttft_s"] == 1.0
    assert result["latency_s"] == 1.8
    assert result["tpot_s"] == pytest.approx(.3)
    assert result["inter_chunk_gaps_s"] == pytest.approx([.2, .4])
    assert result["individual_inter_token_gaps_s"] is None
