import asyncio
from dataclasses import replace
import json
import time
from unittest.mock import Mock
import httpx
import pytest
from inference_lab.benchmark import measure
from inference_lab.cli import require_approval, cleanup, launch
from inference_lab.config import Plan
from inference_lab.providers import Endpoint, ModalProvider
from inference_lab.report import percentile, summarize, compare
from inference_lab.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def test_approval_missing_changed_expired(store):
    from datetime import date
    p = Plan(provider="modal", pricing_checked=date.today().isoformat())
    r = store.create(p)
    with pytest.raises(ValueError, match="Missing approval"):
        require_approval(r, p)
    r["approval"] = {"fingerprint": p.fingerprint(), "expires_at": time.time() + 60}
    require_approval(r, p)
    with pytest.raises(ValueError, match="Missing approval"):
        require_approval(r, replace(p, concurrency=3))
    r["approval"]["expires_at"] = 0
    with pytest.raises(ValueError, match="expired"):
        require_approval(r, p)


@pytest.mark.parametrize("change", [{"rate": float("nan")}, {"concurrency": 0}, {"max_tokens": 100000},
                                    {"gpu": "H100"}, {"runtime_s": 20}, {"quantization": "awq"},
                                    {"model": "unverified/model"}, {"requests": 2.5}])
def test_invalid_profiles(change):
    with pytest.raises(ValueError):
        replace(Plan(), **change).validate()


def test_cost_margin_and_budget():
    from datetime import date
    p = Plan(provider="modal", pricing_checked=date.today().isoformat())
    assert p.estimate()["upper_usd"] > p.estimate()["lower_usd"] > 0
    with pytest.raises(ValueError, match="exceeds"):
        replace(p, max_spend_usd=.01).validate()


def test_cleanup_failure_does_not_leak_secrets(store):
    r = store.create(Plan())
    provider = Mock()
    provider.teardown.side_effect = RuntimeError("secret-token-do-not-log")
    cleanup(provider, r, store)
    assert store.get(r["id"])["shutdown"] == "unverified"
    assert "secret-token" not in json.dumps(store.events(r["id"]))


def test_recover_preserves_finished_and_partial(store):
    r = store.create(Plan())
    store.request(r["id"], {"id": "a", "status": "success", "output": "done"})
    store.request(r["id"], {"id": "b", "status": "running", "output": "part"})
    store.recover(r["id"])
    rows = store.requests(r["id"])
    assert rows[0]["status"] == "success"
    assert rows[1]["status"] == "incomplete"
    assert rows[1]["output"] == "part"


def test_pending_deployment_never_created_twice(store):
    r = store.create(Plan())
    r["deploy_intent"] = True
    with pytest.raises(ValueError, match="already attempted"):
        ModalProvider().deploy(r, Plan(), store)


def test_percentiles_and_throughput_denominators(store):
    assert percentile([1, 2, 3, 4], .5) == 2.5
    assert percentile([], .99) is None
    r = store.create(Plan(requests=3))
    r["segments"] = [{"started_at": 10, "ended_at": 12}]
    r["stages"] = ["benchmark"]
    rows = [{"phase": "steady", "status": "success", "latency_s": 1, "output_tokens": 10},
            {"phase": "steady", "status": "success", "latency_s": 2, "output_tokens": 20},
            {"phase": "steady", "status": "rejected"},
            {"phase": "warmup", "status": "success", "output_tokens": 100}]
    s = summarize(r, rows)
    assert s["successful_rps"] == 1
    assert s["output_tokens_s"] == 15
    assert s["error_rate_including_client_rejections"] == pytest.approx(1 / 3)
    assert not s["within_targets"]
    r["segments"][0]["ended_at"] = None
    assert summarize(r, rows)["successful_rps"] is None


def test_stream_chunks_are_not_called_tokens(store):
    async def run():
        data = '\n\n'.join([
            'data: ' + json.dumps({"choices": [{"delta": {"content": "many words in one chunk"}}]}),
            'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 5}}),
            'data: [DONE]', '',
        ])
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=data))) as client:
            r = store.create(Plan())
            return await measure(client, Endpoint("http://test", {}), Plan(), "prompt", "1", store, r["id"])
    result = asyncio.run(run())
    assert result["status"] == "success"
    assert result["output_tokens"] == 5
    assert len(result["chunk_times_s"]) == 1
    assert result["individual_inter_token_gaps_s"] is None


def test_truncated_stream_not_success(store):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text='data: {"choices": []}\n\n'))) as client:
            r = store.create(Plan())
            return await measure(client, Endpoint("http://test", {}), Plan(), "prompt", "1", store, r["id"])
    assert asyncio.run(run())["status"] == "error"


def test_local_end_to_end_and_no_replay(store):
    p = Plan(requests=4, rate=10, warmup_requests=1)
    r = store.create(p)
    launch(store, r, "run")
    saved = store.get(r["id"])
    assert saved["status"] == "complete"
    assert saved["shutdown"] == "verified"
    assert len(store.requests(r["id"])) == 5
    assert (store.root / r["id"] / "report.json").exists()
    launch(store, saved, "resume")
    assert len(store.requests(r["id"])) == 5


def test_failure_and_cleanup(store):
    p = Plan(requests=4, rate=10, warmup_requests=0, simulated_fault="http503")
    r = store.create(p)
    launch(store, r, "run")
    rows = store.requests(r["id"])
    assert all(x["error"] == "http_503" for x in rows)
    assert store.get(r["id"])["shutdown"] == "verified"


def test_timeout_is_bounded(store):
    p = Plan(requests=2, rate=10, warmup_requests=0, simulated_fault="timeout", request_timeout_s=.05)
    r = store.create(p)
    launch(store, r, "run")
    assert all(x["error"] == "timeout" for x in store.requests(r["id"]))
    assert store.get(r["id"])["shutdown"] == "verified"


def test_comparison_flags_confounds():
    a = {"run": {"plan": Plan().to_dict()}, "summary": {}}
    b = {"run": {"plan": replace(Plan(), concurrency=4, max_tokens=256).to_dict()}, "summary": {}}
    assert not compare(a, b)["single_variable"]


def test_unapproved_run_cannot_reach_provider(store, monkeypatch):
    from datetime import date
    p = Plan(provider="modal", pricing_checked=date.today().isoformat())
    factory = Mock()
    monkeypatch.setattr("inference_lab.cli.provider_for", factory)
    with pytest.raises(ValueError, match="Missing approval"):
        launch(store, store.create(p), "run")
    factory.assert_not_called()


def test_wrong_typed_approval_is_not_saved(store, monkeypatch):
    from inference_lab.cli import main
    r = store.create(Plan())
    monkeypatch.setattr("sys.argv", ["inference-lab", "--data", str(store.root), "approve", r["id"]])
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    assert main() == 1
    assert store.get(r["id"])["approval"] is None


def test_client_cancellation_preserves_incomplete(store):
    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
            await asyncio.sleep(10)

    async def run():
        r = store.create(Plan())
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=SlowStream()))) as client:
            task = asyncio.create_task(measure(client, Endpoint("http://test", {}), Plan(),
                                               "prompt", "1", store, r["id"]))
            await asyncio.sleep(.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        record = store.requests(r["id"])[0]
        assert record["status"] == "incomplete"
        assert record["output"] == "partial"
    asyncio.run(run())


def test_recovered_request_not_replayed(store):
    p = Plan(requests=3, rate=10, warmup_requests=0)
    r = store.create(p)
    store.request(r["id"], {"id": "steady-00000", "phase": "steady", "status": "running", "output": "part"})
    launch(store, r, "resume")
    rows = store.requests(r["id"])
    assert len(rows) == 3
    assert rows[0]["status"] == "incomplete"
    assert rows[0]["output"] == "part"
    assert all(x["status"] == "success" for x in rows[1:])


def test_dashboard_snapshots_across_refresh_threads(store):
    from concurrent.futures import ThreadPoolExecutor
    from inference_lab.store import read_snapshot

    run = store.create(Plan())
    store.request(run["id"], {"id": "first", "status": "success"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        snapshots = list(pool.map(read_snapshot, [store.root] * 6))
    for runs, requests in snapshots:
        assert runs[0]["id"] == run["id"]
        assert requests[run["id"]][0]["status"] == "success"
    store.request(run["id"], {"id": "second", "status": "running"})
    with ThreadPoolExecutor(max_workers=1) as pool:
        _, requests = pool.submit(read_snapshot, store.root).result()
    assert len(requests[run["id"]]) == 2
