from copy import deepcopy
import json
from inference_lab.config import Plan
from inference_lab.datasets import digest
from inference_lab.report import smoke_checks, evidence_bundle, export
from inference_lab.store import Store


def test_smoke_requires_complete_intact_evidence(tmp_path):
    store = Store(tmp_path)
    run = store.create(Plan(provider='modal', requests=10, workload='tokens'))
    run.update(status='complete', stages=['benchmark'], shutdown='verified', server_identity={'verified': True})
    manifest = {'records': list(range(10)), 'model': run['plan']['model'], 'tokenizer_revision': run['plan']['revision']}
    manifest['sha256'] = digest(manifest)
    run['manifest'] = manifest
    rows = [dict(phase='steady', status='success', length_conformant=True) for _ in range(10)]
    assert all(smoke_checks(run, rows).values())
    assert not all(smoke_checks(run, rows[:1]).values())
    rows[0]['length_conformant'] = False
    assert not all(smoke_checks(run, rows).values())
    rows[0]['length_conformant'] = True
    run['manifest']['records'].append('tampered')
    assert not all(smoke_checks(run, rows).values())
    store.db.close()


def test_portable_evidence_matches_cli_without_mutation(tmp_path):
    store = Store(tmp_path)
    run = store.create(Plan())
    run['provenance'] = {'source_tree_sha256': 'test'}
    run['telemetry'] = [{'at': 1, 'values': {'queue': 0}}]
    before = deepcopy(run)
    store.event(run['id'], 'test', detail='retained')
    bundle = evidence_bundle(run, [], store.events(run['id']))
    assert run == before
    assert bundle['run']['provenance'] == before['provenance']
    assert bundle['run']['telemetry'] == before['telemetry']
    assert bundle['events']
    folder = export(store, run)
    assert json.loads((folder / 'report.json').read_text()) == json.loads(json.dumps(bundle))
    store.db.close()


def test_dashboard_navigation_and_invalid_preflight(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('LAB_DATA', str(tmp_path))
    app = AppTest.from_file('src/inference_lab/dashboard.py', default_timeout=30).run()
    for page in ['Models', 'Runs', 'Campaigns', 'New experiment']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    assert not app.error
    app.text_area[0].set_value('{"unknown": true}').run()
    assert app.error
    assert not app.get('download_button')
    assert not app.exception
