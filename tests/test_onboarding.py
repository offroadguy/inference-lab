from dataclasses import replace
from copy import deepcopy
import pytest
from inference_lab.config import Plan
from inference_lab.datasets import digest, create_run, prepare
from inference_lab.onboarding import validate_metadata, preflight_report
from inference_lab.providers import serving_command
from inference_lab.cli import require_approval
from inference_lab.store import Store


def metadata():
    m = {'schema':1, 'model':'example/dense', 'revision':'a'*40,
         'config':{'architectures':['LlamaForCausalLM'], 'num_hidden_layers':12,
                   'hidden_size':768, 'num_attention_heads':12, 'num_key_value_heads':4,
                   'max_position_embeddings':8192},
         'parameter_count':135000000, 'has_tokenizer_json':True,
         'has_chat_template':True, 'requires_auth':False}
    m['sha256']=digest(m)
    return m


def plan(**kw):
    return Plan(model='example/dense',revision='a'*40, model_metadata=metadata(),
                gpu='A100-40GB',gpu_usd_s=.000583,workload='tokens',**kw)


def test_offline_preflight_and_command_bind_engine_controls():
    p=plan(dtype='bfloat16', enforce_eager=False, chunked_prefill=True,
           max_num_batched_tokens=1024,gpu_memory_utilization=.7)
    result=preflight_report(p)
    argv=serving_command(p)
    assert result['status']=='configuration_validated_not_gpu_validated'
    assert '--enforce-eager' not in argv
    assert argv[argv.index('--dtype')+1]=='bfloat16'
    assert '--enable-chunked-prefill' in argv
    assert result['memory_budget']['engine_budget_gib']==28
    assert p.fingerprint()!=replace(p,gpu_memory_utilization=.8).fingerprint()


@pytest.mark.parametrize('change', [{'architectures':['Unknown']}, {'auto_map':{'AutoModel':'code.py'}},
                                   {'quantization_config':{'quant_method':'awq'}}, {'num_experts':8},
                                   {'num_key_value_heads':5}])
def test_unsupported_architecture_rejected_even_with_valid_hash(change):
    m=metadata();m['config'].update(change);m['sha256']=digest({k:v for k,v in m.items() if k!='sha256'})
    with pytest.raises(ValueError): replace(plan(),model_metadata=m).validate()


def test_metadata_tampering_context_memory_and_auth():
    m=metadata();m['parameter_count']=1
    with pytest.raises(ValueError,match='changed'): validate_metadata(m)
    for p in [replace(plan(),max_model_len=16384),replace(plan(),gpu_count=4,ram_gib=1),
              replace(plan(),max_num_batched_tokens=512,chunked_prefill=False),
              replace(plan(),gpu_memory_utilization=float('nan'))]:
        with pytest.raises(ValueError): p.validate()
    m=metadata();m['requires_auth']=True;m['sha256']=digest({k:v for k,v in m.items() if k!='sha256'})
    with pytest.raises(ValueError,match='secret'): replace(plan(),model_metadata=m).validate()
    replace(plan(),model_metadata=m,hf_secret_name='hf-read-access').validate()


def test_engine_change_invalidates_approval_and_preserves_provenance(tmp_path):
    from datetime import date
    p=replace(plan(),provider='modal',max_spend_usd=3,pricing_checked=date.today().isoformat())
    class Tokenizer:
        def encode(self,*a,**kw):return type('Encoded',(),{'ids':[1,2,3]})()
    store=Store(tmp_path)
    r=create_run(store,p,prepare(p,Tokenizer()))
    assert r['provenance']['workload_sha256']==r['manifest']['sha256']
    r['approval']={'fingerprint':p.fingerprint(),'manifest_sha256':r['manifest']['sha256'],'expires_at':1e12}
    require_approval(r,p)
    with pytest.raises(ValueError,match='changed plan'):require_approval(r,replace(p,enforce_eager=False))
    bad=deepcopy(r['manifest']);bad['model']='another/model';bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='does not match'):create_run(store,p,bad)


def test_capacity_rejects_timing_slo_failure_and_comparison_flags_source_change(tmp_path):
    from inference_lab.report import summarize, compare
    from datetime import date
    p=plan(provider='modal',requests=200,max_spend_usd=3,pricing_checked=date.today().isoformat())
    class Tokenizer:
        def encode(self,*a,**kw):return type('Encoded',(),{'ids':[1,2,3]})()
    r=create_run(Store(tmp_path),p,prepare(p,Tokenizer()))
    r.update(status='complete',shutdown='verified',server_identity={'verified':True},
             segments=[{'started_at':0,'ended_at':100}],stages=['benchmark'],
             approval={'fingerprint':p.fingerprint()})
    rows=[{'phase':'steady','status':'success','latency_s':.1,'ttft_s':9,'tpot_s':.01,
           'output_tokens':32,'prompt_tokens':128,'length_conformant':True,'schedule_lag_s':0} for _ in range(200)]
    summary=summarize(r,rows)
    assert 'first-content p95 target not met or samples missing' in summary['capacity_exclusions']
    b=deepcopy(r);b['provenance']['source_tree_sha256']='different'
    result=compare({'run':r,'summary':summary},{'run':b,'summary':summary})
    assert 'Source Code changed between runs' in result['comparability_caveats']
