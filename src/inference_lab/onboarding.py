"""Metadata-only Hugging Face onboarding; never imports repository Python or starts compute."""
import json
import re
from pathlib import Path
from .datasets import digest
from .models import ModelProfile

ARCHITECTURES = {'LlamaForCausalLM', 'MistralForCausalLM', 'Qwen2ForCausalLM', 'Qwen3ForCausalLM'}
HARDWARE = {'L4': (24, .000222), 'A100-40GB': (40, .000583), 'A100-80GB': (80, .000694)}


def validate_metadata(metadata):
    if not isinstance(metadata, dict) or metadata.get('sha256') != digest({k:v for k,v in metadata.items() if k != 'sha256'}):
        raise ValueError('Model metadata changed; onboard again and create a fresh plan')
    if metadata.get('schema') != 1 or not re.fullmatch(r'[0-9a-f]{40}', metadata.get('revision', '')):
        raise ValueError('Model revision must be a resolved commit SHA')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', metadata.get('model', '')):
        raise ValueError('Expected a Hugging Face namespace/model ID')
    config = metadata['config']
    architectures = config.get('architectures', [])
    if len(architectures) != 1 or architectures[0] not in ARCHITECTURES:
        raise ValueError('Unsupported architecture: first version supports dense Llama, Mistral, Qwen2 and Qwen3 causal LMs')
    if config.get('auto_map') or config.get('quantization_config') or config.get('is_encoder_decoder') or config.get('num_local_experts') or config.get('num_experts'):
        raise ValueError('Custom repository code, quantized weights, encoder-decoder and MoE models are not supported')
    for field in ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'max_position_embeddings']:
        if type(config.get(field)) is not int or config[field] <= 0:
            raise ValueError(f'Missing or invalid architecture dimension: {field}')
    heads = config['num_attention_heads']
    kv = config.get('num_key_value_heads', heads)
    dim = config.get('head_dim', config['hidden_size'] // heads)
    if type(kv) is not int or kv <= 0 or heads % kv or type(dim) is not int or dim <= 0 or config['hidden_size'] % heads:
        raise ValueError('Invalid attention dimensions')
    n = metadata.get('parameter_count')
    if type(n) is not int or not 0 < n < 100_000_000_000:
        raise ValueError('Verified safetensors parameter count must be positive and below 100B')
    if metadata.get('has_tokenizer_json') is not True:
        raise ValueError('A tokenizer.json is required for controlled token workloads')
    return config, kv, dim


def custom_profile(plan):
    m = plan.model_metadata
    c, kv, dim = validate_metadata(m)
    if (plan.model, plan.revision) != (m['model'], m['revision']):
        raise ValueError('Plan model/revision does not match the onboarded metadata')
    if plan.gpu not in HARDWARE or plan.gpu_count not in (1, 2, 4):
        raise ValueError('Unsupported hardware; use L4 or A100 with 1, 2 or 4 GPUs')
    if c['num_attention_heads'] % plan.gpu_count or (kv % plan.gpu_count and plan.gpu_count % kv):
        raise ValueError('Tensor parallel degree is incompatible with attention heads')
    if type(plan.cpu_cores) is not int or not 2 <= plan.cpu_cores <= 16 or type(plan.ram_gib) is not int or not 16 <= plan.ram_gib <= 256:
        raise ValueError('Custom hardware requires 2–16 CPU cores and 16–256 GiB RAM')
    if plan.max_model_len > c['max_position_embeddings']:
        raise ValueError('Requested context exceeds the model configuration')
    if plan.workload != 'tokens' and not m.get('has_chat_template'):
        raise ValueError('Chat/quality workloads require a recorded tokenizer chat template')
    if m.get('requires_auth') and not plan.hf_secret_name:
        raise ValueError('Private/gated model requires a Modal secret name containing HF_TOKEN')
    gib, rate = HARDWARE[plan.gpu]
    if plan.gpu_usd_s != rate:
        raise ValueError('GPU price must match the reviewed hardware rate')
    # Replicated KV heads must be counted when TP exceeds the KV-head count.
    effective_kv = max(kv, plan.gpu_count)
    return ModelProfile('custom', plan.model, plan.revision, m['parameter_count']/1e9,
                        plan.gpu, plan.gpu_count, gib, plan.cpu_cores, plan.ram_gib,
                        rate, c['num_hidden_layers'], effective_kv, dim)


def inspect_model(repo_id, revision='main', token=None):
    from huggingface_hub import HfApi, hf_hub_download
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo_id):
        raise ValueError('Expected a Hugging Face namespace/model ID')
    api = HfApi(token=token)
    info = api.model_info(repo_id, revision=revision, files_metadata=False)
    def read(name):
        path = hf_hub_download(repo_id, name, revision=info.sha, token=token)
        if Path(path).stat().st_size > 2_000_000:
            raise ValueError('Metadata JSON is too large')
        return json.loads(Path(path).read_text())
    config = read('config.json')
    files = {s.rfilename for s in info.siblings}
    tokenizer = read('tokenizer_config.json') if 'tokenizer_config.json' in files else {}
    safetensors = info.safetensors
    metadata = {'schema':1, 'model':repo_id, 'revision':info.sha, 'config':config,
                'parameter_count':getattr(safetensors, 'total', None),
                'has_tokenizer_json':'tokenizer.json' in files,
                'has_chat_template':bool(tokenizer.get('chat_template') or 'chat_template.jinja' in files),
                'tokenizer_config_sha256':digest(tokenizer),
                'requires_auth':bool(info.private or info.gated),
                'license':getattr(info.card_data, 'license', None),
                'validation':'metadata_checked_not_gpu_validated'}
    metadata['sha256'] = digest(metadata)
    validate_metadata(metadata)
    return metadata


def preflight_report(plan):
    from .models import profile_for
    from .providers import serving_command
    plan.validate()
    profile = custom_profile(plan) if plan.model_metadata else profile_for(plan.model, plan.gpu)
    budget = profile.memory_budget(plan.max_model_len, plan.server_max_sequences)
    budget['engine_budget_gib'] = profile.gpu_gib * profile.gpu_count * plan.gpu_memory_utilization
    return {'status':'configuration_validated_not_gpu_validated', 'plan_fingerprint':plan.fingerprint(),
            'model':plan.model, 'revision':plan.revision, 'memory_budget':budget,
            'estimate':plan.estimate(), 'server_argv':serving_command(plan),
            'hf_secret_name':plan.hf_secret_name,
            'remaining_checks':['plan prepares the tokenizer and hashed workload',
                                'Modal credentials, secret existence and allocation are not checked here',
                                'GPU smoke test must verify serving identity and requests',
                                'Repeated load tests and quality evaluation remain separate evidence'],
            'limitations':['Analytical memory estimate, not measured peak VRAM',
                           'Metadata hash detects changes; it is not a signature or independent attestation',
                           'Review the repository license and access terms yourself']}
