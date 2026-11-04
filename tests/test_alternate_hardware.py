from dataclasses import replace
import pytest
from inference_lab.config import Plan
from inference_lab.models import PROFILES, profile_config


@pytest.mark.parametrize('model_key', ['qwen-0.5b', 'qwen-3b'])
def test_a100_alternative_keeps_pins_and_validates_resources(model_key):
    original = Plan(**profile_config(model_key))
    alternative = Plan(**profile_config(model_key + '-a100'))
    alternative.validate()
    original.validate()
    assert alternative.model == original.model
    assert alternative.revision == original.revision
    assert alternative.gpu == 'A100-40GB'
    assert alternative.gpu_usd_s == .000583
    assert alternative.fingerprint() != original.fingerprint()
    for invalid in [replace(alternative, gpu='H100'), replace(alternative, gpu_count=2),
                    replace(alternative, ram_gib=1), replace(alternative, revision='unreviewed')]:
        with pytest.raises(ValueError):
            invalid.validate()
    assert len(PROFILES) == 4
