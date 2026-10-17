import numpy as np
import pytest
from types import SimpleNamespace
from noise_module_lucid.lucid_integration import analytic_first


def test_analytic_first_chooses_forward_nearest_sensor_and_misses():
    origins=np.zeros((3,3));directions=np.array([[1.,0,0],[-1.,0,0],[0,0,1.]])
    centres=np.array([[1.,0,0],[2.,0,0],[-1.,0,0]])
    actual=analytic_first(origins,directions,centres,.1,SimpleNamespace(r=3,H=6))
    np.testing.assert_array_equal(actual,[0,2,-1])


def test_digit_subject_ignores_padding_and_sensor_order():
    torch=pytest.importorskip('torch')
    from noise_module_lucid.lucid_baseline import build_subject
    torch.manual_seed(54);model=build_subject().eval();x=torch.randn(2,8,6);x[:,:,5]=1
    padding=torch.randn(2,5,6)*100; padding[:,:,5]=0
    with torch.no_grad():
        base=model(x);permuted=model(x[:,torch.randperm(8)]);padded=model(torch.cat((x,padding),dim=1))
    torch.testing.assert_close(base,permuted,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(base,padded,atol=1e-6,rtol=1e-6)
    assert torch.isfinite(model(torch.zeros(2,8,6))).all()
