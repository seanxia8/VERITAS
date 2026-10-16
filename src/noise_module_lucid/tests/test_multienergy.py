import numpy as np
import pytest
from noise_module_lucid.multienergy_data import transform_event,inside_cylinder
from noise_module_lucid.multienergy_baseline import metrics


def test_primary_vertex_transform_does_not_center_shower():
    raw={'vertex_m':np.array([1.,0,0]),'origins_m':np.array([[1.,0,0],[3.,0,0]]),'directions':np.array([[1.,0,0],[1.,0,0]]),'primary_direction':np.array([1.,0,0]),'primary_path_m':np.array([[1.,0,0],[3.,0,0]])}
    rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);vertex=np.array([.1,.2,.3])
    origins,directions,path,axis=transform_event(raw,rot,vertex)
    np.testing.assert_allclose(origins[0],vertex)
    np.testing.assert_allclose(origins[1],vertex+[0,2,0])
    np.testing.assert_allclose(axis,[0,1,0]);np.testing.assert_allclose(path,origins)
    assert inside_cylinder(path,1,3).tolist()==[True,False]


def test_endpoint_metrics_have_physical_units():
    truth=np.array([[1.,0,0,0,0,0,0]])
    prediction=np.array([[0.,1,0,.3,.4,0,np.log(1.2)]])
    m=metrics(prediction,truth)
    assert m['direction_mean_deg']==pytest.approx(90)
    assert m['vertex_rmse_m']==pytest.approx(.5)
    assert m['energy_relative_mae']==pytest.approx(.2)


def test_multitarget_mask_and_permutation_invariance():
    torch=pytest.importorskip('torch')
    from noise_module_lucid.multienergy_baseline import build_subject
    torch.manual_seed(7);model=build_subject().eval();x=torch.randn(2,8,6);x[:,:,5]=1
    padding=torch.full((2,3,6),100.);padding[:,:,5]=0
    with torch.no_grad():
        expected=model(x);permuted=model(x[:,torch.randperm(8)]);padded=model(torch.cat((x,padding),1))
    torch.testing.assert_close(expected,permuted,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(expected,padded,atol=1e-6,rtol=1e-6)
    assert expected.shape==(2,7)
