"""Supplemental public lifecycle checks after the immutable RED regressions."""
import copy
import io
from pathlib import Path
import hashlib
import json

import nbformat
import pytest
import torch

from nif.optimizers import ReferenceAdam
from nif.pruning import PolynomialMagnitudePruning


def layer_and_controller(**kwargs):
    layer = torch.nn.Linear(1, 10).double()
    with torch.no_grad():
        layer.weight.copy_(torch.arange(1.,11.).reshape(10,1))
        layer.bias.fill_(.25)
    ctl = PolynomialMagnitudePruning(layer, 'weight', **kwargs)
    return layer, ctl


def advance(layer, opt):
    layer.train(); opt.zero_grad()
    layer(torch.ones(1,1,dtype=layer.weight.dtype, device=layer.weight.device)).square().mean().backward()
    opt.step()


def test_exact_full_schedule_events_without_scientific_training():
    layer, ctl = layer_and_controller()
    events = []
    x = torch.ones(1,1,dtype=torch.float64)
    previous = 0
    # Tiny kernel forward-only control: no optimizer, scientific model or training data.
    with torch.no_grad():
        for step in range(1, 200005):
            layer(x)
            if ctl.refresh_count != previous:
                events.append(ctl.step)
                previous = ctl.refresh_count
    assert events == list(range(4000,200001,4000))
    assert ctl.refresh_count == 50
    assert torch.count_nonzero(layer.weight) == 1
    ctl.on_epoch_end()
    assert ctl.refresh_count == 50 and ctl.step == 200004


def test_inference_counter_bias_and_no_gradient_context():
    layer, ctl = layer_and_controller(end_step=4, frequency=2)
    bias = layer.bias.detach().clone()
    x = torch.ones(1,1,dtype=torch.float64)
    layer.eval()
    with torch.no_grad(): layer(x)
    assert ctl.step == 0 and ctl.refresh_count == 0
    layer.train()
    with torch.no_grad(): layer(x); layer(x)
    assert ctl.step == 2 and ctl.refresh_count == 1
    selected = ctl.mask == 0
    with torch.no_grad(): layer.weight[selected] = 7.
    layer.eval()
    layer(x).sum().backward()
    assert ctl.step == 2 and ctl.refresh_count == 1
    assert (layer.weight[selected] == 0).all()
    torch.testing.assert_close(layer.weight.grad, torch.ones_like(layer.weight))
    torch.testing.assert_close(layer.bias, bias, rtol=0, atol=0)
    torch.testing.assert_close(layer.bias.grad, torch.ones_like(layer.bias))


def test_checkpoint_resume_preserves_schedule_slots_and_strip(tmp_path):
    layer, ctl = layer_and_controller(end_step=6, frequency=2)
    opt = ReferenceAdam(layer.parameters())
    for _ in range(3): advance(layer, opt)
    ctl.on_epoch_end()
    buffer = io.BytesIO()
    torch.save(dict(model=layer.state_dict(), optimizer=opt.state_dict()),buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored, restored_ctl = layer_and_controller(end_step=50, frequency=10)
    restored.load_state_dict(saved['model'])
    resumed = ReferenceAdam(restored.parameters(), lr=.9)
    resumed.load_state_dict(saved['optimizer'])
    assert restored_ctl.step == 3 and restored_ctl.refresh_count == 1
    for _ in range(4):
        advance(layer,opt); advance(restored,resumed)
        for p,q in zip(layer.parameters(),restored.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        torch.testing.assert_close(ctl.mask,restored_ctl.mask,rtol=0,atol=0)
    assert restored_ctl.refresh_count == 3
    restored.eval()
    x = torch.ones(3,1,dtype=torch.float64)
    with torch.no_grad(): expected = restored(x)
    parameter = restored.weight
    restored_ctl.strip()
    assert restored.weight is parameter and parameter in resumed.state
    assert not restored._forward_pre_hooks
    assert set(restored.state_dict()) == {'weight','bias'}
    assert torch.count_nonzero(restored.weight) == 1
    plain = torch.nn.Linear(1,10).double()
    plain.load_state_dict(restored.state_dict())
    path = tmp_path/'sparse.pt'
    torch.jit.save(torch.jit.trace(plain,x),path)
    with torch.no_grad():
        torch.testing.assert_close(torch.jit.load(path)(x),expected,rtol=0,atol=0)


def test_deepcopy_and_dtype_movement_keep_hooks_local():
    layer, ctl = layer_and_controller(end_step=2, frequency=1)
    copied = copy.deepcopy(layer).float()
    with torch.no_grad(): copied(torch.ones(1,1))
    assert int(copied._nif_pruning.step) == 1 and ctl.step == 0
    assert copied._nif_pruning.mask.dtype == torch.float32
    assert torch.count_nonzero(layer.weight) == 10


def test_missing_gradients_use_shared_iteration_and_new_group():
    p = torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    q = torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    opt = ReferenceAdam([p], betas=(.5,.75))
    p.grad = torch.tensor([1e-12],dtype=p.dtype); opt.step()
    p.grad = None
    opt.add_param_group({'params':[q]})
    q.grad = torch.tensor([1e-12],dtype=q.dtype); opt.step()
    expected = -.001*(1-.75**2)**.5/(1-.5**2)*(.5e-12)/((.25e-24)**.5+1e-7)
    assert q.item() == pytest.approx(expected, rel=1e-14)
    assert opt.state_dict()['reference_iterations'] == 2
    assert opt.state[p]['exp_avg'].item() == .5e-12


@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_amsgrad_running_max_and_group_schedules(dtype):
    p = torch.nn.Parameter(torch.zeros(1,dtype=dtype))
    opt = ReferenceAdam([p], betas=(.5,.75), amsgrad=True)
    m = v = vmax = expected = torch.zeros(1,dtype=dtype)
    for t,g in enumerate((1.,0.,1e-12),1):
        lr = .001 / t
        opt.param_groups[0]['lr'] = lr
        p.grad = torch.full_like(p,g); opt.step()
        m = .5*m + .5*g
        v = .75*v + .25*g*g
        vmax = torch.maximum(vmax,v)
        expected = expected - lr*(1-.75**t)**.5/(1-.5**t)*m/(vmax.sqrt()+1e-7)
        torch.testing.assert_close(p,expected)
        torch.testing.assert_close(opt.state[p]['max_exp_avg_sq'],vmax)


@pytest.mark.parametrize('options', [{'eps':0.}, {'eps':-1.}, {'lr':-1.}, {'lr':float('nan')}, {'betas':(.9,1.)}])
def test_invalid_optimizer_groups_rejected(options):
    with pytest.raises(ValueError):
        ReferenceAdam([torch.nn.Parameter(torch.zeros(1))],**options)


def test_sparse_optimizer_failure_is_atomic():
    p = torch.nn.Parameter(torch.zeros(2))
    q = torch.nn.Parameter(torch.zeros(2))
    opt = ReferenceAdam([p,q])
    p.grad = torch.ones_like(p)
    q.grad = torch.sparse_coo_tensor(torch.tensor([[0]]),torch.tensor([1.]),(2,))
    with pytest.raises(RuntimeError, match='dense'): opt.step()
    assert opt.state_dict()['reference_iterations'] == 0
    assert torch.count_nonzero(p) == 0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_state_restore_and_masking():
    layer,ctl = layer_and_controller(end_step=2,frequency=1)
    layer.cuda()
    opt = ReferenceAdam(layer.parameters())
    advance(layer,opt); ctl.on_epoch_end()
    assert ctl.mask.is_cuda and opt.state[layer.weight]['exp_avg'].is_cuda
    restored,resumed_ctl = layer_and_controller()
    restored.load_state_dict(layer.cpu().state_dict())
    resumed = ReferenceAdam(restored.parameters())
    resumed.load_state_dict(opt.state_dict())
    advance(restored,resumed); resumed_ctl.on_epoch_end()
    assert torch.isfinite(restored.weight).all()


def test_red_test_fingerprints_unchanged_and_scientific_calls_migrated_v2():
    root=Path(__file__).resolve().parents[1]
    fingerprints=json.loads((root/'dev_tests/pruning_adam_pre_fix_fingerprints.json').read_text())
    for name in ('dev_tests/test_reference_adam.py','dev_tests/test_tutorial7_pruning.py'):
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==fingerprints[name]
    for number in (2,5,6,7,8):
        nb=nbformat.read(next((root/'tutorial').glob(f'{number}_*_pytorch.ipynb')),as_version=4)
        code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
        assert 'torch.optim.Adam(' not in code
        assert 'ReferenceAdam(' in code
    # Scope v2: tutorial 1 has an authorized, exactly bounded GC default correction.
    from dev_tests.tutorial1_gc_scope import assert_tutorial1_gc_scope_v2
    assert_tutorial1_gc_scope_v2()
    for number in (3,4):
        path=next((root/'tutorial').glob(f'{number}_*_pytorch.ipynb'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==fingerprints[path.relative_to(root).as_posix()]


@pytest.mark.parametrize('dtype,scalar', [(torch.float32,'float32'),(torch.float64,'float64')])
def test_new_adam_dense_slot_operation_order(dtype,scalar):
    # Keras new Adam evaluates Python (1-beta) before converting the multiplier,
    # then adds (g-m)*(1-beta). This is a rounding control, not epsilon algebra.
    import numpy as np
    cast=getattr(np,scalar)
    p=torch.nn.Parameter(torch.zeros(2,dtype=dtype))
    opt=ReferenceAdam([p])
    m=np.zeros(2,dtype=cast);v=m.copy()
    for i in range(1,11):
        g=np.array([.2*i,1e-12],dtype=cast)
        p.grad=torch.tensor(g);opt.step()
        m+=(g-m)*cast(1-.9)
        v+=(g*g-v)*cast(1-.999)
        torch.testing.assert_close(opt.state[p]['exp_avg'],torch.tensor(m),rtol=1e-7,atol=0.)
        torch.testing.assert_close(opt.state[p]['exp_avg_sq'],torch.tensor(v),rtol=1e-7,atol=0.)
