"""V4 RED-frozen moment oracle, transcribed from preserved TF AdaBelief source.

Dense recurrence: ad9113d external_optimizers.py:462-493. Cast betas first,
subtract in parameter dtype, multiply each term, add, use UPDATED m in v.
Native float64 callers retain their supplied precision; promoted legacy-f32
hyperparameters are a separately labeled control. No TF runtime or snapshots.
"""
import copy
import io

import numpy as np
import pytest
import torch

from nif.optimizers import AdaBeliefOptimizer
from dev_tests.test_adabelief_scalar_dtype import direction

NOMINAL=(.9,.999)
LEGACY_F32=tuple(float(np.float32(b)) for b in NOMINAL)


def source_moments(m,v,g,betas=NOMINAL,eps=1e-14,observed=None):
    scalar=m.dtype.type
    b1,b2=map(scalar,betas)
    next_m=b1*m+(scalar(1)-b1)*g
    residual=np.square(g-next_m)
    if observed is not None:residual=residual*observed
    # Preserve the native sparse epsilon-addition order; observed-row behavior
    # is tested separately, not claimed as bitwise TF scatter kernel parity.
    next_v=(b2*v+(scalar(1)-b2)*residual)+scalar(eps)
    return next_m,next_v


def exact(a,b):
    torch.testing.assert_close(a,torch.as_tensor(b,dtype=a.dtype),rtol=0,atol=0)


def test_public_float32_complement_values_and_default_constructor_step_one():
    b1,b2=map(np.float32,NOMINAL)
    assert float(np.float32(1)-b1)==.10000002384185791
    assert float(np.float32(1)-b2)==.0009999871253967285
    assert float(np.float32(1-.9))==.10000000149011612
    assert float(np.float32(1-.999))==.0010000000474974513
    p=torch.nn.Parameter(torch.zeros(4))
    opt=AdaBeliefOptimizer([p])  # literal public defaults, NO beta overrides
    g=np.array([1.,-.375,0.,1e-12],dtype=np.float32)
    m,v=source_moments(np.zeros_like(g),np.zeros_like(g),g)
    p.grad=torch.from_numpy(g);opt.step()
    exact(opt.state[p]['exp_avg'],m)
    exact(opt.state[p]['exp_avg_var'],v)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('betas',[NOMINAL,LEGACY_F32],ids=['nominal-native','explicit-legacy-f32'])
def test_source_product_then_add_is_distinct_from_alpha_fusion(dtype,betas):
    # Independent generated operands expose add_(alpha=...) fusion rounding.
    # TF source explicitly multiplies each operand before the sum/assignment.
    scalar=np.float32 if dtype==torch.float32 else np.float64
    rng=np.random.default_rng(4721)
    m=rng.normal(size=256).astype(scalar)
    g=rng.normal(size=256).astype(scalar)
    v=np.abs(rng.normal(size=256)).astype(scalar)
    expected_m,expected_v=source_moments(m,v,g,betas)
    b1=scalar(betas[0]);c1=scalar(1)-b1
    fused=torch.from_numpy(m.copy()).mul_(float(b1)).add_(torch.from_numpy(g),alpha=float(c1))
    assert not torch.equal(fused,torch.from_numpy(expected_m)), 'control must discriminate operation order'
    p=torch.nn.Parameter(torch.zeros(256,dtype=dtype))
    opt=AdaBeliefOptimizer([p],betas=betas)
    opt.state[p].update(exp_avg=torch.from_numpy(m.copy()),exp_avg_var=torch.from_numpy(v.copy()),step=0)
    p.grad=torch.from_numpy(g);opt.step()
    exact(opt.state[p]['exp_avg'],expected_m)
    exact(opt.state[p]['exp_avg_var'],expected_v)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('rectify',[False,True])
@pytest.mark.parametrize('amsgrad',[False,True])
@pytest.mark.parametrize('initial',[0.,.25])
@pytest.mark.parametrize('betas',[NOMINAL,LEGACY_F32],ids=['nominal-native','explicit-legacy-f32'])
def test_source_moments_maximum_updated_residual_and_short_trajectory(dtype,rectify,amsgrad,initial,betas):
    scalar=np.float32 if dtype==torch.float32 else np.float64
    p=torch.nn.Parameter(torch.full((5,),initial,dtype=dtype))
    opts=dict(rectify=rectify,amsgrad=amsgrad)
    if betas!=NOMINAL:opts['betas']=betas
    opt=AdaBeliefOptimizer([p],**opts)
    m=np.zeros(5,dtype=scalar);v=m.copy();maximum=m.copy()
    expected=p.detach().clone()
    for step in range(1,25):
        g=np.array([0.,1e-12,.125*(-1)**step,.75/(step+1),-1.25],dtype=scalar)
        old_m=m.copy()
        m,v=source_moments(m,v,g,betas)
        if step==1:
            b2=scalar(betas[1])
            wrong=(scalar(1)-b2)*np.square(g-old_m)+scalar(1e-14)
            assert not np.array_equal(v,wrong), 'updated-m residual must be discriminating'
        maximum=np.maximum(maximum,v)
        update=direction(torch.from_numpy(m.copy()),torch.from_numpy((maximum if amsgrad else v).copy()),dtype,step,rectify,betas)
        expected=expected-.001*update
        p.grad=torch.from_numpy(g);opt.step()
        state=opt.state[p]
        exact(state['exp_avg'],m);exact(state['exp_avg_var'],v)
        if amsgrad:exact(state['max_exp_avg_var'],maximum)
        # Separate-product moment slots are exact. Unchanged native parameter
        # add_(alpha=-lr) is compared numerically, not silently redefined here.
        torch.testing.assert_close(p,expected,rtol=4e-7 if dtype==torch.float32 else 2e-12,atol=1e-10 if dtype==torch.float32 else 1e-16)
        assert p[0]==initial


@pytest.mark.parametrize('rectify',[False,True])
@pytest.mark.parametrize('amsgrad',[False,True])
def test_float32_nominal_and_dtype_materialized_groups_are_bitwise_equal(rectify,amsgrad):
    p=torch.nn.Parameter(torch.zeros(4));q=torch.nn.Parameter(torch.zeros(4))
    opt=AdaBeliefOptimizer([{'params':[p]},{'params':[q],'betas':LEGACY_F32}],rectify=rectify,amsgrad=amsgrad)
    for step in range(1,25):
        p.grad=q.grad=torch.tensor([0.,1e-12,.1*(-1)**step,.5])
        opt.step();exact(p,q)
        for name,value in opt.state[p].items():
            if isinstance(value,torch.Tensor):exact(value,opt.state[q][name])
            else:assert value==opt.state[q][name]


def test_float64_caller_precision_is_not_unconditionally_rounded_to_legacy_f32():
    p=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64));q=torch.nn.Parameter(torch.zeros_like(p))
    opt=AdaBeliefOptimizer([{'params':[p]},{'params':[q],'betas':LEGACY_F32}])
    g=np.array([.25,0.,-1.],dtype=np.float64)
    p.grad=q.grad=torch.from_numpy(g);opt.step()
    for parameter,betas in ((p,NOMINAL),(q,LEGACY_F32)):
        m,v=source_moments(np.zeros_like(g),np.zeros_like(g),g,betas)
        exact(opt.state[parameter]['exp_avg'],m);exact(opt.state[parameter]['exp_avg_var'],v)
    assert not torch.equal(opt.state[p]['exp_avg'],opt.state[q]['exp_avg'])
    assert not torch.equal(opt.state[p]['exp_avg_var'],opt.state[q]['exp_avg_var'])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_nominal_sparse_observed_rows_keep_dense_momentum_and_epsilon(dtype):
    scalar=np.float32 if dtype==torch.float32 else np.float64
    p=torch.nn.Parameter(torch.ones(3,2,dtype=dtype))
    opt=AdaBeliefOptimizer([p],amsgrad=True)
    m=np.zeros((3,2),dtype=scalar);v=m.copy();maximum=m.copy()
    for step in range(1,15):
        row=step%2
        g=np.zeros_like(m);g[row]=[.25,-.1*step]
        observed=np.zeros_like(m,dtype=bool);observed[row]=True
        m,v=source_moments(m,v,g,observed=observed);maximum=np.maximum(maximum,v)
        before=p.detach().clone()
        p.grad=torch.sparse_coo_tensor(torch.tensor([[row]]),torch.from_numpy(g[row:row+1]),p.shape)
        opt.step()
        exact(opt.state[p]['exp_avg'],m);exact(opt.state[p]['exp_avg_var'],v)
        exact(opt.state[p]['max_exp_avg_var'],maximum)
        exact(p[~torch.from_numpy(observed)],before[~torch.from_numpy(observed)])


@pytest.mark.parametrize('rectify',[False,True])
def test_current_checkpoint_continuation_with_mixed_dtype_schedule_clipping_decay(rectify):
    params=[torch.nn.Parameter(torch.ones(2,3,dtype=d)) for d in (torch.float32,torch.float64)*2]
    def make(values):
        return AdaBeliefOptimizer([{'params':values[:2]},{'params':values[2:],'amsgrad':True}],
            rectify=rectify,total_steps=20,warmup_proportion=.2,min_lr=.0001,decay=.03,
            weight_decay=.01,clipvalue=.3,clipnorm=.5,gradient_centralization=True)
    opt=make(params);resumed=None
    for step in range(1,23):
        for i,group in enumerate(opt.param_groups):group['lr']=.002*(i+1)/(step+1)
        if resumed is not None:
            for a,b in zip(resumed.param_groups,opt.param_groups):a['lr']=b['lr']
        for i,p in enumerate(params):
            p.grad=None if i==0 and step%3==0 else torch.cos(torch.arange(6,dtype=p.dtype).reshape(2,3)+step)
        if resumed is not None:
            for p,q in zip(params,copies):q.grad=None if p.grad is None else p.grad.clone()
        opt.step()
        if resumed is not None:
            resumed.step()
            for p,q in zip(params,copies):
                exact(p,q)
                for key,value in opt.state[p].items():
                    if isinstance(value,torch.Tensor):exact(value,resumed.state[q][key])
                    else:assert value==resumed.state[q][key]
        if step in (5,9):
            buffer=io.BytesIO();torch.save(opt.state_dict(),buffer);buffer.seek(0)
            copies=[torch.nn.Parameter(p.detach().clone()) for p in params]
            resumed=make(copies);resumed.load_state_dict(torch.load(buffer,weights_only=True))
