"""Public atomic-empty-call contracts; existing per-group clocks remain native."""
import copy
import io

import pytest
import torch
from nif.optimizers import AdaBeliefOptimizer


def equal(actual,expected):
    if isinstance(actual,torch.Tensor):
        assert actual.dtype==expected.dtype and actual.device==expected.device
        assert torch.equal(actual.detach().reshape(-1).view(torch.uint8),expected.detach().reshape(-1).view(torch.uint8))
    elif isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual:equal(actual[key],expected[key])
    elif isinstance(actual,(tuple,list)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):equal(a,b)
    else:assert actual==expected


@pytest.mark.parametrize('warmup',[0,3,6])
@pytest.mark.parametrize('closure_clears',[False,True])
def test_empty_call_atomic_before_group_counters_slots_and_schedules(warmup,closure_clears):
    p=torch.nn.Parameter(torch.tensor([.1,-.2],dtype=torch.float32))
    q=torch.nn.Parameter(torch.tensor([.3],dtype=torch.float64))
    calls=[]
    def lr(t):calls.append(('lr',t));return .001
    def wd(t):calls.append(('wd',t));return .01
    opt=AdaBeliefOptimizer([{'params':[p]},{'params':[q]}],lr=lr,weight_decay=wd,amsgrad=True)
    for _ in range(warmup):
        p.grad=torch.ones_like(p);q.grad=torch.zeros_like(q);opt.step()
    opt.zero_grad(set_to_none=True)
    before=copy.deepcopy(opt.state_dict());values=[p.detach().clone(),q.detach().clone()];schedule_calls=calls.copy()
    seen=[]
    def closure():
        assert torch.is_grad_enabled()
        opt.zero_grad(set_to_none=True);seen.append(True)
        return p.square().sum()
    if closure_clears:p.grad=torch.ones_like(p)
    with pytest.raises(ValueError,match='[Nn]o gradients'):
        opt.step(closure if closure_clears else None)
    equal(opt.state_dict(),before);equal([p,q],values)
    assert calls==schedule_calls and seen==([True] if closure_clears else [])


def test_empty_call_cannot_change_next_real_update_or_restored_continuation():
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float32))
    q=torch.nn.Parameter(p.detach().clone())
    opt=AdaBeliefOptimizer([p]);control=AdaBeliefOptimizer([q])
    for step in range(1,10):
        if step==6:
            opt.zero_grad(set_to_none=True)
            try:opt.step()
            except ValueError:pass
            buffer=io.BytesIO();torch.save(opt.state_dict(),buffer);buffer.seek(0)
            opt=AdaBeliefOptimizer([p]);opt.load_state_dict(torch.load(buffer,weights_only=True))
        p.grad=q.grad=torch.tensor([1e-12,-.2])
        opt.step();control.step()
        equal(p,q);equal(opt.state_dict(),control.state_dict())


def test_partial_none_advances_fixed_groups_and_zero_tensor_is_real_update():
    p=torch.nn.Parameter(torch.tensor([1.]))
    q=torch.nn.Parameter(torch.tensor([2.]))
    opt=AdaBeliefOptimizer([{'params':[p]},{'params':[q]}])
    p.grad=torch.zeros_like(p);opt.step()
    assert [g['step'] for g in opt.param_groups]==[1,1]
    assert p in opt.state and q not in opt.state
    before=p.detach().clone();slots=copy.deepcopy(opt.state[p])
    p.grad=None;q.grad=torch.ones_like(q);opt.step()
    assert [g['step'] for g in opt.param_groups]==[2,2]
    assert opt.state[q]['step']==2
    equal(p,before);equal(opt.state[p],slots)


def test_late_added_group_retains_its_own_clock():
    p=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    q=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    opt=AdaBeliefOptimizer([p])
    for _ in range(4):p.grad=torch.ones_like(p);opt.step()
    opt.add_param_group({'params':[q]})
    p.grad=None;q.grad=torch.ones_like(q);opt.step()
    assert [g['step'] for g in opt.param_groups]==[5,1]
    assert opt.state[q]['step']==1 and opt.state[p]['step']==4
    equal(q,torch.tensor([-.001],dtype=q.dtype))
    opt.zero_grad(set_to_none=True)
    before=copy.deepcopy(opt.state_dict())
    with pytest.raises(ValueError,match='[Nn]o gradients'):opt.step()
    equal(opt.state_dict(),before)
    p.grad=torch.zeros_like(p);opt.step()
    assert [g['step'] for g in opt.param_groups]==[6,2]


def test_closure_can_create_real_gradient_before_guard():
    p=torch.nn.Parameter(torch.tensor([1.]))
    opt=AdaBeliefOptimizer([p])
    def closure():
        loss=p.square().sum();loss.backward();return loss
    loss=opt.step(closure)
    assert loss.requires_grad and p.item()<1 and opt.param_groups[0]['step']==1
