"""Bitwise cache regression against our public, frozen pre-cache implementation.

The fixture was copied after the all-None guard reached GREEN, before caching.
It is implementation-owned source, not an independent reference fixture.
"""
import io
from pathlib import Path
import runpy

import pytest
import torch

from nif.optimizers import ReferenceAdam

BASELINE = Path(__file__).with_name('fixtures') / 'reference_adam_guarded_uncached.py'
UncachedAdam = runpy.run_path(str(BASELINE))['ReferenceAdam']


def assert_bitwise(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert torch.equal(actual.detach().contiguous().reshape(-1).view(torch.uint8),
                           expected.detach().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_bitwise(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_bitwise(a, b)
    else:
        assert actual == expected


def make_optimizer(cls, amsgrad, values=None):
    params = [torch.nn.Parameter(torch.linspace(-.2,.3,7,dtype=dtype))
              for dtype in (torch.float32, torch.float64)*4]
    if values is not None:
        with torch.no_grad():
            for p,v in zip(params,values): p.copy_(v)
    optimizer = cls([
        {'params':params[:4], 'lr':.003, 'betas':(.9,.999)},
        {'params':params[4:], 'lr':.0001, 'betas':(.5,.75), 'eps':2e-7},
    ], amsgrad=amsgrad)
    return params, optimizer


def restore(cls, params, optimizer, amsgrad):
    buffer=io.BytesIO()
    torch.save({'parameters':[p.detach() for p in params],
                'optimizer':optimizer.state_dict()},buffer)
    buffer.seek(0)
    checkpoint=torch.load(buffer, weights_only=True)
    params,optimizer=make_optimizer(cls,amsgrad,checkpoint['parameters'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return params,optimizer


@pytest.mark.parametrize('amsgrad',[False,True])
def test_cache_bitwise_trajectory_slots_groups_and_checkpoint(amsgrad):
    params,optimizer=make_optimizer(ReferenceAdam,amsgrad)
    control,uncached=make_optimizer(UncachedAdam,amsgrad)
    for step in range(32):
        for opt in (optimizer,uncached):
            for index,group in enumerate(opt.param_groups):
                # Same device/dtype across groups but different hyperparameters;
                # changing schedules ensure coefficients cannot persist across calls.
                group['lr']=(.003 if index == 0 else .0001)/(1+step%5)
                if step == 17: group['betas']=(.8,.98) if index == 0 else (.6,.875)
        for i,(p,q) in enumerate(zip(params,control)):
            if (step+i)%7 == 0:
                p.grad=q.grad=None
            else:
                scale=(0.,1e-12,.2)[(step+i)%3]
                # Parameter-dependent gradients exercise the whole trajectory.
                p.grad=(p.detach().sin()+step*.01)*scale
                q.grad=(q.detach().sin()+step*.01)*scale
        optimizer.step();uncached.step()
        assert_bitwise(params,control)
        assert_bitwise(optimizer.state_dict(),uncached.state_dict())
        if step in (5,20):
            params,optimizer=restore(ReferenceAdam,params,optimizer,amsgrad)
            control,uncached=restore(UncachedAdam,control,uncached,amsgrad)
            assert_bitwise(optimizer.state_dict(),uncached.state_dict())
