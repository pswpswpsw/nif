"""Read-only native diagnostics derived from public TF baseline mathematics.

This is NOT a TensorFlow execution or controlled training parity gate. No
implementation, notebook, checkpoint, or reference-source writes occur.
"""
import hashlib
import json
import platform
from pathlib import Path

import nbformat
import numpy as np
import torch
from torch import nn

import nif
from nif.demo import CylinderFlow
from nif.optimizers import AdaBeliefOptimizer

ROOT = Path(__file__).resolve().parents[1]


def canonical_parameters(model):
    # Only actual nn.Linear kernels transpose. Custom kernels already store
    # [input, output]. Every bias, including the final additive bias, is copied.
    result = {}
    for name, p in model.named_parameters():
        value = p.T if name.endswith('.linear.weight') else p
        result[name] = value.detach().clone().requires_grad_()
    assert len(result) == 19
    return result


def reference_forward(x, weights):
    def linear(x, key):
        return x @ weights[key + '.linear.weight'] + weights[key + '.linear.bias']
    def swish(x):
        return x * x.sigmoid()
    trace = {}
    temporal = swish(linear(x[:, :1], 'pnet_list.0'))
    trace['pnet_list.0'] = temporal
    for i in (1, 2):
        key = f'pnet_list.{i}.L1'
        temporal = temporal + swish(temporal @ weights[key + '.w'] + weights[key + '.b'])
        trace[f'pnet_list.{i}'] = temporal
    temporal = linear(temporal, 'pnet_list.3')
    trace['pnet_list.3'] = temporal
    temporal = temporal @ weights['pnet_list.4.w'] + weights['pnet_list.4.b']
    trace['pnet_list.4'] = temporal
    spatial = x[:, 1:3]
    for i in range(4):
        key = f'snet_list.{i}'
        product = spatial @ weights[key + '.w']
        spatial = torch.sin(30. * product + weights[key + '.b']) if i < 3 else product + weights[key + '.b']
        trace[key] = spatial
    phi = spatial.reshape(-1, 2, 10)
    output = (phi * temporal[:, None, :]).sum(dim=2) + weights['last_bias_layer.last_layer_bias']
    return output, trace


def max_error(a, b):
    return float((a.detach() - b.detach()).abs().max())


def whole_model(scope, data, dtype):
    torch.manual_seed(173)
    model = nif.NIFMultiScaleLastLayerParameterized(scope['cfg_shape_net'], scope['cfg_parameter_net'], dtype)
    weights = canonical_parameters(model)
    rows = np.arange(1024) * 187 % len(data.data)
    x = torch.tensor(data.data[rows, :3], dtype=dtype, requires_grad=True)
    target = torch.tensor(data.data[rows, 3:5], dtype=dtype)
    area = torch.tensor(data.sample_weight[rows].reshape(-1), dtype=dtype)
    trace = {}
    handles = [module.register_forward_hook(lambda m, args, out, name=name: trace.__setitem__(name, out))
               for name, module in model.named_modules() if name in
               [f'pnet_list.{i}' for i in range(5)] + [f'snet_list.{i}' for i in range(4)]]
    prediction = model(x)
    for handle in handles:
        handle.remove()
    rx = x.detach().clone().requires_grad_()
    reference, reference_trace = reference_forward(rx, weights)
    jacobian = torch.stack([torch.autograd.grad(prediction[:, i].sum(), x, retain_graph=True)[0]
                            for i in range(2)], dim=1)
    reference_jacobian = torch.stack([torch.autograd.grad(reference[:, i].sum(), rx, retain_graph=True)[0]
                                      for i in range(2)], dim=1)
    loss = ((prediction-target).square().mean(dim=-1) * area).mean()
    reference_loss = ((reference-target).square().mean(dim=-1) * area).mean()
    native_grad = torch.autograd.grad(loss, [x, *model.parameters()])
    reference_grad = torch.autograd.grad(reference_loss, [rx, *weights.values()])
    for p, grad in zip(model.parameters(), native_grad[1:]):
        p.grad = grad.detach().clone()
    scope['centralize_gradients'](model)
    grad_errors, gc_errors = {}, {}
    for (name, p), actual, expected in zip(model.named_parameters(), native_grad[1:], reference_grad[1:]):
        actual = actual.T if name.endswith('.linear.weight') else actual
        gc = p.grad.T if name.endswith('.linear.weight') else p.grad
        gc_expected = expected - expected.mean(0, keepdim=True) if expected.ndim > 1 else expected
        grad_errors[name] = max_error(actual, expected)
        gc_errors[name] = max_error(gc, gc_expected)
    first = model.pnet_list[0].linear.weight
    assert torch.count_nonzero(first.grad) == 0
    before = first.detach().clone()
    updates = full_parameter_optimizer_probe(model)
    torch.testing.assert_close(first, before, rtol=0, atol=0)
    views = [model.model_p_to_w(), model.model_x_to_phi(), model.model_x_to_u_given_w()]
    shared = {id(p) for v in views for p in v.parameters()}
    assert shared == {id(p) for p in model.parameters()}
    return {
        'parameter_tensors': len(weights), 'parameter_values': sum(p.numel() for p in model.parameters()),
        'semantic_layout': {name: list(p.shape) for name, p in weights.items()},
        'buffer_shapes': {name: list(value.shape) for name, value in model.named_buffers()},
        'output_max_abs_error': max_error(prediction, reference),
        'intermediate_max_abs_error': max(max_error(trace[n], reference_trace[n]) for n in trace),
        'weighted_loss': float(loss.detach()), 'loss_abs_error': max_error(loss, reference_loss),
        'input_loss_gradient_max_abs_error': max_error(native_grad[0], reference_grad[0]),
        'full_input_jacobian_max_abs_error': max_error(jacobian, reference_jacobian),
        'all_parameter_gradient_max_abs_error': max(grad_errors.values()),
        'all_gc_gradient_max_abs_error': max(gc_errors.values()),
        'first_temporal_gc_nonzero': int(torch.count_nonzero(first.grad)),
        'first_temporal_kernel_update_max_abs': max_error(first, before),
        'all_parameters_shared_by_views': True,
        'all_parameter_same_gradient_updates': updates,
    }


def full_parameter_optimizer_probe(model):
    """All 19 actual tensors/GC gradients, shared across 12 update comparisons.

    Freeze the whole-model gradients to isolate optimizer arithmetic from new
    forward/backward differences. This is not a coupled training trajectory.
    """
    named = dict(model.named_parameters())
    gradients = {n: p.grad.clone() for n, p in named.items()}
    q = {n: p.detach().clone() for n, p in named.items()}
    m = {n: torch.zeros_like(p) for n, p in named.items()}
    v = {n: torch.zeros_like(p) for n, p in named.items()}
    dtype = next(model.parameters()).dtype
    b1, b2, eps, lr = [torch.tensor(x, dtype=dtype) for x in (.9, .999, 1e-14, .001)]
    optimizer = AdaBeliefOptimizer(model.parameters(), lr=.001)
    records = []
    for step in range(1, 13):
        t = torch.tensor(step, dtype=dtype)
        b1pow, b2pow = b1**t, b2**t
        inf = 2/(1-b2)-1
        rho = inf-2*t*b2pow/(1-b2pow)
        for name, p in named.items():
            g = gradients[name]
            p.grad = g.clone()
            m[name] = b1*m[name]+(1-b1)*g
            v[name] = b2*v[name]+(1-b2)*(g-m[name]).square()+eps
            direction = m[name]/(1-b1pow)
            if rho >= 5:
                r = torch.sqrt((rho-4)/(inf-4)*(rho-2)/(inf-2)*inf/rho)
                direction = r*direction/(torch.sqrt(v[name]/(1-b2pow))+eps)
            q[name] = q[name]-lr*direction
        optimizer.step()
        records.append({'step': step,
                        'all_parameter_max_abs': max(max_error(p,q[n]) for n,p in named.items()),
                        'all_moment_max_abs': max(max_error(optimizer.state[p]['exp_avg'],m[n]) for n,p in named.items()),
                        'all_variance_max_abs': max(max_error(optimizer.state[p]['exp_avg_var'],v[n]) for n,p in named.items())})
    return records


def optimizer_arithmetic(dtype):
    """Emulate TF source operations in var_dtype, using Torch tensors only.

    Baseline external_optimizers.py lines 456-536 uses tensor-valued powers,
    differences and square roots. This does not emulate TF compiler fusion.
    """
    p = nn.Parameter(torch.tensor([.4, -.2, .1], dtype=dtype))
    q = p.detach().clone()
    optimizer = AdaBeliefOptimizer([p], lr=.001)
    m, v = torch.zeros_like(p), torch.zeros_like(p)
    b1, b2, eps, lr = [torch.tensor(x, dtype=dtype) for x in (.9, .999, 1e-14, .001)]
    records = []
    for step in range(1, 13):
        grad = torch.tensor([.03*step, (-1)**step*.2, 0.], dtype=dtype)
        t = torch.tensor(step, dtype=dtype)
        b1pow, b2pow = b1**t, b2**t
        rho_inf = 2. / (1. - b2) - 1.
        rho = rho_inf - 2.*t*b2pow/(1.-b2pow)
        m = b1*m+(1.-b1)*grad
        v = b2*v+(1.-b2)*(grad-m).square()+eps
        direction = m/(1.-b1pow)
        if rho >= 5.:
            r = torch.sqrt((rho-4.)/(rho_inf-4.)*(rho-2.)/(rho_inf-2.)*rho_inf/rho)
            direction = r*direction/(torch.sqrt(v/(1.-b2pow))+eps)
        old_p, old_q = p.detach().clone(), q.clone()
        q = q-lr*direction
        p.grad = grad.clone()
        optimizer.step()
        state = optimizer.state[p]
        records.append({'step': step, 'tensor_rho': float(rho),
                        'python_rho': 2/(1-.999)-1-2*step*.999**step/(1-.999**step),
                        'parameter_max_abs': max_error(p, q),
                        'update_max_abs': max_error(p-old_p, q-old_q),
                        'update_relative_l2': float(((p.detach()-old_p)-(q-old_q)).norm()/(q-old_q).norm()),
                        'moment_max_abs': max_error(state['exp_avg'], m),
                        'variance_max_abs': max_error(state['exp_avg_var'], v)})
    return records


def main():
    torch.set_num_threads(1)
    nb = nbformat.read(next((ROOT/'tutorial').glob('3_*_pytorch.ipynb')), as_version=4)
    scope = dict(torch=torch, nn=nn, np=np, device=torch.device('cpu'), SMOKE=False,
                 DataLoader=torch.utils.data.DataLoader, TensorDataset=torch.utils.data.TensorDataset)
    exec(nb.cells[2].source, scope)  # helper definitions only, no training
    exec(next(c.source for c in nb.cells if c.cell_type=='code' and 'cfg_shape_net = {' in c.source), scope)
    data = CylinderFlow()
    scope.update(train_data=data.data, sample_weight=data.sample_weight)
    exec(next(c.source for c in nb.cells if c.cell_type=='code' and 'nepoch, lr, batch_size = 600' in c.source), scope)
    loader = scope['train_dataset']
    sampled = []
    class ObserveSampler:
        def __init__(self, source): self.source = source
        def __iter__(self):
            for index in self.source:
                sampled.append(index)
                yield index
        def __len__(self): return len(self.source)
    loader.batch_sampler.sampler = ObserveSampler(loader.batch_sampler.sampler)
    sizes, offset = [], 0
    for features, targets, weights in loader:
        ids = sampled[offset:offset+len(features)]
        torch.testing.assert_close(features, torch.tensor(data.data[ids,:3],dtype=torch.float32),rtol=0,atol=0)
        torch.testing.assert_close(targets, torch.tensor(data.data[ids,3:5],dtype=torch.float32),rtol=0,atol=0)
        torch.testing.assert_close(weights, torch.tensor(data.sample_weight[ids].reshape(-1),dtype=torch.float32),rtol=0,atol=0)
        sizes.append(len(features));offset+=len(features)
    assert sorted(sampled)==list(range(191775)) and sizes==[1024]*187+[287]
    schedule={};lr=.001
    for epoch in range(600):
        lr=scope['scheduler'](epoch,lr)
        if epoch in (0,199,200,399,400,599):schedule[epoch]=lr
    paths = sorted((ROOT/'nif').rglob('*.py')) + [ROOT/'dev_tests/generate_native_notebooks.py',
            next((ROOT/'tutorial').glob('3_*_pytorch.ipynb'))]
    result={'scope':'Native/public-source diagnosis only; no TF execution or parity acceptance',
            'reference_sources':{'tensorflow':'f8f9d31f4aa331b6e7dbc48758302c801441bcce',
                'compatibility':'ad9113df14590189b12068b6b3fa7c8c8d9187d4',
                'native_baseline_from_contract':'220771c6fb5e567fb4e38fa5021253c3682b1cca'},
            'current_source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'environment':{'python':platform.python_version(),'platform':platform.platform(),
                           'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads()},
            'limitations':['References execute explicit public formulas in Torch, not TensorFlow',
                'Tensor scalar emulation does not establish TF compiler fusion or hyperparameter storage/cast behavior',
                'Float64/float32 cases each transfer actual tensors internally; no shared-seed cross-framework claim',
                'Optimizer probes use identical frozen gradients to isolate arithmetic; not coupled training trajectories'],
            'batches':{'rows':offset,'steps':len(sizes),'remainder':sizes[-1],
                       'actual_ids_once_each':True,'schedule':schedule},
            'whole_model':{},'optimizer_same_gradient_diagnostics':{}}
    for dtype in (torch.float64,torch.float32):
        result['whole_model'][str(dtype)]=whole_model(scope,data,dtype)
        result['optimizer_same_gradient_diagnostics'][str(dtype)]=optimizer_arithmetic(dtype)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
