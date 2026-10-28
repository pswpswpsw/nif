"""Public-workload structural checks; runtime checks run with nbclient separately."""
import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]

def test_eight_native_notebooks_compile_and_are_native():
    paths = sorted((ROOT / 'tutorial').glob('*_pytorch.ipynb'))
    assert len(paths) == 8, 'All eight native tutorials must exist'
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        source = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith(('tensorflow', 'keras')) for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(('tensorflow', 'keras'))
        assert 'NIF_SMOKE' in source
        assert 'torch.save' in source
        assert 'torch.testing.assert_close' in source

def test_toy_dense_models_preserve_glorot_zero_bias_initialization():
    import torch
    for number in (4, 5):
        path=next((ROOT / 'tutorial').glob(f'{number}_*_pytorch.ipynb'))
        notebook=nbformat.read(path,as_version=4)
        scope={'torch':torch,'nn':torch.nn,'device':torch.device('cpu')}
        exec(notebook.cells[2].source,scope)
        creation=next(c.source for c in notebook.cells if c.cell_type=='code'
                      and c.source.startswith(('model = nn.Sequential','model=nn.Sequential',
                                               'model = dense_tanh','model=dense_tanh')))
        exec(creation,scope)
        linear=scope['model'][0]
        torch.testing.assert_close(linear.bias,torch.zeros_like(linear.bias),rtol=0,atol=0)
        bound=(6/(linear.in_features+linear.out_features))**0.5
        assert linear.weight.abs().max()<=bound

def test_gradient_centralization_respects_native_and_tf_kernel_layouts():
    import torch
    path=next((ROOT/'tutorial').glob('3_*_pytorch.ipynb'))
    notebook=nbformat.read(path,as_version=4)
    scope={'torch':torch,'nn':torch.nn,'device':torch.device('cpu')}
    exec(notebook.cells[2].source,scope)
    model=torch.nn.Module()
    model.register_parameter('w',torch.nn.Parameter(torch.zeros(2,3)))
    model.linear=torch.nn.Linear(2,3)
    gradient=torch.tensor([[1.,3.,7.],[3.,9.,13.]])
    model.w.grad=gradient.clone()
    model.linear.weight.grad=gradient.T.contiguous()
    model.linear.bias.grad=torch.tensor([1.,2.,3.])
    scope['centralize_gradients'](model)
    expected=gradient-gradient.mean(0,keepdim=True)
    torch.testing.assert_close(model.w.grad,expected)
    torch.testing.assert_close(model.linear.weight.grad,expected.T)
    torch.testing.assert_close(model.linear.bias.grad,torch.tensor([1.,2.,3.]))


def test_smoke_is_opt_in_and_preserves_full_epoch_counts():
    import torch
    notebook=nbformat.read(next((ROOT/'tutorial').glob('1_*_pytorch.ipynb')),as_version=4)
    scope={'torch':torch,'nn':torch.nn,'device':torch.device('cpu'),'SMOKE':False}
    exec(notebook.cells[2].source,scope)
    for value in (600,3000,5000,50000):assert scope['epochs'](value)==value
    scope['SMOKE']=True
    assert scope['epochs'](50000)==2
