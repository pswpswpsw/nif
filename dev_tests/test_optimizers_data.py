"""Self-owned public-contract checks; no independent acceptance fixtures."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
def load(relative):
    spec = importlib.util.spec_from_file_location('development_' + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adabelief_exact_and_resume():
    AdaBelief = load('nif/optimizers/external_optimizers.py').AdaBeliefOptimizer
    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    opt = AdaBelief([p], learning_rate=.1, beta_1=.8, beta_2=.9, epsilon=1e-6, rectify=False, amsgrad=True, weight_decay=.02)
    expected = p.detach().clone(); m = torch.zeros_like(p); v = m.clone(); vmax = m.clone()
    for i, grad in enumerate(([2., -3.], [-1., 4.], [0.5, 0.1]), 1):
        g = torch.tensor(grad, dtype=p.dtype); p.grad = g
        m = .8*m + .2*g
        v = .9*v + .1*(g-m)**2 + 1e-6
        vmax = torch.maximum(vmax, v)
        expected -= .1 * (m/(1-.8**i)/(torch.sqrt(vmax/(1-.9**i))+1e-6) + .02*expected)
        opt.step()
        torch.testing.assert_close(p, expected)
    q = torch.nn.Parameter(p.detach().clone()); resumed = AdaBelief([q]); resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    p.grad = q.grad = torch.tensor([.3, -.4], dtype=p.dtype)
    opt.step(); resumed.step(); torch.testing.assert_close(p, q)


def test_lion_exact():
    Lion = load('nif/optimizers/external_optimizers.py').Lion
    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    opt = Lion([p], lr=.1, betas=(.9, .99), weight_decay=.2)
    expected = p.detach().clone(); m = torch.zeros_like(p)
    for g in (torch.tensor([2., -3.]), torch.tensor([-.1, 4.])):
        p.grad = g.to(p); expected -= .1 * ((.9*m+.1*g).sign()+.2*expected); m=.99*m+.01*g
        opt.step(); torch.testing.assert_close(p, expected)


def test_l4_global_two_steps_and_resume():
    L4 = load('nif/optimizers/external_optimizers.py').L4Adam
    p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64)); q=torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))
    opt = L4([{'params':[p]}, {'params':[q]}], lr=.15, tau_m=10, tau_s=1000, epsilon=1e-7)
    exp = torch.cat([p.detach(),q.detach()]); m=torch.zeros(3,dtype=p.dtype); s=m.clone(); minimum=None
    for t,(loss,g) in enumerate(((4., [2.,-3.,1.]),(3.,[.5,1.,-2.])),1):
        g=torch.tensor(g,dtype=p.dtype); p.grad=g[:2]; q.grad=g[2:]
        minimum=.75*loss if minimum is None else minimum; minimum=min(minimum,loss)
        m=.9*m+.1*g; s=.999*s+.001*g*g
        corrected=m/(1-.9**t); v=corrected/(torch.sqrt(s/(1-.999**t))+1e-7)
        exp-=.15*(loss-.9*minimum)/((corrected*v).sum()+1e-7)*v; minimum*=1.001
        opt.step(loss=torch.tensor(loss)); torch.testing.assert_close(torch.cat([p,q]),exp)
    p2=torch.nn.Parameter(p.detach().clone()); q2=torch.nn.Parameter(q.detach().clone())
    resumed=L4([{'params':[p2]}, {'params':[q2]}]); resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for a,b in ((p,p2),(q,q2)): a.grad=torch.ones_like(a); b.grad=torch.ones_like(b)
    opt.step(loss=2.); resumed.step(loss=2.); torch.testing.assert_close(p,p2); torch.testing.assert_close(q,q2)


def test_normalization_roundtrip_and_zero_range():
    Data = load('nif/data/point_wise_data.py').PointWiseData
    data=np.array([[1.,2.,-4.,1.],[2.,4.,3.,2.],[3.,9.,2.,3.]])
    normalized,mean,scale,weights=Data.minmax_normalize(data,1,1,1,True)
    np.testing.assert_allclose(normalized*scale[:-1]+mean[:-1],data[:,:-1])
    assert weights.mean()==1
    with pytest.raises(ValueError,match='(zero|constant|range)'):
        Data.standard_normalize(np.ones((3,2)))
    with pytest.raises(ValueError):
        Data.minmax_normalize(np.full((3,3),np.nan),1,1,1)


def test_tfrecord_roundtrip_remainder_and_epoch(tmp_path):
    Dataset = load('nif/data/tfr_dataset.py').TFRDataset
    data=np.arange(35,dtype=np.float32).reshape(7,5); data[:,-1]+=1
    np.savez(tmp_path/'data.npz',data=data)
    ds=Dataset(2,2,area_weight=True)
    ds.create_from_npz(3,tmp_path/'data.npz','data',tmp_path/'records','chunk')
    meta=ds.get_tfr_meta_dataset(tmp_path/'records',epoch=2)
    rows=[]
    for shard in meta:
        for x,y,w in ds.gen_dataset_from_batch_file(shard,2):
            rows.append(torch.cat([x,y,w],-1).numpy())
    restored=np.concatenate(rows)
    assert restored.shape==(14,5)
    np.testing.assert_array_equal(restored[np.argsort(restored[:,0])],np.repeat(data,2,axis=0))


def test_centralization_native_transpose_and_real_step():
    from nif.optimizers import centralized_gradients_for_optimizer
    layer=torch.nn.Linear(3,2,bias=False,dtype=torch.float64)
    opt=torch.optim.SGD(layer.parameters(),lr=.1)
    centralized_gradients_for_optimizer(opt)
    grad=torch.tensor([[1.,2.,6.],[-3.,1.,4.]],dtype=torch.float64)
    original=layer.weight.detach().clone(); layer.weight.grad=grad.clone(); opt.step()
    torch.testing.assert_close(layer.weight,original-.1*(grad-grad.mean(1,keepdim=True)))
    # A custom kernel retaining [in,out] storage must reduce over input axis 0.
    custom=torch.nn.Parameter(torch.zeros(3,2,dtype=torch.float64)); custom._nif_gc_axes=(0,)
    opt=torch.optim.SGD([custom],lr=.1); centralized_gradients_for_optimizer(opt)
    custom.grad=grad.T.clone(); opt.step()
    torch.testing.assert_close(custom, -.1*(grad-grad.mean(1,keepdim=True)).T)


def test_lbfgs_pack_closure_and_rounds():
    from nif.optimizers import LBFGSOptimizer, TFPLBFGS, pack_tensors, unpack_tensors
    p=torch.nn.Parameter(torch.tensor([3.,-2.],dtype=torch.float64))
    q=torch.nn.Parameter(torch.tensor([[8.]],dtype=torch.float64))
    values=unpack_tensors([p,q],pack_tensors([p,q]))
    assert values[0].dtype==torch.float64 and values[1].shape==(1,1)
    opt=LBFGSOptimizer(lambda:((p-torch.tensor([1.,2.]))**2).sum()+((q+3)**2).sum(),[p,q],steps=20,verbose=0)
    result=opt.minimize(); assert result.objective_value<1e-15
    torch.testing.assert_close(p,torch.tensor([1.,2.],dtype=p.dtype)); torch.testing.assert_close(q,torch.tensor([[-3.]],dtype=q.dtype))
    model=torch.nn.Linear(1,1,dtype=torch.float64)
    x=torch.linspace(-1,1,20,dtype=torch.float64).reshape(-1,1); y=2*x+1
    lb=TFPLBFGS(model,torch.nn.functional.mse_loss,x,y,display_epoch=0)
    lb.minimize(rounds=2,max_iter=20)
    assert torch.nn.functional.mse_loss(model(x),y)<1e-15
    assert len(lb.history['loss'])>1


def test_legacy_tfrecord_independent_writer_numeric_column_order(tmp_path):
    from tfrecord.writer import TFRecordWriter
    Dataset=load('nif/data/tfr_dataset.py').TFRDataset
    path=tmp_path/'public-format.tfrecord'; writer=TFRecordWriter(str(path))
    # 12 features exercise numeric ordering (input_10 must not precede input_2).
    feature={f'input_{i}':([float(i),float(i+100)],'float') for i in range(12)}
    feature.update(output_0=([21.,22.],'float'),output_1=([31.,32.],'float'),weight=([.5,1.5],'float'))
    writer.write(feature); writer.close()
    ds=Dataset(12,2,True)
    shard=next(iter(ds.get_tfr_meta_dataset(tmp_path,1)))
    x,y,w=next(iter(ds.gen_dataset_from_batch_file(shard,3,shuffle=False)))
    torch.testing.assert_close(x[0],torch.arange(12,dtype=torch.float32))
    torch.testing.assert_close(y,torch.tensor([[21.,31.],[22.,32.]]))
    torch.testing.assert_close(w,torch.tensor([[.5],[1.5]]))
    raw=bytearray(path.read_bytes()); raw[-1]^=1; path.write_bytes(raw)
    with pytest.raises(ValueError,match='checksum'):
        list(ds.get_tfr_meta_dataset(tmp_path,1))


def test_optimizer_training_finite_and_l4_closure():
    from nif.optimizers import AdaBeliefOptimizer, Lion, L4Adam
    for cls,opts in [(AdaBeliefOptimizer,{'lr':.03}),(Lion,{'lr':.03}),(L4Adam,{})]:
        p=torch.nn.Parameter(torch.tensor([3.,-2.]))
        opt=cls([p],**opts); initial=(p*p).sum().item()
        for _ in range(60):
            opt.zero_grad(); loss=p.square().sum(); loss.backward()
            if cls is L4Adam: opt.step(loss=loss)
            else: opt.step()
        assert torch.isfinite(p).all() and p.square().sum()<initial
    p=torch.nn.Parameter(torch.tensor([3.])); opt=L4Adam([p]); opt.step(lambda:p.square().sum()); assert p.item()<3


def test_demo_data_physical_inverse_and_weights():
    from nif.demo import TravelingWave, TravelingWaveHighFreq, CylinderFlow
    for cls in (TravelingWave, TravelingWaveHighFreq, CylinderFlow):
        data=cls(); offset=data.n_p+data.n_x
        recovered=data.inverse_normalize(torch.from_numpy(data.u),data.mean[offset:offset+data.n_o],data.std[offset:offset+data.n_o])
        np.testing.assert_allclose(recovered.numpy(),data.data_raw[:,offset:offset+data.n_o],atol=1e-12)
        if cls is CylinderFlow:
            assert len(data.data)==191775
            np.testing.assert_allclose(data.sample_weight.mean(),1)


def test_tfrecord_external_reader_and_multiworker_epochs(tmp_path):
    from nif.data import TFRDataset
    from tfrecord.reader import tfrecord_loader
    data=np.arange(65,dtype=np.float32).reshape(13,5)
    np.savez(tmp_path/'rows.npz',data=data)
    ds=TFRDataset(2,2,True)
    paths=ds.create_from_npz(3,tmp_path/'rows.npz','data',tmp_path/'shards','public')
    records=[]
    for path in paths:
        for record in tfrecord_loader(path,None):
            records.append(np.stack([record[key] for key in ['input_0','input_1','output_0','output_1','weight']],axis=-1))
    restored=np.concatenate(records); np.testing.assert_array_equal(restored[np.argsort(restored[:,0])],data)
    stream=torch.utils.data.DataLoader(ds.get_tfr_meta_dataset(tmp_path/'shards',3,2),batch_size=None,num_workers=2)
    seen=[]
    for shard in stream:
        for x,y,w in ds.gen_dataset_from_batch_file(shard,4):
            seen.append(torch.cat([x,y,w],-1).numpy())
    restored=np.concatenate(seen)
    np.testing.assert_array_equal(restored[np.argsort(restored[:,0])],np.repeat(data,3,axis=0))


def test_lion_legacy_inverse_time_decay():
    from nif.optimizers import Lion
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    opt=Lion([p],lr=.1,decay=.5)
    p.grad=torch.ones_like(p); opt.step(); torch.testing.assert_close(p,torch.tensor([.9],dtype=p.dtype))
    p.grad=torch.ones_like(p); opt.step(); torch.testing.assert_close(p,torch.tensor([.9-.1/1.5],dtype=p.dtype))


def test_l4_missing_gradient_and_group_state():
    from nif.optimizers import L4Adam
    p=torch.nn.Parameter(torch.tensor([3.])); q=torch.nn.Parameter(torch.tensor([4.]))
    opt=L4Adam([{'params':[p], 'tau_m':5.}, {'params':[q], 'tau_m':10.}])
    p.grad=torch.ones_like(p); opt.step(loss=5.)
    assert q not in opt.state and q.item()==4
    q.grad=torch.ones_like(q); p.grad=None; opt.step(loss=4.)
    assert opt.state[q]['step']==1 and opt.state[p]['step']==1
    assert opt.state['global']['step']==2


def test_adabelief_rectification_warmup_decay_clipping_reference():
    from nif.optimizers import AdaBeliefOptimizer
    p=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64))
    opt=AdaBeliefOptimizer([p],lr=.2,betas=(.7,.98),eps=1e-5,rectify=True,
                          total_steps=12,warmup_proportion=.25,min_lr=.01,
                          decay=.2,clipnorm=2.,clipvalue=1.5)
    expected=p.detach().clone(); m=torch.zeros_like(p); v=m.clone()
    for t in range(1,16):
        grad=torch.tensor([3.+t/10,-2.],dtype=p.dtype); p.grad=grad.clone()
        grad=grad.clamp(-1.5,1.5); grad=grad*min(1.,2./float(grad.norm()))
        lr=.2/(1+.2*(t-1))
        lr=lr*t/3 if t<=3 else lr+(.01-lr)*min(t-3,9)/9
        m=.7*m+.3*grad; v=.98*v+.02*(grad-m).square()+1e-5
        rho_inf=2/.02-1; rho=rho_inf-2*t*.98**t/(1-.98**t)
        direction=m/(1-.7**t)
        if rho>=5:
            factor=((rho-4)/(rho_inf-4)*(rho-2)/(rho_inf-2)*rho_inf/rho)**.5
            direction=factor*direction/(torch.sqrt(v/(1-.98**t))+1e-5)
        expected-=lr*direction; opt.step(); torch.testing.assert_close(p,expected)


def test_sparse_adabelief_observed_rows_and_lion_momentum():
    from nif.optimizers import AdaBeliefOptimizer, Lion
    for cls in (AdaBeliefOptimizer,Lion):
        p=torch.nn.Parameter(torch.ones(3,2,dtype=torch.float64)); opt=cls([p],lr=.1)
        p.grad=torch.sparse_coo_tensor(torch.tensor([[1]]),torch.tensor([[2.,-3.]],dtype=p.dtype),p.shape)
        opt.step()
        assert torch.isfinite(p).all()
        torch.testing.assert_close(p[0],torch.ones(2,dtype=p.dtype))
        torch.testing.assert_close(p[2],torch.ones(2,dtype=p.dtype))
        assert not torch.equal(p[1],torch.ones(2,dtype=p.dtype))


def test_optimizer_clipnorm_is_per_tensor_and_clipvalue_precedes_norm():
    from nif.optimizers import AdaBeliefOptimizer
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64)); q=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
    opt=AdaBeliefOptimizer([p,q],lr=.1,clipnorm=2.)
    p.grad=torch.tensor([3.,4.],dtype=p.dtype); q.grad=torch.tensor([0.,12.],dtype=q.dtype)
    # First rectified-AdaBelief step uses the unbiased gradient directly.
    opt.step()
    torch.testing.assert_close(p,torch.tensor([-.12,-.16],dtype=p.dtype),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(q,torch.tensor([0.,-.2],dtype=q.dtype),rtol=1e-12,atol=1e-12)
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64)); q=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
    opt=AdaBeliefOptimizer([{'params':[p],'clipnorm':1.}, {'params':[q],'clipnorm':2.}],lr=.1,clipvalue=1.5)
    p.grad=torch.tensor([3.,4.],dtype=p.dtype); q.grad=torch.tensor([0.,12.],dtype=q.dtype)
    opt.step()
    torch.testing.assert_close(p,torch.tensor([-.1/2**.5,-.1/2**.5],dtype=p.dtype),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(q,torch.tensor([0.,-.15],dtype=q.dtype),rtol=1e-12,atol=1e-12)


def test_optimizer_global_clipnorm_spans_parameter_groups():
    from nif.optimizers import AdaBeliefOptimizer
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64)); q=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
    opt=AdaBeliefOptimizer([{'params':[p]}, {'params':[q]}],lr=.1,global_clipnorm=2.)
    p.grad=torch.tensor([3.,4.],dtype=p.dtype); q.grad=torch.tensor([0.,12.],dtype=q.dtype)
    opt.step()
    torch.testing.assert_close(p,torch.tensor([-.6/13,-.8/13],dtype=p.dtype),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(q,torch.tensor([0.,-2.4/13],dtype=q.dtype),rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError):
        AdaBeliefOptimizer([p],clipnorm=1.,global_clipnorm=1.)
    with pytest.raises(ValueError):
        AdaBeliefOptimizer([p],clipvalue=-1.)


def test_gtcf_preserves_historical_global_norm_then_value_order():
    from nif.optimizers import AdaBeliefOptimizer, get_centralized_gradients
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64)); q=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
    opt=AdaBeliefOptimizer([p,q],clipnorm=2.,clipvalue=.5)
    loss=(p*torch.tensor([3.,4.])).sum()+(q*torch.tensor([0.,12.])).sum()
    grads=get_centralized_gradients(opt,loss,[p,q])
    # The historical helper explicitly forms sqrt(5**2+12**2)=13 first,
    # unlike ordinary OptimizerV2 clipping. It then clamps component values.
    torch.testing.assert_close(grads[0],torch.tensor([6/13,.5],dtype=p.dtype),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(grads[1],torch.tensor([0.,.5],dtype=q.dtype),rtol=1e-12,atol=1e-12)


def test_gtcf_global_clipping_tiny_gradients_has_no_added_epsilon():
    from nif.optimizers import AdaBeliefOptimizer, get_centralized_gradients
    p=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64)); q=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    opt=AdaBeliefOptimizer([p,q],clipnorm=1e-20)
    grads=get_centralized_gradients(opt,(p*3e-18+q*4e-18).sum(),[p,q])
    torch.testing.assert_close(grads[0],torch.tensor([6e-21],dtype=p.dtype),rtol=1e-12,atol=0)
    torch.testing.assert_close(grads[1],torch.tensor([8e-21],dtype=q.dtype),rtol=1e-12,atol=0)
