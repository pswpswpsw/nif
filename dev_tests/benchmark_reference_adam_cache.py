"""CPU-only execution-cost comparison with the frozen guarded, uncached source."""
import json
from pathlib import Path
import platform
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.benchmark import Timer
from nif.optimizers import ReferenceAdam

UncachedAdam=runpy.run_path(str(ROOT/'dev_tests/fixtures/reference_adam_guarded_uncached.py'))['ReferenceAdam']


def make_optimizer(cls,count,width):
    params=[torch.nn.Parameter(torch.zeros(width,dtype=torch.float32,device='cpu')) for _ in range(count)]
    for p in params: p.grad=torch.full_like(p,.01)
    opt=cls(params)
    for _ in range(10): opt.step()
    return opt


def main():
    torch.set_num_threads(1)
    results=[]
    for count,width in ((64,64),(512,4)):
        optimizers={name:make_optimizer(cls,count,width)
                    for name,cls in (('uncached',UncachedAdam),('cached',ReferenceAdam))}
        timings={name:[] for name in optimizers}
        for repeat in range(3):
            order=('uncached','cached') if repeat%2==0 else ('cached','uncached')
            for name in order:
                result=Timer('optimizer.step()',globals={'optimizer':optimizers[name]},num_threads=1).blocked_autorange(min_run_time=.5)
                timings[name].append(result.median)
        import statistics
        medians={name:statistics.median(values) for name,values in timings.items()}
        results.append(dict(parameters=count,scalars_per_parameter=width,dtype='float32',
                            per_repeat_median_seconds=timings,median_seconds=medians,
                            uncached_over_cached=medians['uncached']/medians['cached']))
    output=dict(python=platform.python_version(),torch=torch.__version__,device='cpu',threads=1,
                note='Local CPU microbenchmark, not GPU speed or full-training evidence.',results=results)
    (ROOT/'dev_tests/reference_adam_cache_benchmark.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
