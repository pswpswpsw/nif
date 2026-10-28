"""Boundary follow-up: actual tutorial 1/3 cells in separate CPU smoke kernels."""
import hashlib
import json
import os
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'dev_tests/artifacts/adabelief_boundary/smoke'
    output.mkdir(parents=True,exist_ok=True)
    os.environ.update(NIF_SMOKE='1',NIF_DEVICE='cpu',NIF_OUTPUT_DIR=str(output),MPLBACKEND='Agg')
    summary=[]
    for number in (1,3):
        path=next((ROOT/'tutorial').glob(f'{number}_*_pytorch.ipynb'))
        notebook=nbformat.read(path,as_version=4)
        result=dict(notebook=path.name,mode='smoke',device='cpu',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        start=time.monotonic()
        try:
            NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
            code=[c for c in notebook.cells if c.cell_type=='code' and c.source.strip()]
            assert all(c.execution_count is not None for c in code)
            metrics=json.loads((output/f'tutorial{number}/metrics.json').read_text())
            if number==1:
                assert metrics['adabelief_epochs']==2 and metrics['lbfgs_rounds']==1 and metrics['lbfgs_max_iter']==2
            else:
                assert metrics['epochs']==2 and metrics['training_rows']==4096 and metrics['all_rows']==191775
            result.update(status='passed',executed_code_cells=len(code),metrics=metrics)
        except Exception as exc:result.update(status='failed',error=str(exc))
        result['seconds']=time.monotonic()-start
        nbformat.write(notebook,output/path.name)
        summary.append(result)
        (ROOT/'dev_tests/adabelief_boundary_smoke_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    if any(r['status']!='passed' for r in summary):raise SystemExit(1)


if __name__=='__main__':main()
