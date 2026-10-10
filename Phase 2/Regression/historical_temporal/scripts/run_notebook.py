"""Execute the companion in the calling Python environment, not a global kernel."""
from pathlib import Path
import json
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
kernel_root = ROOT / '.venv/share/jupyter/kernels'
spec_dir = kernel_root / 'delivery-regression'
spec_dir.mkdir(parents=True, exist_ok=True)
(spec_dir/'kernel.json').write_text(json.dumps({
    'argv':[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name':'Delivery Regression', 'language':'python'}))
path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'notebooks/delivery_regression_phase2.ipynb')
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=1200, kernel_name='delivery-regression',
                        resources={'metadata':{'path':str(ROOT)}},
                        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
client.execute()
nbformat.write(nb,path)
errors=[out for c in nb.cells if c.cell_type=='code' for out in c.get('outputs',[]) if out.output_type=='error']
assert not errors
print('Executed:',sum(c.cell_type=='code' for c in nb.cells),'code cells; no errors',flush=True)
