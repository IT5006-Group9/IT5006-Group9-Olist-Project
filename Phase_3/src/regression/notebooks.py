"""Execute the two reading notebooks with an isolated temporary kernel spec."""
import sys,os,json,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from . import config as C,data

def run():
    records=[]
    with tempfile.TemporaryDirectory(prefix='it5006-r3-kernel-') as folder:
        spec=Path(folder)/'kernels/it5006_r3';spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'IT5006 Regression','language':'python'}))
        previous=os.environ.get('JUPYTER_PATH');os.environ['JUPYTER_PATH']=folder
        try:
            for p in sorted((C.ROOT/'notebooks/regression').glob('*.ipynb')):
                nb=nbformat.read(p,as_version=4)
                NotebookClient(nb,timeout=300,kernel_name='it5006_r3',resources={'metadata':{'path':str(C.ROOT)}}).execute()
                errors=[o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
                assert not errors
                images=sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))
                nbformat.write(nb,p)
                records.append({'notebook':p.name,'code_cells':sum(c.cell_type=='code' for c in nb.cells),
                    'errors':len(errors),'rendered_images':images})
                print(records[-1],flush=True)
        finally:
            if previous is None:os.environ.pop('JUPYTER_PATH',None)
            else:os.environ['JUPYTER_PATH']=previous
    data.save_json(C.OUT/'notebook_execution.json',records)

if __name__=='__main__':run()
