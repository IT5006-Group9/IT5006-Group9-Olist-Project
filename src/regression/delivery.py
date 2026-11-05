"""Build and independently load the local model handoff; no model fitting."""
from pathlib import Path
import json,zipfile,tempfile,subprocess,sys,shutil
from . import config as C,data

def run():
    assert json.loads((C.OUT/'verification.json').read_text())['passed']
    destination=C.MODELS/'regression_deployment_bundle.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['regression','common']:
            for p in (C.ROOT/'src'/name).glob('*.py'):
                z.write(p,str(p.relative_to(C.ROOT)))
        for p in [C.MODELS/'final_model.joblib',C.PRIVATE/'deployment_orders.pkl',C.PRIVATE/'deployment_expected.csv',
                  C.ROOT/'requirements-regression.lock.txt',C.OUT/'frozen_selection.json',C.OUT/'test_summary.json']:
            z.write(p,str(p.relative_to(C.ROOT)))
        z.writestr('README.md','''# Frozen regression handoff

Use Python 3.12 and install requirements-regression.lock.txt. This bundle holds the frozen chronological model and 20 held-out purchase-input examples. It contains no receipt/review target in the input example table.

```python
import sys
sys.path.insert(0, 'src')
import pandas as pd
from regression.score import load_bundle, predict_lead_time
orders = pd.read_pickle('data/regression/deployment_orders.pkl')
predictions = predict_lead_time(orders, load_bundle())
print(predictions.head())
```

Compare predicted_days to data/regression/deployment_expected.csv before app integration. See the source repository's docs/regression/deployment_handoff.md for interpretation, inputs and error handling. This is a purchase-time point estimate, not a guaranteed arrival date or calibrated lateness probability.
''')
    with tempfile.TemporaryDirectory(prefix='it5006-r3-bundle-') as folder:
        with zipfile.ZipFile(destination) as z:z.extractall(folder)
        code="""import sys
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from regression.score import load_bundle,predict_lead_time
orders=pd.read_pickle('data/regression/deployment_orders.pkl')
assert not {'lead_time_days','delivered_ts','review_score','order_status'} & set(orders.columns)
actual=predict_lead_time(orders,load_bundle())
expected=pd.read_csv('data/regression/deployment_expected.csv')
assert actual.order_id.tolist()==expected.order_id.tolist()
np.testing.assert_allclose(actual.predicted_days,expected.predicted_days,rtol=1e-12,atol=1e-12)
print('20 isolated bundle predictions match the frozen evidence.')
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=folder,text=True,capture_output=True,check=True)
    data.save_json(C.OUT/'deployment_bundle_check.json',{'passed':True,'golden_n':20,'isolated_process':True,
        'raw_course_data_needed_for_inference':False,'bundle_sha256':data.digest(destination),
        'bundle_bytes':destination.stat().st_size,'result':result.stdout.strip(),
        'publication':'Local handoff only; model and per-order example records remain git-ignored.'})
    print(destination);print(result.stdout)

if __name__=='__main__':run()
