"""Deployment interface for the frozen regression bundle (no HTTP service)."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from . import config as C, models, risk

def load_bundle(path=None):
    path=Path(path or C.MODELS/'final_model.joblib')
    if not path.exists():raise FileNotFoundError('Model bundle missing. Run the audited pipeline through freeze/test first.')
    return joblib.load(path)

def predict_bundle(frame,bundle):
    if bundle.get('ensemble'):
        return np.column_stack([models.predict(m,frame) for m in bundle['base_models']])@bundle['weights']
    work=risk.enrich_scoring(frame,bundle['risk_bundle']) if bundle.get('pack')=='risk' else frame
    return models.predict(bundle,work)

def predict_lead_time(orders,bundle=None):
    if orders is None or len(orders)==0:raise ValueError('Provide at least one held-out historical order.')
    bundle=bundle or load_bundle()
    required=[c for c in bundle['input_columns'] if c not in ['risk_p30','risk_unavailable']]
    missing=[c for c in required if c not in orders]
    if missing:raise ValueError('Missing purchase-time input columns: '+', '.join(missing))
    if 'order_id' not in orders:raise ValueError('order_id is required for identifying the output rows.')
    for col in ['total_price','total_freight','promised_lead_time_days','max_distance_km','item_count','seller_count']:
        v=pd.to_numeric(orders[col],errors='coerce')
        if np.isinf(v).any():raise ValueError(col+' cannot contain infinite values.')
        if ((v<0)&v.notna()).any():raise ValueError(col+' cannot be negative; missing optional values may remain NaN.')
    if 'purchase_ts' in orders and pd.to_datetime(orders.purchase_ts,errors='coerce').isna().any():
        raise ValueError('purchase_ts must be a valid timestamp.')
    prediction=predict_bundle(orders,bundle)
    result=pd.DataFrame({'order_id':orders.order_id.to_numpy(),'predicted_days':prediction,
                         'quoted_days':orders.promised_lead_time_days.to_numpy()})
    result['predicted_exceeds_quote']=result.predicted_days>result.quoted_days
    if 'purchase_ts' in orders:
        result['estimated_arrival']=pd.to_datetime(orders.purchase_ts).to_numpy()+pd.to_timedelta(prediction,unit='D')
    return result
