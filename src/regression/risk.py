"""Optional slow-delivery score built with strictly past-only training blocks."""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from . import config as C, data, models, audit

PARAMS={'loss':'log_loss','max_iter':120,'learning_rate':.05,'max_leaf_nodes':15,
        'min_samples_leaf':50,'l2_regularization':10.,'early_stopping':False,'random_state':C.SEED}

def fit_classifier(frame,cutoff):
    audit.require_pass()
    assert frame.delivered_ts.lt(cutoff).all()
    prep=models.preprocessing('calendar');x=prep.fit_transform(frame)
    y=frame.lead_time_days.gt(30).astype(int)
    if y.nunique()<2:return {'constant':float(y.mean())}
    clf=HistGradientBoostingClassifier(**PARAMS)
    with threadpool_limits(limits=4):clf.fit(x,y)
    return {'prep':prep,'classifier':clf}

def probability(bundle,frame):
    if 'constant' in bundle:return np.full(len(frame),bundle['constant'])
    with threadpool_limits(limits=4):return bundle['classifier'].predict_proba(bundle['prep'].transform(frame))[:,1]

def crossfit(frame,cutoff,key):
    cache=C.PRIVATE/f'risk_{key}.joblib'
    if cache.exists():
        v=joblib.load(cache);assert v['fit_hash']==data.ids_hash(frame.order_id);return v
    p=np.zeros(len(frame));missing=np.ones(len(frame));records=[]
    month=frame.purchase_ts.dt.to_period('M')
    # The first three months are an explicit cold-start block, with no learnt score.
    for m in sorted(month.unique()):
        start=m.start_time
        if start<pd.Timestamp('2017-04-01'):continue
        train=frame[(frame.purchase_ts<start)&(frame.delivered_ts<start)]
        idx=np.flatnonzero(month.eq(m))
        if len(train)<1000:continue
        bundle=fit_classifier(train,start)
        p[idx]=probability(bundle,frame.iloc[idx]);missing[idx]=0
        assert set(train.order_id).isdisjoint(frame.iloc[idx].order_id)
        records.append({'month':str(m),'train_n':len(train),'score_n':len(idx),
            'latest_label':str(train.delivered_ts.max()),'cutoff':str(start),
            'future_training_labels':int(train.delivered_ts.ge(start).sum()),
            'fit_ids_sha256':data.ids_hash(train.order_id)})
    full=fit_classifier(frame,cutoff)
    value={'probability':p,'missing':missing,'bundle':full,'audit':records,
           'cold_start_n':int(missing.sum()),'fit_hash':data.ids_hash(frame.order_id)}
    joblib.dump(value,cache);return value

def enrich_training(frame,cutoff,key):
    v=crossfit(frame,cutoff,key);x=frame.copy()
    x['risk_p30']=v['probability'];x['risk_unavailable']=v['missing']
    return x,v

def enrich_scoring(frame,bundle):
    x=frame.copy();x['risk_p30']=probability(bundle,x);x['risk_unavailable']=0.
    return x
