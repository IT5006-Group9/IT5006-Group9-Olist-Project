"""Expanding windows with an embargo AND observed-label eligibility."""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from . import config as C

@dataclass
class Fold:
    name:str
    cutoff:pd.Timestamp
    end:pd.Timestamp
    train:np.ndarray
    score:np.ndarray
    all_score:np.ndarray

def gap_spec(df):
    # This calibration could have been performed before the first deployment date.
    ref=df[(df.purchase_ts<C.MONTHS[0]+'-01')&(df.delivered_ts<pd.Timestamp(C.MONTHS[0]+'-01'))]
    return {'reference':'2017 purchases delivered before 2018-01-01', 'reference_n':len(ref),
        'q95_days':int(math.ceil(ref.lead_time_days.quantile(.95))),
        'q99_days':int(math.ceil(ref.lead_time_days.quantile(.99))),
        'limitation':'Quantiles describe already observed outcomes, not the unobserved pending tail.'}

def window(df,name,start,end,gap_days,selection=True):
    cutoff=pd.Timestamp(start);end=pd.Timestamp(end)
    train=np.flatnonzero((df.purchase_ts>=C.START)&(df.purchase_ts<cutoff-pd.Timedelta(days=gap_days))&
                        (df.delivered_ts<cutoff))
    membership=(df.purchase_ts>=cutoff)&(df.purchase_ts<end)
    all_score=np.flatnonzero(membership)
    if selection: membership &= df.delivered_ts<C.SELECTION_CUTOFF
    return Fold(name,cutoff,end,train,np.flatnonzero(membership),all_score)

def folds(df,gap_days):
    return [window(df,f'fold{i}',m+'-01',pd.Timestamp(m+'-01')+pd.offsets.MonthBegin(1),gap_days)
            for i,m in enumerate(C.MONTHS,1)]

def final(df,gap_days):return window(df,'final',C.TEST_START,C.TEST_END,gap_days,selection=False)

def summary(df,fs,gap):
    rows=[]
    for f in fs:
        tr=df.iloc[f.train];sc=df.iloc[f.score]
        assert not set(tr.order_id)&set(sc.order_id)
        rows.append({'fold':f.name,'cutoff':str(f.cutoff.date()),'window_end_exclusive':str(f.end.date()),
            'gap_days':gap,'train_n':len(tr),'score_n':len(sc),'window_completed_n':len(f.all_score),
            'score_label_coverage':len(sc)/len(f.all_score),'train_latest_purchase':str(tr.purchase_ts.max()),
            'train_latest_label':str(tr.delivered_ts.max()),'labels_at_or_after_cutoff':int(tr.delivered_ts.ge(f.cutoff).sum()),
            'scoring_labels_at_or_after_selection_cutoff':int(sc.delivered_ts.ge(C.SELECTION_CUTOFF).sum()) if f.name!='final' else None,
            'pending_removed_after_gap':int(((df.purchase_ts>=C.START)&(df.purchase_ts<f.cutoff-pd.Timedelta(days=gap))&
                                            df.delivered_ts.ge(f.cutoff)).sum())})
    return pd.DataFrame(rows)
