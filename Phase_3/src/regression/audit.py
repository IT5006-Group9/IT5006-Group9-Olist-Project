"""Pre-model gate: independent raw checks and adversarial time-boundary tests."""
from pathlib import Path
import json, os, subprocess
import numpy as np
import pandas as pd
from . import config as C, data, splits

def source_hashes():
    return {n:data.digest(Path(__file__).parent/(n+'.py')) for n in ['config','data','splits','audit']}

def boundary_check():
    f=pd.DataFrame({'purchase_ts':pd.to_datetime(['2017-10-01','2017-11-01','2018-01-01','2018-05-15','2018-05-30']),
                    'delivered_ts':pd.to_datetime(['2017-10-10','2018-01-01','2018-02-10','2018-05-25','2018-06-10'])})
    first=splits.window(f,'check','2018-01-01','2018-02-01',0)
    assert first.train.tolist()==[0]   # label exactly at cutoff is unavailable
    assert first.score.tolist()==[2]   # validation can finish after its own month
    may=splits.window(f,'may','2018-05-01','2018-06-01',0)
    assert may.score.tolist()==[3] and may.all_score.tolist()==[3,4]

def run():
    C.directories(); gate=C.OUT/'audit_pass.json'
    if (C.OUT/'frozen_selection.json').exists():
        raise RuntimeError('A model selection already exists; preserve its input audit instead of overwriting.')
    df,t=data.prepare();boundary_check()
    # Direct target reconstruction from raw orders, independent of the feature join.
    o=t['orders'];p=pd.to_datetime(o.order_purchase_timestamp);d=pd.to_datetime(o.order_delivered_customer_date)
    y=(d-p).dt.total_seconds()/86400
    eligible=o.order_status.eq('delivered')&y.notna()&y.ge(0)&p.ge(C.START)&p.lt(C.TEST_END)
    assert set(o.loc[eligible,'order_id'])==set(df.order_id)
    expected=pd.Series(y[eligible].to_numpy(),index=o.loc[eligible,'order_id'])
    np.testing.assert_array_equal(expected.loc[df.order_id].to_numpy(),df.lead_time_days.to_numpy())
    # Direct basket check on 120 deterministic orders, without reusing feature helper.
    sample=df.sample(120,random_state=C.SEED)
    raw=t['items'][t['items'].order_id.isin(sample.order_id)]
    sums=raw.groupby('order_id').agg(price=('price','sum'),freight=('freight_value','sum'),n=('order_item_id','size'))
    for row in sample.itertuples():
        v=sums.loc[row.order_id]
        assert np.isclose(row.total_price,v.price) and np.isclose(row.total_freight,v.freight) and row.item_count==v.n
    # Alter all post-outcome fields and show the complete purchase feature table is unchanged.
    changed={k:v.copy() if k=='orders' else v for k,v in t.items()}
    for col in ['order_delivered_customer_date','order_delivered_carrier_date','order_approved_at']:
        changed['orders'][col]='2099-12-31 23:59:59'
    changed['orders']['order_status']='invoiced'
    original=data.features(t);mutated=data.features(changed)
    pd.testing.assert_frame_equal(original,mutated,check_exact=True)
    gap=splits.gap_spec(df);data.save_json(C.OUT/'purge_gap.json',gap)
    fs=splits.folds(df,gap['q99_days']);final=splits.final(df,gap['q99_days'])
    table=splits.summary(df,fs+[final],gap['q99_days'])
    assert table.labels_at_or_after_cutoff.eq(0).all()
    assert table.loc[table.fold.ne('final'),'scoring_labels_at_or_after_selection_cutoff'].eq(0).all()
    test_ids=set(df.iloc[final.score].order_id)
    membership=[]
    for f in fs+[final]:
        train=df.iloc[f.train];score=df.iloc[f.score]
        assert train.purchase_ts.lt(f.cutoff).all() and train.delivered_ts.lt(f.cutoff).all()
        assert not set(train.order_id)&set(score.order_id)
        if f.name!='final': assert not (set(train.order_id)|set(score.order_id))&test_ids
        for role,idx in [('train',f.train),('score',f.score),('full_window',f.all_score)]:
            membership.append(df.iloc[idx][['order_id','purchase_ts','delivered_ts']].assign(fold=f.name,role=role))
    pd.concat(membership,ignore_index=True).to_csv(C.PRIVATE/'fold_manifest.csv.gz',index=False)
    table.to_csv(C.TABLES/'fold_table.csv',index=False)
    # Time anomalies are diagnostics, never purchase predictors.
    approved=pd.to_datetime(o.order_approved_at);carrier=pd.to_datetime(o.order_delivered_carrier_date)
    flagged=set(o.loc[d.lt(approved)|d.lt(carrier),'order_id'])
    df['chronology_flag']=df.order_id.isin(flagged)
    df.to_pickle(C.PRIVATE/'orders.pkl')
    lock={'test_start':'2018-06-01','test_end_exclusive':'2018-09-01','n':len(test_ids),
          'ids_sha256':data.ids_hash(test_ids),'evaluations':[],
          'previous_exposure':'These historical orders were used in earlier Phase 2 experiments; not an unseen external test.',
          'rule':'Freeze configuration and comparison list before scoring; no changes based on test performance.'}
    data.save_json(C.OUT/'test_lock.json',lock)
    catalog=[]
    num,cat=C.columns('absolute')
    for col in num+cat:
        group=next((g for g,cs in {**C.NUMERIC_GROUPS}.items() if col in cs),None)
        if group is None: group=next((g for g,cs in C.CAT_GROUPS.items() if col in cs),'absolute_calendar')
        catalog.append({'feature':col,'group':group,'type':'numeric' if col in num else 'categorical',
                        'availability':'purchase; archived metadata assumption',
                        'source':'purchase/quote, order items, seller, customer, product or static geography',
                        'target_derived':False,'used_in_main_pack':group!='absolute_calendar'})
    pd.DataFrame(catalog).to_csv(C.TABLES/'feature_catalogue.csv',index=False)
    protocol={'prediction_point':'purchase timestamp','target':'(receipt-purchase) fractional days',
        'validation_months':C.MONTHS,'training_start':str(C.START.date()),
        'selection_labels_before':str(C.SELECTION_CUTOFF.date()),'test_window':['2018-06-01','2018-09-01'],
        'main_gap_days':gap['q99_days'],'training_rule':'purchase < cutoff-gap AND receipt < cutoff',
        'selection_rule':'mean monthly validation MAE, sample SD; pooled MAE separately reported',
        'success_criteria':C.SUCCESS,'seed':C.SEED,'primary_target_scale':'original fractional days',
        'test_previously_exposed':True,'source_hashes':source_hashes()}
    data.save_json(C.OUT/'protocol.json',protocol)
    prep=json.loads((C.OUT/'preparation.json').read_text())
    evidence={'passed':True,'stage':'pre-model audit; no estimator fitted',
        'raw_target_rows_verified':len(df),'basket_orders_independently_verified':120,
        'post_outcome_perturbation_rows_verified':len(original),'time_boundary_tests':True,
        'zero_future_training_labels':True,'zero_unavailable_selection_labels':True,
        'train_score_disjoint':True,'test_excluded_from_development':True,
        'input_sha256':data.digest(C.PRIVATE/'orders.pkl'),'manifest_sha256':data.digest(C.PRIVATE/'fold_manifest.csv.gz'),
        'source_hashes':source_hashes(),'reference_v2_fields_verified':prep['reference_v2_fields_verified'],
        'limitations':prep['assumptions']+['Selection-label coverage excludes outcomes not yet observed by June 1; see per-fold coverage.',
            'An embargo does not resolve completion selection bias; strict observed-label filtering remains required.']}
    data.save_json(gate,evidence)
    print(table.to_string(index=False));print('AUDIT PASSED; model fitting is now permitted.',flush=True)

def require_pass():
    evidence=json.loads((C.OUT/'audit_pass.json').read_text())
    assert evidence['passed'] and evidence['source_hashes']==source_hashes(),'Audit source changed; rerun before fitting.'
    assert evidence['input_sha256']==data.digest(C.PRIVATE/'orders.pkl'),'Audited data changed.'
    return evidence

if __name__=='__main__':run()
