"""Independent final evidence reconciliation, interface tests and golden orders."""
import json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from . import config as C, data, splits, audit, score

def run():
    audit.require_pass();df=data.load();selection=json.loads((C.OUT/'frozen_selection.json').read_text())
    lock=json.loads((C.OUT/'test_lock.json').read_text());summary=json.loads((C.OUT/'test_summary.json').read_text())
    assert len(lock['evaluations'])==1
    assert lock['evaluations'][0]['selection_sha256']==selection['selection_sha256']
    gap=selection['gap_days'];f=splits.final(df,gap);test=df.iloc[f.score]
    assert data.ids_hash(test.order_id)==lock['ids_sha256']
    predictions=pd.read_pickle(C.PRIVATE/'test_predictions.pkl')
    published=pd.read_csv(C.TABLES/'train_test_results.csv')
    checked=[]
    for name,part in predictions.groupby('candidate'):
        assert part.order_id.is_unique and set(part.order_id)==set(test.order_id)
        p=part.set_index('order_id').loc[test.order_id].prediction.to_numpy()
        y=test.lead_time_days.to_numpy();err=p-y
        expected={'mae':np.abs(err).mean(),'rmse':np.sqrt(np.dot(err,err)/len(err)),
                  'bias':err.mean(),'r2':1-np.dot(err,err)/np.square(y-y.mean()).sum(),
                  'within3':np.count_nonzero(np.abs(err)<=3)/len(y)}
        row=published[(published.candidate==name)&(published.split=='test')].iloc[0]
        for k,v in expected.items():np.testing.assert_allclose(row[k],v,rtol=1e-12,atol=1e-12)
        path=C.MODELS/(name+'.joblib')
        if path.exists():
            loaded=joblib.load(path);np.testing.assert_allclose(score.predict_bundle(test,loaded),p,rtol=1e-12,atol=1e-12)
        checked.append(name)
    # Inspect every cached primary CV prediction against its own fold, and check mean/SD independently.
    candidate_folds=pd.read_csv(C.TABLES/'cv_all_candidates_folds.csv')
    for fold in splits.folds(df,gap):
        actual=df.iloc[fold.score].lead_time_days.to_numpy();ids=df.iloc[fold.score].order_id.tolist()
        rows=candidate_folds[candidate_folds.fold==fold.name]
        for row in rows.itertuples():
            value=joblib.load(C.PRIVATE/f'cv_{fold.name}_{row.candidate}.joblib')
            assert value['order_id']==ids
            error=np.asarray(value['prediction'])-actual
            np.testing.assert_allclose(row.val_mae,np.abs(error).mean(),rtol=1e-12,atol=1e-12)
    candidates=pd.read_csv(C.TABLES/'cv_candidates.csv')
    for row in candidates.itertuples():
        values=candidate_folds[candidate_folds.candidate==row.candidate].val_mae.to_numpy()
        np.testing.assert_allclose([row.cv_mae,row.cv_sd],[values.sum()/5,np.sqrt(np.square(values-values.mean()).sum()/4)])
    # Reconcile disjoint duration slices with overall order and absolute-error totals.
    groups=pd.read_csv(C.TABLES/'test_error_groups.csv').iloc[:4]
    assert groups.n.sum()==len(test)
    np.testing.assert_allclose(groups.order_share.sum(),1)
    np.testing.assert_allclose(groups.absolute_error_share.sum(),1)
    np.testing.assert_allclose(np.sum(groups.n*groups.mae)/len(test),summary['test']['mae'])
    # New process reload, field-deletion/perturbation checks, unknown categories and missing values.
    bundle=score.load_bundle();sample=test.sample(20,random_state=C.SEED).copy()
    needed=[c for c in bundle['input_columns'] if c not in ['risk_p30','risk_unavailable']]
    input_columns=list(dict.fromkeys(['order_id','purchase_ts','promised_lead_time_days',*needed]))
    safe=sample[input_columns].copy();safe.to_pickle(C.PRIVATE/'deployment_orders.pkl')
    expected=score.predict_lead_time(safe,bundle);expected.to_csv(C.PRIVATE/'deployment_expected.csv',index=False)
    np.testing.assert_allclose(expected.predicted_days,score.predict_lead_time(sample,bundle).predicted_days)
    changed=sample.copy()
    for c in ['lead_time_days','delivered_ts','order_status','review_score']:
        changed[c]=99999 if c=='lead_time_days' else 'future_changed'
    np.testing.assert_allclose(expected.predicted_days,score.predict_lead_time(changed,bundle).predicted_days)
    unknown=safe.copy();unknown['primary_seller_id']='__never_seen__';unknown['max_distance_km']=np.nan
    assert np.isfinite(score.predict_lead_time(unknown,bundle).predicted_days).all()
    failures=0
    for value in [safe.iloc[:0],safe.drop(columns=['total_price']),safe.assign(total_price=-1)]:
        try:score.predict_lead_time(value,bundle)
        except ValueError:failures+=1
    assert failures==3
    # Alternative formulas and all stored temporal inner boundaries, not just the helper's assertion.
    inner=pd.read_csv(C.TABLES/'stacking_inner_audit.csv')
    assert inner.overlap.eq(0).all()
    assert (pd.to_datetime(inner.train_latest_label)<pd.to_datetime(inner.cutoff)).all()
    assert (pd.to_datetime(inner.meta_latest_label)<pd.to_datetime(inner.meta_label_cutoff)).all()
    stack=pd.read_csv(C.TABLES/'stacking_folds.csv')
    w=stack[['ridge_weight','rf_weight','hgb_weight']].to_numpy()
    assert np.min(w)>=-1e-10
    np.testing.assert_allclose(w.sum(axis=1),1.,atol=1e-9)
    risk_checks=0
    for p in C.PRIVATE.glob('risk_*.joblib'):
        v=joblib.load(p)
        for r in v['audit']:
            assert pd.Timestamp(r['latest_label'])<pd.Timestamp(r['cutoff'])
            assert r['future_training_labels']==0;risk_checks+=1
    record={'passed':True,'test_comparators_recomputed':checked,'primary_cv_predictions_verified':len(candidate_folds),
        'primary_cv_summary_rows_verified':len(candidates),'test_model_reloads':sum((C.MODELS/(n+'.joblib')).exists() for n in checked),
        'duration_groups_reconcile':True,'golden_order_n':20,'outcome_deletion_and_perturbation_invariance':True,
        'unknown_category_and_missing_numeric':True,'interface_errors_checked':failures,
        'stack_inner_windows_checked':len(inner),'risk_inner_windows_checked':risk_checks,
        'test_evaluation_count':len(lock['evaluations']),'model_sha256':data.digest(C.MODELS/'final_model.joblib'),
        'full_hyperparameter_search_repeated':False,
        'frozen_reproduction_record':'frozen_reproduction_check.json' if (C.OUT/'frozen_reproduction_check.json').exists() else None,
        'limitations':'This verifier reconciles existing fitted evidence; a separate frozen reproduction record documents a fresh raw-ZIP pipeline refit.'}
    data.save_json(C.OUT/'verification.json',record)
    print(json.dumps(record,indent=2))

if __name__=='__main__':run()
