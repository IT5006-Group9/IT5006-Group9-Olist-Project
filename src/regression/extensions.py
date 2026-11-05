"""Pre-test feature/gap/ensemble checks and one frozen final evaluation."""
import json,time,copy,warnings
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from . import config as C, data, splits, audit, models, risk
from .pipeline import log,read,save_table

def main_winners():return read('main_cv_complete.json')['family_winners']
def specification(name):return models.specs()[name]

def evaluate(name,spec,df,fs,pack='calendar',log_target=False,risk_key=''):
    rows=[];predictions=[]
    for f in fs:
        path=C.PRIVATE/f'extension_{name}_{f.name}.joblib'
        if path.exists():v=joblib.load(path)
        else:
            fit,score=df.iloc[f.train],df.iloc[f.score];started=time.perf_counter();rb=None
            if pack=='risk':
                fit,rv=risk.enrich_training(fit,f.cutoff,'outer_'+f.name)
                score=risk.enrich_scoring(score,rv['bundle']);rb=rv['bundle']
            bundle=models.fit_model(fit,spec,pack,log_target)
            bundle['risk_bundle']=rb
            a=models.predict(bundle,fit);b=models.predict(bundle,score)
            row={'candidate':name,'family':spec['family'],'fold':f.name,'pack':pack,'log_target':log_target,
                 **{'train_'+k:v for k,v in models.metrics(fit.lead_time_days,a).items()},
                 **{'val_'+k:v for k,v in models.metrics(score.lead_time_days,b).items()}}
            v={'row':row,'prediction':b,'bundle':bundle,'order_id':score.order_id.tolist()}
            joblib.dump(v,path);log(f'diagnostic {name} {f.name}: MAE {row["val_mae"]:.4f}, {time.perf_counter()-started:.1f}s')
        rows.append(v['row'])
        predictions.append(pd.DataFrame({'candidate':name,'fold':f.name,'order_id':v['order_id'],'prediction':v['prediction']}))
    return rows,pd.concat(predictions,ignore_index=True)

def diagnostics():
    audit.require_pass()
    if (C.OUT/'diagnostics_complete.json').exists():log('diagnostics: completed results retained');return
    df=data.load();gaps=read('purge_gap.json');gap=gaps['q99_days'];fs=splits.folds(df,gap)
    winners=main_winners();hgb=specification(winners['HGB']);rf=specification(winners['RF'])
    plan={'hgb_config':hgb,'packs':['no_calendar','calendar','absolute','risk'],
          'rf_log_target_config':rf,'gap_sensitivity':[0,gaps['q95_days'],gap],
          'gap_strict_label_eligibility_always':True,'risk_parameters':risk.PARAMS,
          'risk_cold_start':'Jan-Mar 2017 score=0 plus unavailable flag; later monthly strictly past cross-fit',
          'stack_rule':'One Ridge/RF/HGB convex stack; retain only if mean outer validation MAE improves >=1% over best single model.'}
    design=C.OUT/'diagnostics_design.json'
    if design.exists():assert read('diagnostics_design.json')==plan
    else:data.save_json(design,plan)
    rows=[];oofs=[]
    for pack in plan['packs']:
        rr,oo=evaluate('HGB_'+pack,hgb,df,fs,pack);rows+=rr;oofs.append(oo)
        save_table(pd.DataFrame(rows),'diagnostic_folds')
    rr,oo=evaluate('RF_log_target',rf,df,fs,log_target=True);rows+=rr;oofs.append(oo)
    for g in [0,gaps['q95_days']]:
        rr,_=evaluate('HGB_gap'+str(g),hgb,df,splits.folds(df,g));rows+=rr
    full=pd.DataFrame(rows);save_table(full,'diagnostic_folds')
    summary=full.groupby(['candidate','family','pack','log_target']).agg(cv_mae=('val_mae','mean'),cv_sd=('val_mae','std'),
        train_mae=('train_mae','mean'),cv_rmse=('val_rmse','mean')).reset_index()
    save_table(summary.sort_values('cv_mae'),'diagnostic_summary')
    pd.concat(oofs).to_pickle(C.PRIVATE/'extension_oof.pkl')
    # Gap is a protocol sensitivity, not an invitation to select on easier training labels.
    eligible=summary[~summary.candidate.str.startswith('HGB_gap')].copy()
    main=pd.read_csv(C.TABLES/'cv_family_summary.csv').assign(pack='calendar',log_target=False)
    combined=pd.concat([main[['candidate','family','pack','log_target','cv_mae','cv_sd']],
                        eligible[['candidate','family','pack','log_target','cv_mae','cv_sd']]],ignore_index=True)
    combined['complexity']=combined.candidate.str.contains('risk|absolute').astype(int)
    best=combined.sort_values(['cv_mae','complexity'],kind='stable').iloc[0]
    data.save_json(C.OUT/'diagnostics_complete.json',{'best_single':best.drop('complexity').to_dict(),
        'design_sha256':data.digest(design),'candidate_count':len(combined)})
    log('Feature/gap diagnostics complete; chronological test is still locked.')

def weights(y,x):
    # A small linear program exactly minimises MAE with convex weights and no intercept.
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, hstack, vstack, eye
    n,k=x.shape
    a=vstack([hstack([csr_matrix(x),-eye(n)]),hstack([-csr_matrix(x),-eye(n)])]).tocsr()
    b=np.concatenate([y,-y]);c=np.r_[np.zeros(k),np.ones(n)/n]
    eq=csr_matrix(np.r_[np.ones(k),np.zeros(n)].reshape(1,-1))
    result=linprog(c,A_ub=a,b_ub=b,A_eq=eq,b_eq=[1],bounds=[(0,1)]*k+[(0,None)]*n,method='highs')
    if not result.success:raise RuntimeError(result.message)
    return result.x[:k]

def inner_windows(frame,outer_cutoff,gap):
    # Most recent three purchased months in outer TRAIN, never the outer validation month.
    months=sorted(frame.purchase_ts.dt.to_period('M').unique())[-3:]
    fs=[]
    for m in months:
        start=m.start_time;end=start+pd.offsets.MonthBegin(1)
        train=np.flatnonzero((frame.purchase_ts<start-pd.Timedelta(days=gap))&(frame.delivered_ts<start))
        score=np.flatnonzero((frame.purchase_ts>=start)&(frame.purchase_ts<end)&(frame.delivered_ts<outer_cutoff))
        if len(train)>1000 and len(score)>0:fs.append((str(m),start,train,score))
    assert fs
    return fs

def inner_meta(frame,cutoff,key,gap,winners):
    path=C.PRIVATE/f'meta_training_{key}.joblib'
    if path.exists():return joblib.load(path)
    xs=[];ys=[];audit_rows=[]
    for month,date,tr,sc in inner_windows(frame,cutoff,gap):
        a,b=frame.iloc[tr],frame.iloc[sc];columns=[]
        for family in ['Ridge','RF','HGB']:
            model=models.fit_model(a,specification(winners[family]));columns.append(models.predict(model,b))
        xs.append(np.column_stack(columns));ys.append(b.lead_time_days.to_numpy())
        assert a.delivered_ts.lt(date).all() and b.delivered_ts.lt(cutoff).all()
        audit_rows.append({'outer':key,'inner_month':month,'cutoff':str(date),'meta_label_cutoff':str(cutoff),
            'train_n':len(a),'meta_n':len(b),'train_latest_label':str(a.delivered_ts.max()),
            'meta_latest_label':str(b.delivered_ts.max()),'overlap':len(set(a.order_id)&set(b.order_id))})
        log(f'stack inner {key}/{month}: {len(a)} train / {len(b)} meta rows')
    x,y=np.concatenate(xs),np.concatenate(ys);w=weights(y,x)
    value={'x':x,'y':y,'weights':w,'audit':audit_rows,'fit_hash':data.ids_hash(frame.order_id)}
    joblib.dump(value,path);return value

def stacking():
    audit.require_pass()
    if (C.OUT/'stacking_complete.json').exists():log('stacking: completed results retained');return
    df=data.load();gap=read('purge_gap.json')['q99_days'];winners=main_winners();fs=splits.folds(df,gap)
    rows=[];audit_rows=[];oof=[]
    for f in fs:
        frame=df.iloc[f.train];score=df.iloc[f.score]
        meta=inner_meta(frame,f.cutoff,f.name,gap,winners);audit_rows+=meta['audit'];columns=[]
        # Reuse audited OUTER predictions, never global OOF to train this fold's combiner.
        for fam in ['Ridge','RF','HGB']:
            v=joblib.load(C.PRIVATE/f'cv_{f.name}_{winners[fam]}.joblib')
            assert v['order_id']==score.order_id.tolist();columns.append(v['prediction'])
        p=np.column_stack(columns)@meta['weights'];tp=meta['x']@meta['weights']
        rows.append({'candidate':'Convex_stack','fold':f.name,'ridge_weight':meta['weights'][0],
                     'rf_weight':meta['weights'][1],'hgb_weight':meta['weights'][2],
                     **{'val_'+k:v for k,v in models.metrics(score.lead_time_days,p).items()},
                     'meta_training_mae':models.metrics(meta['y'],tp)['mae']})
        oof.append(pd.DataFrame({'order_id':score.order_id,'fold':f.name,'prediction':p}))
    tab=pd.DataFrame(rows);save_table(tab,'stacking_folds');save_table(pd.DataFrame(audit_rows),'stacking_inner_audit')
    pd.concat(oof).to_pickle(C.PRIVATE/'stacking_oof.pkl')
    best=read('diagnostics_complete.json')['best_single'];cv=float(tab.val_mae.mean());gain=1-cv/best['cv_mae']
    data.save_json(C.OUT/'stacking_complete.json',{'cv_mae':cv,'cv_sd':float(tab.val_mae.std(ddof=1)),
        'best_single_cv_mae':best['cv_mae'],'relative_gain':gain,'retained':bool(gain>=.01),
        'threshold':.01,'base_configuration_scope':'Base parameters selected using main five-month CV; not fully nested hyperparameter selection.',
        'meta_rule':'Three recent inner expanding windows per outer training frame; convex weights, zero intercept.'})
    log(f'Stacking CV MAE {cv:.4f}, gain {gain:.2%}; retained={gain>=.01}. Test remains locked.')

def run(stage):
    if stage=='diagnostics':return diagnostics()
    if stage=='stacking':return stacking()
    from . import finalise
    return finalise.run(stage)
