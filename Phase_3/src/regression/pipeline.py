"""Audited chronological search, diagnostics, freezing and final assessment."""
import json,time,warnings,hashlib,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from . import config as C, data, splits, audit, models

def log(text):
    line=datetime.now(timezone.utc).isoformat()+' '+text
    print(line,flush=True)
    with (C.OUT/'run_log.txt').open('a') as f:f.write(line+'\n')
def read(name):return json.loads((C.OUT/name).read_text())
def save_table(df,name):df.to_csv(C.TABLES/(name+'.csv'),index=False)

def prepared(df,fold,pack='calendar'):
    """Fold-only transform cache; only purchase inputs, no outcomes in the transform."""
    cache=C.PRIVATE/f'preprocessor_{fold.name}_{pack}.joblib'
    fit=df.iloc[fold.train];score=df.iloc[fold.score]
    if cache.exists():
        value=joblib.load(cache)
        assert value['fit_hash']==data.ids_hash(fit.order_id) and value['score_hash']==data.ids_hash(score.order_id)
        return value
    prep=models.preprocessing(pack)
    a=prep.fit_transform(fit);b=prep.transform(score)
    record=models.preprocessor_audit(prep,fit) if pack=='calendar' else {'fit_n':len(fit),'encoded_features':a.shape[1]}
    value={'prep':prep,'a':a,'b':b,'fit_hash':data.ids_hash(fit.order_id),'score_hash':data.ids_hash(score.order_id),'audit':record}
    joblib.dump(value,cache)
    return value

def select():
    audit.require_pass();C.directories()
    if (C.OUT/'main_cv_complete.json').exists():log('select: existing completed main CV retained');return
    df=data.load();gap=read('purge_gap.json')['q99_days'];fs=splits.folds(df,gap)
    design={'candidates':models.specs(),'pack':'calendar','selection':'mean five monthly MAEs; tie fewer effective parameters',
            'planned_later_diagnostics':['calendar ablation','RF original/log target','gap sensitivity','temporal risk feature','one convex stack'],
            'source_sha256':data.digest(Path(models.__file__)),'written_before_fitting':True}
    path=C.OUT/'search_design.json'
    if path.exists():assert read('search_design.json')==design
    else:data.save_json(path,design)
    specs=models.specs();rows=[];prep_audits=[]
    for fold in fs:
        fit,score=df.iloc[fold.train],df.iloc[fold.score];value=prepared(df,fold);prep_audits.append({'fold':fold.name,**value['audit']})
        for name,spec in specs.items():
            checkpoint=C.PRIVATE/f'cv_{fold.name}_{name}.joblib'
            if checkpoint.exists():v=joblib.load(checkpoint)
            else:
                started=time.perf_counter()
                reg=models.fit_on_arrays(value['a'],fit.lead_time_days.to_numpy(),spec)
                trainp=np.maximum(0,reg.predict(value['a']));valp=np.maximum(0,reg.predict(value['b']))
                row={'candidate':name,'family':spec['family'],'fold':fold.name,'seconds':time.perf_counter()-started,
                    **{'train_'+k:v for k,v in models.metrics(fit.lead_time_days,trainp).items()},
                    **{'val_'+k:v for k,v in models.metrics(score.lead_time_days,valp).items()}}
                v={'row':row,'prediction':valp,'order_id':score.order_id.tolist(),'learner':reg}
                joblib.dump(v,checkpoint)
                log(f'select {fold.name} {name}: MAE {row["val_mae"]:.4f}, {row["seconds"]:.1f}s')
            rows.append(v['row']);save_table(pd.DataFrame(rows),'cv_all_candidates_folds')
    full=pd.DataFrame(rows)
    summary=full.groupby(['candidate','family'],sort=False).agg(cv_mae=('val_mae','mean'),cv_sd=('val_mae','std'),
        cv_rmse=('val_rmse','mean'),train_mae=('train_mae','mean'),seconds=('seconds','sum')).reset_index().sort_values('cv_mae',kind='stable')
    save_table(summary,'cv_candidates')
    winners=summary.sort_values('cv_mae',kind='stable').drop_duplicates('family')
    save_table(winners,'cv_family_summary')
    # May coverage sensitivity: re-aggregate existing predictions; no new fit or new selection rule.
    stable=full[full.fold!='fold5'].groupby(['candidate','family']).val_mae.mean().reset_index(name='jan_apr_mae')
    stable=summary[['candidate','family','cv_mae']].merge(stable,on=['candidate','family'])
    stable['five_month_rank']=stable.cv_mae.rank(method='min');stable['jan_apr_rank']=stable.jan_apr_mae.rank(method='min')
    save_table(stable.sort_values('five_month_rank'),'selection_maturity_sensitivity')
    base_rows=[]
    for f in fs:
        tr,sc=df.iloc[f.train],df.iloc[f.score]
        for name,constant in [('Mean',tr.lead_time_days.mean()),('Median',tr.lead_time_days.median()),('Promise',None)]:
            a=tr.promised_lead_time_days.to_numpy() if constant is None else np.full(len(tr),constant)
            b=sc.promised_lead_time_days.to_numpy() if constant is None else np.full(len(sc),constant)
            base_rows.append({'candidate':name,'family':'Reference','fold':f.name,
                **{'train_'+k:v for k,v in models.metrics(tr.lead_time_days,a).items()},
                **{'val_'+k:v for k,v in models.metrics(sc.lead_time_days,b).items()}})
    save_table(pd.DataFrame(base_rows),'cv_reference_folds')
    data.save_json(C.OUT/'preprocessing_audit.json',prep_audits)
    data.save_json(C.OUT/'main_cv_complete.json',{'family_winners':dict(zip(winners.family,winners.candidate)),
        'best_candidate':summary.iloc[0].candidate,'new_fits':len(rows),'search_sha256':data.digest(path),
        'cv_table_sha256':data.digest(C.TABLES/'cv_all_candidates_folds.csv')})
    log('Main CV complete; test remains locked. Next: diagnostics and temporal ensemble.')

def run(stage):
    if stage=='select':return select()
    from . import extensions
    return extensions.run(stage)
