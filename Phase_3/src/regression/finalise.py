"""Freeze, assess once, interpret, and generate the regression handoff evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,copy
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from . import config as C, data, splits, audit, models, risk
from .pipeline import log,read,save_table
from .extensions import specification,main_winners,inner_meta
from .score import predict_bundle,predict_lead_time

def single_descriptor():
    s=read('diagnostics_complete.json')['best_single'];name=s['candidate']
    if name in models.specs():spec=specification(name)
    elif name.startswith('HGB_'):spec=specification(main_winners()['HGB'])
    elif name=='RF_log_target':spec=specification(main_winners()['RF'])
    else:raise ValueError(name)
    return {'candidate':name,'spec':spec,'pack':s['pack'],'log_target':bool(s['log_target'])}

def fit_descriptor(frame,cutoff,descriptor,key):
    work=frame;rv=None
    if descriptor['pack']=='risk':work,rv=risk.enrich_training(frame,cutoff,key)
    bundle=models.fit_model(work,descriptor['spec'],descriptor['pack'],descriptor['log_target'])
    bundle['risk_bundle']=rv['bundle'] if rv else None
    bundle['candidate']=descriptor['candidate'];bundle['training_cutoff']=str(cutoff)
    return bundle,work

def freeze():
    audit.require_pass()
    if (C.OUT/'frozen_selection.json').exists():log('freeze: existing frozen selection retained');return
    descriptor=single_descriptor();st=read('stacking_complete.json');winners=main_winners()
    config={'best_single':descriptor,'deploy_ensemble':st['retained'],'family_winners':winners,
            'selection_metric':'mean MAE across Jan-May monthly validation, labels observed before Jun 1',
            'ensemble_threshold':.01,'gap_days':read('purge_gap.json')['q99_days'],
            'test_comparisons':['Mean','Median','Promise',*winners.values(),descriptor['candidate']],
            'success_criteria':C.SUCCESS,'test_previously_exposed':True}
    if st['retained']:config['test_comparisons'].append('Convex_stack')
    config['test_comparisons']=list(dict.fromkeys(config['test_comparisons']))
    config['source_hashes']={p.name:data.digest(p) for p in Path(__file__).parent.glob('*.py')}
    config['selection_sha256']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config['frozen_at']=datetime.now(timezone.utc).isoformat()
    data.save_json(C.OUT/'frozen_selection.json',config)
    log('Final selection and all test comparators frozen before test scoring: '+descriptor['candidate'])

def evaluate():
    audit.require_pass();selection=read('frozen_selection.json');lock=read('test_lock.json')
    if lock['evaluations']:raise RuntimeError('This frozen test assessment has already run; use saved evidence, not a new search.')
    df=data.load();f=splits.final(df,selection['gap_days']);train,test=df.iloc[f.train],df.iloc[f.score]
    assert data.ids_hash(test.order_id)==lock['ids_sha256']
    rows=[];predictions=[];bundles={};fitted_training={}
    descriptors=[{'candidate':n,'spec':specification(n),'pack':'calendar','log_target':False}
                 for n in selection['family_winners'].values()]
    descriptors+=[selection['best_single']]
    for d in descriptors:
        name=d['candidate']
        if name in bundles:continue
        bundle,work=fit_descriptor(train,C.TEST_START,d,'final_training')
        bundles[name]=bundle;fitted_training[name]=work
        joblib.dump(bundle,C.MODELS/(name+'.joblib'),compress=3)
    selected=selection['best_single']['candidate']
    if selection['deploy_ensemble']:
        meta=inner_meta(train,C.TEST_START,'final_training',selection['gap_days'],selection['family_winners'])
        bases=[bundles[selection['family_winners'][fam]] for fam in ['Ridge','RF','HGB']]
        bundle={'ensemble':True,'base_models':bases,'weights':meta['weights'],'candidate':'Convex_stack',
            'input_columns':bases[0]['input_columns'],'training_cutoff':str(C.TEST_START)}
        bundles['Convex_stack']=bundle;selected='Convex_stack';fitted_training[selected]=train
    # All fits and the comparison list are fixed before any test performance is calculated.
    for name,constant in [('Mean',train.lead_time_days.mean()),('Median',train.lead_time_days.median()),('Promise',None)]:
        for split,frame in [('train',train),('test',test)]:
            p=frame.promised_lead_time_days.to_numpy() if constant is None else np.full(len(frame),constant)
            rows.append({'candidate':name,'split':split,**models.metrics(frame.lead_time_days,p)})
            if split=='test':predictions.append(pd.DataFrame({'candidate':name,'order_id':frame.order_id,'prediction':p}))
    for name,bundle in bundles.items():
        for split,frame in [('train',train),('test',test)]:
            # In-sample risk training uses the cross-fitted values used during regression fitting.
            if split=='train' and bundle.get('pack')=='risk':p=models.predict(bundle,fitted_training[name])
            else:p=predict_bundle(frame,bundle)
            rows.append({'candidate':name,'split':split,**models.metrics(frame.lead_time_days,p)})
            if split=='test':predictions.append(pd.DataFrame({'candidate':name,'order_id':frame.order_id,'prediction':p}))
    table=pd.DataFrame(rows);save_table(table,'train_test_results')
    pred=pd.concat(predictions,ignore_index=True);pred.to_pickle(C.PRIVATE/'test_predictions.pkl')
    champion=bundles[selected];champion['selection_sha256']=selection['selection_sha256']
    joblib.dump(champion,C.MODELS/'final_model.joblib',compress=3)
    best=table[(table.split=='test')&(table.candidate==selected)].iloc[0]
    median=table[(table.split=='test')&(table.candidate=='Median')].iloc[0]
    promise=table[(table.split=='test')&(table.candidate=='Promise')].iloc[0]
    success={'mae_reduction_vs_median':1-best.mae/median.mae,'minimum_reduction':.05,
             'median_criterion_met':bool(best.mae<=.95*median.mae),'promise_criterion_met':bool(best.mae<promise.mae)}
    data.save_json(C.OUT/'test_summary.json',{'selected_candidate':selected,'test':best.to_dict(),'success':success,
        'selection_sha256':selection['selection_sha256'],'test_n':len(test),'train_n':len(train)})
    lock['evaluations'].append({'date':datetime.now(timezone.utc).isoformat(),'selection_sha256':selection['selection_sha256'],
        'comparisons':selection['test_comparisons'],'results_sha256':data.digest(C.TABLES/'train_test_results.csv')})
    data.save_json(C.OUT/'test_lock.json',lock)
    log(f'Frozen test complete: {selected}, MAE {best.mae:.4f}, RMSE {best.rmse:.4f}. No further model selection permitted.')

def analysis():
    audit.require_pass();df=data.load();selection=read('frozen_selection.json');g=selection['gap_days']
    f=splits.final(df,g);test=df.iloc[f.score].copy();summary=read('test_summary.json');name=summary['selected_candidate']
    saved=pd.read_pickle(C.PRIVATE/'test_predictions.pkl');p=saved[saved.candidate==name].set_index('order_id').loc[test.order_id].prediction.to_numpy()
    error=np.abs(p-test.lead_time_days.to_numpy());rows=[]
    for group,mask in [('0-14 days',test.lead_time_days<=14),('14-30 days',test.lead_time_days.between(14,30,inclusive='right')),
            ('30-60 days',test.lead_time_days.between(30,60,inclusive='right')),('>60 days',test.lead_time_days>60),
            ('Same-state',test.same_state==1),('Cross-state',test.same_state==0),
            ('Chronology flags removed',~test.chronology_flag),('Missing distance',test.max_distance_km.isna())]:
        mask=np.asarray(mask)
        if not mask.any():continue
        rows.append({'group':group,'order_share':float(mask.mean()),'absolute_error_share':float(error[mask].sum()/error.sum()),
                     **models.metrics(test.lead_time_days.to_numpy()[mask],p[mask])})
    save_table(pd.DataFrame(rows),'test_error_groups')
    monthly=[]
    for m in sorted(test.purchase_ts.dt.to_period('M').unique()):
        mask=test.purchase_ts.dt.to_period('M').eq(m)
        monthly.append({'month':str(m),**models.metrics(test.loc[mask,'lead_time_days'],p[mask])})
    save_table(pd.DataFrame(monthly),'test_by_month')
    route=test.assign(prediction=p,absolute_error=error).groupby('primary_state_route').agg(n=('order_id','size'),
        mae=('absolute_error','mean'),absolute_error=('absolute_error','sum')).reset_index()
    route['absolute_error_share']=route.absolute_error/error.sum();save_table(route.sort_values('n',ascending=False),'test_by_route')
    # Full validation outcomes are post-selection diagnostics only, preserving the original model choices.
    diag=[]
    desc=selection['best_single']
    for fold in splits.folds(df,g):
        model_path=C.PRIVATE/f'validation_selected_{fold.name}.joblib'
        if model_path.exists():b=joblib.load(model_path)
        else:
            b,_=fit_descriptor(df.iloc[fold.train],fold.cutoff,desc,'outer_'+fold.name)
            joblib.dump(b,model_path)
        full=df.iloc[fold.all_score];pred=predict_bundle(full,b)
        for label,mask in [('Selection-time observed',full.delivered_ts<C.SELECTION_CUTOFF),('Eventual completed outcomes',np.ones(len(full),bool))]:
            mask=np.asarray(mask);diag.append({'fold':fold.name,'scope':label,**models.metrics(full.lead_time_days.to_numpy()[mask],pred[mask])})
    save_table(pd.DataFrame(diag),'validation_outcome_maturity_diagnostic')
    maturity=[];selection_coverage=[]
    for fold in splits.folds(df,g):
        before=(df.purchase_ts>=C.START)&(df.purchase_ts<fold.cutoff)
        embargo=(df.purchase_ts>=C.START)&(df.purchase_ts<fold.cutoff-pd.Timedelta(days=g))
        for scope,mask in [('Before embargo',before),('Embargo-eligible',embargo),
                           ('Excluded: label pending after embargo',embargo&df.delivered_ts.ge(fold.cutoff))]:
            part=df.loc[mask]
            for label,lo,hi in [('0-14',-1,14),('14-30',14,30),('30-60',30,60),('>60',60,np.inf)]:
                maturity.append({'fold':fold.name,'scope':scope,'duration_group':label,
                    'n':int(((part.lead_time_days>lo)&(part.lead_time_days<=hi)).sum())})
        part=df.iloc[fold.all_score]
        for label,lo,hi in [('0-14',-1,14),('14-30',14,30),('30-60',30,60),('>60',60,np.inf)]:
            band=part[(part.lead_time_days>lo)&(part.lead_time_days<=hi)]
            selection_coverage.append({'fold':fold.name,'duration_group':label,'eventual_n':len(band),
                'observed_by_selection_n':int(band.delivered_ts.lt(C.SELECTION_CUTOFF).sum())})
    save_table(pd.DataFrame(maturity),'training_maturity_exclusions')
    save_table(pd.DataFrame(selection_coverage),'validation_label_coverage_by_duration')
    # Include the feature variants, not only primary families, in the maturity-ranking check.
    ranking_folds=pd.concat([pd.read_csv(C.TABLES/'cv_all_candidates_folds.csv'),
                            pd.read_csv(C.TABLES/'diagnostic_folds.csv')],ignore_index=True)
    ranking_folds=ranking_folds[~ranking_folds.candidate.str.startswith('HGB_gap')]
    ranking=ranking_folds.groupby(['candidate','family']).val_mae.mean().reset_index(name='five_month_mae')
    early=ranking_folds[ranking_folds.fold!='fold5'].groupby('candidate').val_mae.mean().rename('jan_apr_mae')
    ranking=ranking.join(early,on='candidate')
    ranking['five_month_rank']=ranking.five_month_mae.rank(method='min')
    ranking['jan_apr_rank']=ranking.jan_apr_mae.rank(method='min')
    save_table(ranking.sort_values('five_month_rank'),'selection_maturity_sensitivity')
    # Group importance of the fixed best single on May labels available before final fitting.
    fold=splits.folds(df,g)[-1];b=joblib.load(C.PRIVATE/f'validation_selected_{fold.name}.joblib')
    score=df.iloc[fold.score].sample(min(3000,len(fold.score)),random_state=C.SEED).copy()
    base=models.metrics(score.lead_time_days,predict_bundle(score,b))['mae'];rng=np.random.default_rng(C.SEED)
    groups={k:list(dict.fromkeys(C.NUMERIC_GROUPS.get(k,[])+C.CAT_GROUPS.get(k,[])))
            for k in set(C.NUMERIC_GROUPS)|set(C.CAT_GROUPS)}
    if desc['pack']=='absolute':groups['calendar']+=['purchase_elapsed_days','purchase_year_month']
    importance=[]
    for group,cols in sorted(groups.items()):
        if desc['pack']=='no_calendar' and group=='calendar':continue
        for repeat in range(5):
            shuffled=score.copy();permutation=rng.permutation(len(score))
            shuffled[cols]=score.iloc[permutation][cols].to_numpy()
            delta=models.metrics(score.lead_time_days,predict_bundle(shuffled,b))['mae']-base
            importance.append({'group':group,'repeat':repeat,'mae_increase':delta,'baseline_mae':base,'n':len(score)})
    it=pd.DataFrame(importance);save_table(it,'permutation_importance_repeats')
    save_table(it.groupby('group').mae_increase.agg(['mean','std']).reset_index().sort_values('mean',ascending=False),'permutation_importance_groups')
    # Assemble one coherent Train-CV-Test table, including no-training references.
    train_test=pd.read_csv(C.TABLES/'train_test_results.csv');cv=pd.read_csv(C.TABLES/'cv_candidates.csv')
    extra=pd.read_csv(C.TABLES/'diagnostic_summary.csv');cv=pd.concat([cv[['candidate','cv_mae','cv_sd']],extra[['candidate','cv_mae','cv_sd']]])
    refs=pd.read_csv(C.TABLES/'cv_reference_folds.csv').groupby('candidate').val_mae.agg(['mean','std']).reset_index().rename(columns={'mean':'cv_mae','std':'cv_sd'})
    cv=pd.concat([cv,refs]).drop_duplicates('candidate')
    stack=read('stacking_complete.json');cv=pd.concat([cv,pd.DataFrame([{'candidate':'Convex_stack','cv_mae':stack['cv_mae'],'cv_sd':stack['cv_sd']}])])
    matrix=train_test.pivot(index='candidate',columns='split',values='mae').reset_index().rename(columns={'train':'train_mae','test':'test_mae'})
    matrix=matrix.merge(cv,on='candidate',how='outer');matrix=matrix[matrix.candidate.isin(selection['test_comparisons']+['Convex_stack'])]
    pooled_frames=[pd.read_csv(C.TABLES/'cv_all_candidates_folds.csv'),pd.read_csv(C.TABLES/'cv_reference_folds.csv'),
                   pd.read_csv(C.TABLES/'diagnostic_folds.csv'),pd.read_csv(C.TABLES/'stacking_folds.csv')]
    pooled=[]
    for candidate,part in pd.concat(pooled_frames,ignore_index=True).groupby('candidate'):
        pooled.append({'candidate':candidate,'pooled_cv_mae':float(np.sum(part.val_n*part.val_mae)/part.val_n.sum()),
                       'pooled_cv_rmse':float(np.sqrt(np.sum(part.val_n*np.square(part.val_rmse))/part.val_n.sum()))})
    matrix=matrix.merge(pd.DataFrame(pooled),on='candidate',how='left')
    save_table(matrix,'train_cv_test_comparison')
    # A tidy companion exposes all requested metrics without overloading the short MAE table.
    detail=[]
    all_folds=pd.concat(pooled_frames,ignore_index=True)
    for name in selection['test_comparisons']+['Convex_stack']:
        for r in train_test[train_test.candidate==name].to_dict('records'):detail.append(r)
        part=all_folds[all_folds.candidate==name]
        if part.empty:continue
        row={'candidate':name,'split':'CV monthly mean','n':int(part.val_n.sum())}
        for metric in ['mae','rmse','r2','bias','within3']:
            row[metric]=float(part['val_'+metric].mean())
            row[metric+'_monthly_sd']=float(part['val_'+metric].std(ddof=1))
        detail.append(row)
    save_table(pd.DataFrame(detail),'model_metrics_by_stage')
    data.save_json(C.OUT/'analysis_complete.json',{'importance_n':len(score),'importance_repeats':5,
        'importance_scope':'May validation; outcomes observed before Jun 1; conditional on completed deliveries.',
        'future_outcome_diagnostics_used_for_selection':False})
    log('Post-freeze error and interpretation tables written; selection unchanged.')

def reference():
    audit.require_pass()
    if (C.OUT/'reference_complete.json').exists():return
    df=data.load();ids=np.arange(len(df));tr,te=train_test_split(ids,test_size=.33,random_state=33)
    desc=read('frozen_selection.json')['best_single']
    bundle,work=fit_descriptor(df.iloc[tr],pd.Timestamp('2100-01-01'),desc,'random_reference')
    rows=[]
    for name,frame in [('train',df.iloc[tr]),('test',df.iloc[te])]:
        p=models.predict(bundle,work) if name=='train' and desc['pack']=='risk' else predict_bundle(frame,bundle)
        rows.append({'procedure':'Random 67/33 reference','split':name,**models.metrics(frame.lead_time_days,p)})
    save_table(pd.DataFrame(rows),'random_reference')
    main=read('test_summary.json');single=pd.read_csv(C.TABLES/'train_test_results.csv')
    s=single[(single.candidate==desc['candidate'])&(single.split=='test')].iloc[0]
    comparison=pd.DataFrame([{'procedure':'Chronological main; best single','candidate':desc['candidate'],
        'n':int(s.n),'mae':s.mae,'rmse':s.rmse,'r2':s.r2},
        {'procedure':'Random reference; fixed chronological configuration','candidate':desc['candidate'],**rows[-1]}])
    # Avoid dictionary overwrite of the readable procedure label.
    comparison.loc[1,'procedure']='Random reference; fixed chronological configuration'
    save_table(comparison,'chronological_vs_random')
    data.save_json(C.OUT/'reference_complete.json',{'seed':33,'train_fraction':.67,'cohort_n':len(df),
        'chronological_configuration_reused':True,'used_for_selection':False,
        'limitation':'Different training sizes, maturity filters and test populations; score difference is not isolated algorithm improvement.'})
    log('Fixed-configuration random reference complete; final selection unchanged.')

def run(stage):
    functions={'freeze':freeze,'test':evaluate,'analysis':analysis,'reference':reference}
    if stage in functions:return functions[stage]()
    from . import reporting
    return reporting.run(stage)
