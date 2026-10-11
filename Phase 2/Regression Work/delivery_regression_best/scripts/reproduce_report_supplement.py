"""Rebuild fixed report models, without parameter search or model selection.

Default: train/test scores and HGB interpretation. --with-cv also independently
rebuilds all fifteen rows' five-fold scores, with nested risk cross-fitting.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import reproduce_best as original
import random_experiments as ex
import tree_extensions as ext
import time_risk_features as features
import report_supplement as report


def fit_model(config,train,risk,risk_bundle):
    specification=config['specification']
    form=config['form']
    if form=='standard':
        fields=ex.schema(specification)[2]
        model=ex.build_pipeline(specification).fit(train[fields],train.lead_time_days)
        return model,np.maximum(0,model.predict(train[fields]))
    if form=='extension':
        prep=ext.ExtensionPreprocessor(groups=specification['groups'],encoding=specification['encoding']).fit(train)
        learner=HistGradientBoostingRegressor(random_state=33,early_stopping=False,
            categorical_features=prep.categorical_indices(),**specification['params']).fit(
                ext.transform(prep,train),train.lead_time_days)
        from sklearn.pipeline import Pipeline
        model=Pipeline([('preprocess',prep),('regressor',learner)])
        return model,np.maximum(0,model.predict(train))
    prep=(ext.ExtensionPreprocessor(groups=list(ext.GROUPS),encoding='onehot')
        if config['uses_purchase_extension'] else ex.preprocessor(specification)).fit(train)
    time=features.TimeEncoder().fit(train)
    scale=StandardScaler().fit(risk.reshape(-1,1))
    matrix=features.add_blocks(ext.transform(prep,train),train,config['pack'],time,risk,scale)
    learner=ex.estimator(specification).fit(matrix,train.lead_time_days.to_numpy())
    model=features.InferenceBundle(prep,learner,config['pack'],time,scale,risk_bundle)
    return model,np.maximum(0,learner.predict(matrix))


def predict(config,model,score):
    frame=score[ex.schema(config['specification'])[2]] if config['form']=='standard' else score
    return np.maximum(0,model.predict(frame))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--with-cv',action='store_true')
    parser.add_argument('--skip-importance',action='store_true')
    args=parser.parse_args()
    dest=args.output.resolve() if args.output.is_absolute() else (ROOT/args.output).resolve()
    if (ROOT/'runs').resolve() not in dest.parents: parser.error('Output must be a new directory under runs/.')
    if dest.exists(): parser.error('Use a new output directory.')
    original.verify_bundle(ROOT)
    dest.mkdir(parents=True)
    specs=json.loads((ROOT/'config/report_models.json').read_text())
    features.RISK_PARAMS=specs['risk_params']
    with threadpool_limits(limits=4):
        train,test=original.prepare(args.archive.resolve(),dest)
        oofs={config['model']:[] for config in specs['models']}
        risk_oof=np.full(len(train),np.nan)
        features.DEST=dest/'risk_cache'
        (features.DEST/'data').mkdir(parents=True)
        for fold in range(1,6):
            mask=train.oof_fold.eq(fold).to_numpy()
            fitting,scoring=train.loc[~mask].copy(),train.loc[mask].copy()
            if args.with_cv:
                risk=features.outer_risk(train,fold)
                fit_risk,score_risk=risk['fit_risk'],risk['score_risk']
                risk_bundle=risk['outer_bundle']
                for config in specs['models']:
                    if 'reference' in config:
                        values=report.reference_predictions(fitting,scoring,config['reference'])
                    else:
                        model,_=fit_model(config,fitting,fit_risk,risk_bundle)
                        values=predict(config,model,scoring)
                    oofs[config['model']].append(pd.DataFrame({'order_id':scoring.order_id,
                        'fold':fold,'actual_days':scoring.lead_time_days,'predicted_days':values}))
            else:
                score_risk,_,_=features.risk_fit(fitting,scoring)
            risk_oof[mask]=score_risk
            print('Risk/optional regression CV fold',fold,'complete',flush=True)
        assert np.isfinite(risk_oof).all()
        _,risk_bundle,_=features.risk_fit(train,test)
        records=[]
        train_test_rows=[]
        for config in specs['models']:
            if 'reference' in config:
                train_p=report.reference_predictions(train,train,config['reference'])
                test_p=report.reference_predictions(train,test,config['reference'])
            else:
                model,train_p=fit_model(config,train,risk_oof,risk_bundle)
                test_p=predict(config,model,test)
                if config['model']=='Final HGB (63 leaves)':
                    best=model
                    joblib.dump(best,dest/'selected_model.joblib')
            for phase,y,p in [('train',train.lead_time_days,train_p),('test',test.lead_time_days,test_p)]:
                train_test_rows.append({'model':config['model'],'phase':phase,**report.metrics(y,p)})
            if args.with_cv:
                records.append({**config,'train_actual':train.lead_time_days,'train_prediction':train_p,
                    'test_actual':test.lead_time_days,'test_prediction':test_p,
                    'oof':pd.concat(oofs[config['model']]).sort_values('order_id')})
            print('Fixed fit',config['model'],'complete',flush=True)
        tables=dest/'tables'
        tables.mkdir()
        frame=pd.DataFrame(train_test_rows)
        frame.to_csv(tables/'train_test_reproduction.csv',index=False)
        expected=pd.read_csv(ROOT/'results/report_supplement/tables/metrics_by_phase.csv')
        for row in frame.to_dict('records'):
            reference=expected.loc[expected.model.eq(row['model']) & expected.phase.eq(row['phase'])].iloc[0]
            for metric in ['MAE_days','RMSE_days','R2','bias_days','within_3_days_pct']:
                np.testing.assert_allclose(row[metric],reference[metric],atol=1e-8,rtol=0)
        if args.with_cv:
            comparison=report.export_comparison(records,tables)
            recorded=pd.read_csv(ROOT/'results/report_supplement/tables/train_cv_test_comparison.csv')
            np.testing.assert_allclose(comparison.CV_MAE_mean_days,recorded.CV_MAE_mean_days,atol=1e-8,rtol=0)
        if not args.skip_importance:
            _,protocol=report.grouped_permutation(best,test,tables)
            ext.save(dest/'importance_protocol.json',protocol)
        report.error_diagnostics(train,test,best.predict(test),ext.raw_tables(args.archive.resolve())[0]['orders'],tables)
        ext.save(dest/'reproduction_verification.json',{'fixed_models':15,'train_and_test_match_published':True,
            'CV_recomputed':args.with_cv,'CV_matches_published':True if args.with_cv else None,
            'no_tuning_or_selection':True,'risk_training_features_cross_fitted':True})
    print('Fixed report supplement reproduced without model selection',flush=True)


if __name__=='__main__': main()
