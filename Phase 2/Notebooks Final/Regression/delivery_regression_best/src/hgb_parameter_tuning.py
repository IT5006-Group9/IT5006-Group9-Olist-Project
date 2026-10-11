"""Bounded pre-stacking HGB tuning on fixed, validated time/risk inputs.

Stage A: eight predefined parameter settings on time+risk. Stage B: the
Stage A winner's parameters on time-only, alongside the original controls.
Select by existing training CV; never select by previously exposed test.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import time_risk_features as features

ROOT = features.ROOT
PREVIOUS = features.DEST
DEST = features.ex.DEST / 'experiments/12_hgb_parameter_tuning'
THREADS = 2


def configurations():
    original = copy.deepcopy(features.specs()['HGB_best']['params'])
    differences = {
        'control': {}, 'leaves15': {'max_leaf_nodes': 15},
        'leaves63': {'max_leaf_nodes': 63}, 'leaf15': {'min_samples_leaf': 15},
        'leaf60': {'min_samples_leaf': 60}, 'l2zero': {'l2_regularization': 0},
        'rounds600': {'max_iter': 600}, 'rate010': {'learning_rate': .1},
    }
    return {name: {**original, **change} for name, change in differences.items()}


def preserve_check():
    for rel, digest in json.loads((DEST / 'preserved_hashes.json').read_text()).items():
        assert features.ext.sha(ROOT / rel) == digest, rel
    features.preserve_check()


def initialize():
    for name in ['data', 'models', 'outputs/oof', 'outputs/predictions', 'outputs/tables', 'outputs/figures']:
        (DEST / name).mkdir(parents=True, exist_ok=True)
    path = DEST / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['source_sha256'] == features.ext.sha(__file__)
        preserve_check()
        return
    assert (PREVIOUS / 'outputs/verification.json').exists()
    assert all((PREVIOUS / f'data/risk_fold_{fold}.joblib').exists() for fold in range(1, 6))
    features.preserve_check()
    files = [p for p in PREVIOUS.rglob('*') if p.is_file()]
    files += [ROOT / 'src/time_risk_features.py', ROOT / 'src/tree_extensions.py', ROOT / 'src/random_experiments.py',
              features.ex.DEST / 'data/cv_manifest.csv', features.ex.DEST / 'data/split_manifest.csv',
              ROOT / 'versions/preparation_v2/data/eligible_orders.csv']
    features.ext.save(DEST / 'preserved_hashes.json', {str(p.relative_to(ROOT)): features.ext.sha(p) for p in files})
    features.ext.save(path, {
        'date': '2026-10-10', 'source_sha256': features.ext.sha(__file__),
        'input_sha256': features.ext.INPUT_SHA, 'archive_sha256': features.ext.ARCHIVE_SHA,
        'cv_manifest_sha256': features.ext.sha(features.ex.DEST / 'data/cv_manifest.csv'),
        'split_manifest_sha256': features.ext.sha(features.ex.DEST / 'data/split_manifest.csv'),
        'configurations': configurations(), 'train_n': 64634, 'test_n': 31836,
        'random_split': 'fixed 67/33, seed33', 'outer_cv': 'unchanged five development folds',
        'stage_A': '8 settings on time_risk; 40 CV rows, 5 original controls reused, 35 new fits',
        'stage_B': 'Stage A CV winner parameters on time-only; original time-only control reused; at most 5 new fits',
        'selection': 'minimum mean five-fold training MAE across A and B; original control first on exact tie',
        'risk_generation': 'reuse hash-verified per-outer-fold nested three-fold cross-fitted training risks; not global OOF',
        'fixed_risk_classifier': features.RISK_PARAMS, 'classifier_tuning': False,
        'features': '09 all-onehot + 11 fixed purchase elapsed days/year-month + optional continuous p30',
        'loss': 'absolute_error fixed; no reweighting or target/cohort changes',
        'preprocessing': 'fold-local original numeric/category/time transforms and risk scaling unchanged',
        'concurrency': 'two independent settings, two threads each; no simultaneous dataset mutation',
        'test_scoring': 'only frozen winner plus previous matched time_risk and time controls',
        'test_previously_exposed': True, 'development_and_adaptive_selection': True,
        'not_unbiased_nested_model_selection': True,
        'stacking': False, 'upload': False, 'no_new_models_or_families': True,
        'assumptions': 'purchase archive attributes available; completed cohort; historical random mixture, not future performance',
        'software': features.ex.environment()})


def fold_data(train, fold):
    fit = train.loc[train.oof_fold.ne(fold)].copy()
    score = train.loc[train.oof_fold.eq(fold)].copy()
    risk = joblib.load(PREVIOUS / f'data/risk_fold_{fold}.joblib')
    assert risk['fit_ids'] == fit.order_id.tolist() and risk['score_ids'] == score.order_id.tolist()
    prep = features.prep_for('HGB_best', fit)
    time_encoder = features.TimeEncoder().fit(fit)
    scaler = StandardScaler().fit(risk['fit_risk'].reshape(-1, 1))
    a, b = features.ext.transform(prep, fit), features.ext.transform(prep, score)
    return fit, score, {
        pack: (features.add_blocks(a, fit, pack, time_encoder, risk['fit_risk'], scaler),
               features.add_blocks(b, score, pack, time_encoder, risk['score_risk'], scaler))
        for pack in ['time_risk', 'time']}, {
        'fold': fold, 'fit_n': len(fit), 'score_n': len(score),
        'fit_ids_sha256': features.ext.ids_sha(fit.order_id), 'score_ids_sha256': features.ext.ids_sha(score.order_id),
        'risk_cache_sha256': features.ext.sha(PREVIOUS / f'data/risk_fold_{fold}.joblib'),
        'date_fit_mean': float(time_encoder.scale.mean_[0]),
        'date_categories': time_encoder.month.categories_[0].tolist(),
        'risk_fit_mean': float(scaler.mean_[0]), 'columns': {pack: x[0].shape[1] for pack, x in
            {'time_risk': (features.add_blocks(a[:1], fit.head(1), 'time_risk', time_encoder, risk['fit_risk'][:1], scaler),),
             'time': (features.add_blocks(a[:1], fit.head(1), 'time', time_encoder, risk['fit_risk'][:1], scaler),)}.items()}}


def candidate_id(config, pack):
    return f'{config}__{pack}'


def reuse_control(config, pack, fold):
    identifier = candidate_id(config, pack)
    checkpoint = DEST / f'data/{identifier}_fold{fold}.json'
    if checkpoint.exists():
        return json.loads(checkpoint.read_text())
    old = pd.read_csv(PREVIOUS / 'outputs/tables/cv_folds.csv')
    row = old.loc[old.model.eq('HGB_best') & old.pack.eq(pack) & old.fold.eq(fold)].iloc[0].to_dict()
    row.update({'candidate': identifier, 'configuration': config, 'stage': 'A' if pack == 'time_risk' else 'B',
                'reused_original_control': True, 'seconds': 0.0})
    predictions = pd.read_csv(PREVIOUS / f'outputs/oof/HGB_best_{pack}_fold{fold}.csv.gz')
    predictions.to_csv(DEST / f'outputs/oof/{identifier}_fold{fold}.csv.gz', index=False)
    features.ext.save(checkpoint, row)
    return row


def fit_setting(config, pack, fold, fit, score, arrays, stage):
    identifier = candidate_id(config, pack)
    checkpoint = DEST / f'data/{identifier}_fold{fold}.json'
    if checkpoint.exists():
        return json.loads(checkpoint.read_text())
    started = time.perf_counter()
    a, b = arrays[pack]
    learner = HistGradientBoostingRegressor(random_state=33, early_stopping=False, **configurations()[config])
    learner.fit(a, fit.lead_time_days.to_numpy())
    prediction = np.maximum(0, learner.predict(b))
    fitted = np.maximum(0, learner.predict(a))
    row = {'candidate': identifier, 'configuration': config, 'pack': pack, 'fold': fold, 'stage': stage,
           'reused_original_control': False, **features.ex.metrics(score.lead_time_days, prediction),
           'fit_MAE': float(np.abs(fitted - fit.lead_time_days.to_numpy()).mean()),
           'columns': a.shape[1], 'seconds': time.perf_counter() - started}
    pd.DataFrame({'order_id': score.order_id.to_numpy(), 'fold': fold,
                  'actual_days': score.lead_time_days.to_numpy(), 'predicted_days': prediction}).to_csv(
        DEST / f'outputs/oof/{identifier}_fold{fold}.csv.gz', index=False)
    features.ext.save(checkpoint, row)
    print(f"CV {stage} {fold}/5 {identifier}: MAE {row['MAE_days']:.4f}; {row['seconds']:.1f}s", flush=True)
    return row


def summarize(rows):
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['candidate', 'configuration', 'pack'], sort=False).agg(
        CV_MAE=('MAE_days', 'mean'), CV_SD=('MAE_days', 'std'), CV_RMSE=('RMSE_days', 'mean'),
        fit_MAE=('fit_MAE', 'mean'), folds=('fold', 'count')).reset_index()
    assert summary.folds.eq(5).all()
    return frame, summary


def run_cv():
    initialize()
    assert not (DEST / 'outputs/frozen_selection.json').exists(), 'Completed experiments are preserved.'
    train = features.ext.load_partition('train')
    rows, audits = [], []
    with threadpool_limits(limits=THREADS):
        for fold in range(1, 6):
            fit, score, arrays, audit = fold_data(train, fold)
            audits.append(audit)
            rows.append(reuse_control('control', 'time_risk', fold))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(fit_setting, name, 'time_risk', fold, fit, score, arrays, 'A')
                           for name in configurations() if name != 'control']
                rows.extend(f.result() for f in futures)
            del arrays
        frame, summary = summarize(rows)
        frame.to_csv(DEST / 'outputs/tables/stage_A_folds.csv', index=False)
        summary.to_csv(DEST / 'outputs/tables/stage_A_summary.csv', index=False)
        best = summary.sort_values('CV_MAE', kind='stable').iloc[0].configuration
        features.ext.save(DEST / 'outputs/stage_A_selection.json', {'configuration': best,
            'by': 'training five-fold MAE', 'before_test': True, 'summary_sha256': features.ext.sha(DEST / 'outputs/tables/stage_A_summary.csv')})
        print('Stage A selected:', best, flush=True)
        for fold in range(1, 6):
            rows.append(reuse_control('control', 'time', fold))
            if best != 'control':
                fit, score, arrays, _ = fold_data(train, fold)
                rows.append(fit_setting(best, 'time', fold, fit, score, arrays, 'B'))
                del arrays
    frame, summary = summarize(rows)
    frame.to_csv(DEST / 'outputs/tables/cv_folds.csv', index=False)
    summary.to_csv(DEST / 'outputs/tables/cv_summary.csv', index=False)
    group_rows = []
    for identifier in summary.candidate:
        prediction = pd.concat([pd.read_csv(DEST / f'outputs/oof/{identifier}_fold{fold}.csv.gz')
                                for fold in range(1, 6)]).sort_values('order_id').reset_index(drop=True)
        assert prediction.order_id.tolist() == train.order_id.tolist()
        prediction.to_csv(DEST / f'outputs/oof/{identifier}.csv.gz', index=False)
        config, pack = identifier.split('__')
        group_rows.extend(features.group_rows(config, pack, 'OOF', prediction.actual_days.to_numpy(), prediction.predicted_days.to_numpy()))
    pd.DataFrame(group_rows).to_csv(DEST / 'outputs/tables/oof_duration_groups.csv', index=False)
    features.ext.save(DEST / 'outputs/fold_audit.json', audits)
    winner = summary.sort_values('CV_MAE', kind='stable').iloc[0]
    features.ext.save(DEST / 'outputs/frozen_selection.json', {'candidate': winner.candidate,
        'configuration': winner.configuration, 'pack': winner.pack, 'params': configurations()[winner.configuration],
        'stage_A_best_parameters': best, 'selection_metric': 'training five-fold mean MAE',
        'summary_sha256': features.ext.sha(DEST / 'outputs/tables/cv_summary.csv'),
        'source_sha256': features.ext.sha(__file__), 'frozen_before_test': True,
        'test_previously_exposed': True, 'stacking': False})
    preserve_check()
    print('Final frozen candidate:', winner.candidate, 'CV MAE', winner.CV_MAE, flush=True)


def run_test():
    initialize()
    frozen = json.loads((DEST / 'outputs/frozen_selection.json').read_text())
    assert (DEST / 'outputs/cv_verification.json').exists()
    assert features.ext.sha(DEST / 'outputs/tables/cv_summary.csv') == frozen['summary_sha256']
    assert not (DEST / 'outputs/test_complete.json').exists()
    train = features.ext.load_partition('train')
    score = features.ext.load_partition('test')
    assert set(train.order_id).isdisjoint(score.order_id)
    controls = ['control__time_risk', 'control__time']
    candidates = list(dict.fromkeys([*controls, frozen['candidate']]))
    rows, groups = [], []
    full_risk = pd.read_csv(PREVIOUS / 'data/final_train_risk.csv.gz')
    assert full_risk.order_id.tolist() == train.order_id.tolist()
    with threadpool_limits(limits=THREADS):
        for identifier in candidates:
            config, pack = identifier.split('__')
            if config == 'control' and pack == 'time_risk':
                bundle = joblib.load(PREVIOUS / 'models/HGB_best_time_risk.joblib')
            else:
                # Fixed preprocessing and full-training classifier from verified stage 11.
                template = joblib.load(PREVIOUS / 'models/HGB_best_time_risk.joblib')
                a = features.add_blocks(features.ext.transform(template.prep, train), train, pack,
                    template.time_encoder, full_risk.risk_p30.to_numpy(), template.risk_scaler)
                learner = HistGradientBoostingRegressor(random_state=33, early_stopping=False,
                    **configurations()[config]).fit(a, train.lead_time_days.to_numpy())
                bundle = features.InferenceBundle(template.prep, learner, pack, template.time_encoder,
                    template.risk_scaler, template.risk_bundle if pack == 'time_risk' else None)
                del a, template
            prediction = bundle.predict(score)
            path = DEST / f'models/{identifier}.joblib'
            joblib.dump(bundle, path)
            np.testing.assert_allclose(joblib.load(path).predict(score.head(100)), prediction[:100], atol=1e-10, rtol=0)
            rows.append({'candidate': identifier, 'configuration': config, 'pack': pack,
                'role': 'CV selected' if identifier == frozen['candidate'] else 'predeclared control',
                **features.ex.metrics(score.lead_time_days, prediction)})
            groups.extend(features.group_rows(config, pack, 'test', score.lead_time_days.to_numpy(), prediction))
            pd.DataFrame({'order_id': score.order_id.to_numpy(), 'actual_days': score.lead_time_days.to_numpy(),
                'predicted_days': prediction}).to_csv(DEST / f'outputs/predictions/{identifier}.csv.gz', index=False)
            print('Test', identifier, 'MAE', rows[-1]['MAE_days'], flush=True)
        pd.DataFrame(rows).to_csv(DEST / 'outputs/tables/test_metrics.csv', index=False)
        pd.DataFrame(groups).to_csv(DEST / 'outputs/tables/test_duration_groups.csv', index=False)
    preserve_check()
    features.ext.save(DEST / 'outputs/test_complete.json', {'selected_by_CV': frozen['candidate'],
        'test_previously_exposed': True, 'no_selection_by_test': True, 'stacking': False,
        'metrics_sha256': features.ext.sha(DEST / 'outputs/tables/test_metrics.csv')})
