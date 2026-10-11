"""Matched date/risk feature ablation. No stacking or specialist regression.

Run by importing this module, to keep saved inference bundles reloadable.
Risk training features use nested cross-fitting, never global OOF reuse.
"""
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import random_experiments as ex
import tree_extensions as ext
import purchase_risk_diagnostics as diag

ROOT = ext.ROOT
DEST = ex.DEST / 'experiments/11_time_and_risk_features'
PACKS = ['original', 'time', 'risk', 'time_risk']
RISK_PARAMS = diag.PROBES['hgb_all']['params']


def specs():
    return {
        'OLS': ex.read_specs('baseline')['baseline_linear'],
        'Ridge': ex.read_specs('B')['B_ridge_F1_mi100'],
        'Tree': ex.read_specs('baseline')['baseline_tree_d8'],
        'RF_log': ex.read_specs('C')['C_rf_logy_F1'],
        'HGB_best': ext.control_spec(),
    }


def initialize():
    for name in ['data', 'models', 'outputs/oof', 'outputs/predictions', 'outputs/tables', 'outputs/figures']:
        (DEST / name).mkdir(parents=True, exist_ok=True)
    path = DEST / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())['source_sha256'] == ext.sha(__file__)
        preserve_check()
        return
    prior = diag.preserve()
    for name in ['07_stacking', '10_purchase_risk_diagnosis']:
        for p in (ex.DEST / 'experiments' / name).rglob('*'):
            if p.is_file():
                prior[str(p.relative_to(ROOT))] = ext.sha(p)
    ext.save(DEST / 'preserved_hashes.json', prior)
    ext.save(path, {'date': '2026-10-10', 'source_sha256': ext.sha(__file__),
        'input_sha256': ext.INPUT_SHA, 'archive_sha256': ext.ARCHIVE_SHA,
        'manifest_sha256': ext.sha(ex.DEST / 'data/cv_manifest.csv'),
        'split_manifest_sha256': ext.sha(ex.DEST / 'data/split_manifest.csv'),
        'models': specs(), 'feature_packs': PACKS,
        'base_features': {'OLS': 'original F0', 'Tree': 'original F0',
            'Ridge': 'original log-input F1', 'RF_log': 'original F1, log1p target',
            'HGB_best': '09 F_all_onehot; all seller/route/product groups'},
        'time_features': ['purchase_elapsed_days since fixed 2016-01-01', 'purchase_year_month'],
        'time_encoding': 'training-only standard scaling; year-month onehot min_frequency=20 drop=first; unseen ignored',
        'risk_feature': 'continuous P(lead_time_days > 30) from fixed all-purchase HGB classifier',
        'risk_classifier': RISK_PARAMS, 'risk_input': '09 F_all_onehot, no new time fields',
        'outer_cv': 'unchanged existing five folds, seed 33',
        'inner_cv': '3 shuffled KFold splits, seed 33; inside each outer regression fit only',
        'full_training_risk_cv': 'existing five outer folds; each training row predicted without its fold',
        'train_n': 64634, 'test_n': 31836, 'train_test_split': 'existing random 67/33, seed 33',
        'selection': 'per-model minimum mean outer-CV MAE, tie original first; freeze before test scoring',
        'test_models': 'CV winner and original control only, per model; do not select by test',
        'test_previously_exposed': True, 'development_folds_previously_used': True,
        'risk_is_learning_step': True, 'stacking': False, 'specialist_regressor': False,
        'dates_caveat': 'archive/completed-cohort effects possible; random-split date improvement is not future performance',
        'purchase_availability_assumptions': 'archived seller/product/quote attributes assumed available at purchase',
        'versions': ex.environment()})


def preserve_check():
    for rel, sha in json.loads((DEST / 'preserved_hashes.json').read_text()).items():
        assert ext.sha(ROOT / rel) == sha, rel
    diag.check_preservation()


class TimeEncoder:
    @staticmethod
    def fields(frame):
        dates = pd.to_datetime(frame.purchase_ts)
        assert dates.notna().all()
        return ((dates - pd.Timestamp('2016-01-01')).dt.total_seconds().to_numpy().reshape(-1, 1) / 86400,
                dates.dt.strftime('%Y-%m').to_numpy().reshape(-1, 1))

    def fit(self, frame):
        elapsed, month = self.fields(frame)
        self.scale = StandardScaler().fit(elapsed)
        self.month = OneHotEncoder(handle_unknown='ignore', min_frequency=20, drop='first',
                                   sparse_output=False).fit(month)
        return self

    def transform(self, frame):
        elapsed, month = self.fields(frame)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Found unknown categories.*')
            return np.column_stack([self.scale.transform(elapsed), self.month.transform(month)])


def risk_fit(fit, score):
    assert set(fit.order_id).isdisjoint(score.order_id)
    prep = ext.ExtensionPreprocessor(groups=list(ext.GROUPS), encoding='onehot').fit(fit)
    learner = HistGradientBoostingClassifier(**RISK_PARAMS).fit(
        ext.transform(prep, fit), fit.lead_time_days.gt(30).astype(int))
    prob = learner.predict_proba(ext.transform(prep, score))[:, 1]
    audit = {'fit_n': len(fit), 'score_n': len(score),
             'fit_ids_sha256': ext.ids_sha(fit.order_id), 'score_ids_sha256': ext.ids_sha(score.order_id),
             'overlap': 0, 'positive_fit_n': int(fit.lead_time_days.gt(30).sum()),
             'fit_numeric_medians': prep.numeric_.named_steps['impute'].statistics_.tolist()}
    return prob, (prep, learner), audit


def risk_predict(bundle, frame):
    prep, learner = bundle
    return learner.predict_proba(ext.transform(prep, frame))[:, 1]


def outer_risk(train, fold):
    fit = train.loc[train.oof_fold.ne(fold)].copy()
    score = train.loc[train.oof_fold.eq(fold)].copy()
    cache = DEST / f'data/risk_fold_{fold}.joblib'
    if cache.exists():
        value = joblib.load(cache)
        assert value['fit_ids'] == fit.order_id.tolist() and value['score_ids'] == score.order_id.tolist()
        return value
    train_prob = np.full(len(fit), np.nan)
    audits = []
    for inner, (a, b) in enumerate(KFold(3, shuffle=True, random_state=33).split(fit), 1):
        p, _, audit = risk_fit(fit.iloc[a], fit.iloc[b])
        train_prob[b] = p
        audits.append({'outer_fold': fold, 'inner_fold': inner, **audit})
        print(f'Risk outer {fold}, inner {inner}: complete', flush=True)
    score_prob, bundle, audit = risk_fit(fit, score)
    audits.append({'outer_fold': fold, 'inner_fold': 'full', **audit})
    assert np.isfinite(train_prob).all()
    value = {'fit_ids': fit.order_id.tolist(), 'score_ids': score.order_id.tolist(),
             'fit_risk': train_prob, 'score_risk': score_prob, 'audits': audits,
             'outer_bundle': bundle}
    joblib.dump(value, cache)
    return value


def prep_for(name, fit):
    if name == 'HGB_best':
        return ext.ExtensionPreprocessor(groups=list(ext.GROUPS), encoding='onehot').fit(fit)
    return ex.preprocessor(specs()[name]).fit(fit)


def add_blocks(base_x, frame, pack, time_encoder, risk, risk_scaler):
    blocks = [base_x]
    if pack in ['time', 'time_risk']:
        blocks.append(time_encoder.transform(frame))
    if pack in ['risk', 'time_risk']:
        blocks.append(risk_scaler.transform(np.asarray(risk).reshape(-1, 1)))
    return np.column_stack(blocks)


class InferenceBundle:
    """Inference consumes only whitelisted purchase inputs; outcomes are ignored."""
    def __init__(self, prep, regressor, pack, time_encoder, risk_scaler, risk_bundle):
        self.prep, self.regressor, self.pack = prep, regressor, pack
        self.time_encoder, self.risk_scaler, self.risk_bundle = time_encoder, risk_scaler, risk_bundle

    def predict(self, frame):
        risk = risk_predict(self.risk_bundle, frame) if self.pack in ['risk', 'time_risk'] else None
        x = add_blocks(ext.transform(self.prep, frame), frame, self.pack, self.time_encoder, risk, self.risk_scaler)
        return np.maximum(0, self.regressor.predict(x))


def group_rows(model, pack, phase, actual, pred):
    rows = []
    for label, lo, hi in [('0-14', -1, 14), ('14-30', 14, 30), ('30-60', 30, 60), ('>60', 60, np.inf)]:
        mask = (actual > lo) & (actual <= hi)
        rows.append({'model': model, 'pack': pack, 'phase': phase, 'group': label,
                     **ex.metrics(actual[mask], pred[mask])})
    return rows


def run_cv():
    initialize()
    assert not (DEST / 'outputs/frozen_selection.json').exists(), 'Completed CV is preserved.'
    train = ext.load_partition('train')
    assert len(train) == 64634
    rows, audits = [], []
    with threadpool_limits(limits=4):
        for fold in range(1, 6):
            fit = train.loc[train.oof_fold.ne(fold)].copy()
            score = train.loc[train.oof_fold.eq(fold)].copy()
            risk = outer_risk(train, fold)
            audits.extend(risk['audits'])
            date = TimeEncoder().fit(fit)
            scale = StandardScaler().fit(risk['fit_risk'].reshape(-1, 1))
            for name, spec in specs().items():
                prep = prep_for(name, fit)
                fit_x, score_x = ext.transform(prep, fit), ext.transform(prep, score)
                for pack in PACKS:
                    checkpoint = DEST / f'data/regression_{fold}_{name}_{pack}.json'
                    if checkpoint.exists():
                        rows.append(json.loads(checkpoint.read_text()))
                        continue
                    started = time.perf_counter()
                    a = add_blocks(fit_x, fit, pack, date, risk['fit_risk'], scale)
                    b = add_blocks(score_x, score, pack, date, risk['score_risk'], scale)
                    reg = ex.estimator(spec).fit(a, fit.lead_time_days.to_numpy())
                    p = np.maximum(0, reg.predict(b))
                    fit_p = np.maximum(0, reg.predict(a))
                    row = {'model': name, 'pack': pack, 'fold': fold, **ex.metrics(score.lead_time_days, p),
                           'fit_MAE': float(np.abs(fit_p - fit.lead_time_days.to_numpy()).mean()),
                           'columns': a.shape[1], 'seconds': time.perf_counter() - started}
                    rows.append(row)
                    pd.DataFrame({'order_id': score.order_id.to_numpy(), 'fold': fold,
                                  'actual_days': score.lead_time_days.to_numpy(), 'predicted_days': p}).to_csv(
                        DEST / f'outputs/oof/{name}_{pack}_fold{fold}.csv.gz', index=False)
                    ext.save(checkpoint, row)
                    print(f"CV {fold}/5 {name} {pack}: MAE {row['MAE_days']:.4f} ({row['seconds']:.1f}s)", flush=True)
                    del a, b, reg
    frame = pd.DataFrame(rows)
    frame.to_csv(DEST / 'outputs/tables/cv_folds.csv', index=False)
    summary = frame.groupby(['model', 'pack'], sort=False).agg(
        CV_MAE=('MAE_days', 'mean'), CV_MAE_SD=('MAE_days', 'std'), CV_RMSE=('RMSE_days', 'mean'),
        fit_MAE=('fit_MAE', 'mean')).reset_index()
    summary.to_csv(DEST / 'outputs/tables/cv_summary.csv', index=False)
    groups = []
    for name in specs():
        for pack in PACKS:
            parts = [pd.read_csv(DEST / f'outputs/oof/{name}_{pack}_fold{f}.csv.gz') for f in range(1, 6)]
            p = pd.concat(parts).sort_values('order_id').reset_index(drop=True)
            assert p.order_id.tolist() == train.order_id.tolist()
            p.to_csv(DEST / f'outputs/oof/{name}_{pack}.csv.gz', index=False)
            groups.extend(group_rows(name, pack, 'OOF', p.actual_days.to_numpy(), p.predicted_days.to_numpy()))
    pd.DataFrame(groups).to_csv(DEST / 'outputs/tables/oof_duration_groups.csv', index=False)
    ext.save(DEST / 'outputs/risk_fit_audit.json', audits)
    selected = {name: summary.loc[summary.model.eq(name)].sort_values('CV_MAE', kind='stable').iloc[0]['pack']
                for name in specs()}
    ext.save(DEST / 'outputs/frozen_selection.json', {'selected': selected, 'selected_by': 'mean training CV MAE',
        'before_this_stage_test_scoring': True, 'test_previously_exposed': True,
        'summary_sha256': ext.sha(DEST / 'outputs/tables/cv_summary.csv'),
        'source_sha256': ext.sha(__file__), 'test_scoring_not_started': True})
    preserve_check()
    print('Frozen:', selected, flush=True)


def run_test():
    initialize()
    frozen = json.loads((DEST / 'outputs/frozen_selection.json').read_text())
    assert ext.sha(DEST / 'outputs/tables/cv_summary.csv') == frozen['summary_sha256']
    assert (DEST / 'outputs/cv_verification.json').exists(), 'Verify CV independently before heldout scoring.'
    assert not (DEST / 'outputs/test_complete.json').exists()
    train = ext.load_partition('train')
    score = ext.load_partition('test')
    assert set(train.order_id).isdisjoint(score.order_id)
    # Each outer-full classifier already trained excluding exactly its score fold.
    full_risk = np.full(len(train), np.nan)
    for fold in range(1, 6):
        risk = joblib.load(DEST / f'data/risk_fold_{fold}.joblib')
        mask = train.oof_fold.eq(fold).to_numpy()
        assert train.loc[mask, 'order_id'].tolist() == risk['score_ids']
        full_risk[mask] = risk['score_risk']
    with threadpool_limits(limits=4):
        score_risk, risk_bundle, audit = risk_fit(train, score)
        ext.save(DEST / 'outputs/full_risk_audit.json', audit)
        pd.DataFrame({'order_id': train.order_id, 'risk_p30': full_risk}).to_csv(DEST / 'data/final_train_risk.csv.gz', index=False)
        pd.DataFrame({'order_id': score.order_id, 'risk_p30': score_risk}).to_csv(DEST / 'data/final_test_risk.csv.gz', index=False)
        date = TimeEncoder().fit(train)
        scale = StandardScaler().fit(full_risk.reshape(-1, 1))
        rows, groups = [], []
        for name, spec in specs().items():
            prep = prep_for(name, train)
            x = ext.transform(prep, train)
            for pack in dict.fromkeys(['original', frozen['selected'][name]]):
                a = add_blocks(x, train, pack, date, full_risk, scale)
                learner = ex.estimator(spec).fit(a, train.lead_time_days.to_numpy())
                bundle = InferenceBundle(prep, learner, pack, date, scale,
                                         risk_bundle if pack in ['risk', 'time_risk'] else None)
                path = DEST / f'models/{name}_{pack}.joblib'
                joblib.dump(bundle, path)
                p = joblib.load(path).predict(score)
                np.testing.assert_allclose(p, bundle.predict(score), atol=1e-10, rtol=0)
                mutated = score.head(80).drop(columns=['lead_time_days', 'delivered_ts', 'order_status', 'estimated_ts'])
                np.testing.assert_allclose(bundle.predict(mutated), p[:80], atol=1e-10, rtol=0)
                rows.append({'model': name, 'pack': pack, 'role': 'CV winner' if pack == frozen['selected'][name] else 'original control',
                             **ex.metrics(score.lead_time_days, p)})
                groups.extend(group_rows(name, pack, 'test', score.lead_time_days.to_numpy(), p))
                pd.DataFrame({'order_id': score.order_id, 'actual_days': score.lead_time_days,
                              'predicted_days': p}).to_csv(DEST / f'outputs/predictions/{name}_{pack}.csv.gz', index=False)
                print(f'Test {name} {pack}: MAE {rows[-1]["MAE_days"]:.4f}', flush=True)
        pd.DataFrame(rows).to_csv(DEST / 'outputs/tables/test_metrics.csv', index=False)
        pd.DataFrame(groups).to_csv(DEST / 'outputs/tables/test_duration_groups.csv', index=False)
    preserve_check()
    ext.save(DEST / 'outputs/test_complete.json', {'stacking': False, 'no_test_selection': True,
        'test_metrics_sha256': ext.sha(DEST / 'outputs/tables/test_metrics.csv')})
