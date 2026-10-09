"""Rebuild the already selected HGB from the fixed course ZIP, no prior caches.

This reproduces the frozen winner, not the earlier multi-scheme search.
Optional --with-cv rebuilds its five nested-risk regression folds and OOF.
No stacking, resampling, new split, parameter search, or test-driven selection.
"""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import delivery_regression as data
import random_split_protocol as split
import random_experiments as ex
import tree_extensions as ext
import time_risk_features as features
from verify_bundle import verify_bundle


def prepare(archive, destination):
    protocol = json.loads((ROOT / 'config/random_split_protocol.json').read_text())
    tuning = json.loads((ROOT / 'results/protocol.json').read_text())
    purchase = json.loads((ROOT / 'config/purchase_feature_protocol.json').read_text())
    assert ext.sha(archive) == tuning['archive_sha256'], 'Use the fixed course ZIP.'
    if os.environ.get('OLIST_CSV_DIR'):
        raise ValueError('Unset OLIST_CSV_DIR so the verified ZIP is the only data source.')
    data.prepare_data(archive=archive, output_root=destination / 'prepared', geography_policy='screened_median')
    eligible = destination / 'prepared/data/eligible_orders.csv'
    assert ext.sha(eligible) == tuning['input_sha256']
    source = pd.read_csv(eligible, parse_dates=['purchase_ts', 'delivered_ts', 'estimated_ts'])
    assert len(source) == 96470 and source.order_id.is_unique
    assert int(source.lead_time_days.gt(60).sum()) == 306
    manifest = split.make_manifest(source.order_id, protocol)
    folds = split.make_cv_manifest(manifest, protocol)
    manifest.to_csv(destination / 'split_manifest.csv', index=False)
    folds.to_csv(destination / 'cv_manifest.csv', index=False)
    assert ext.sha(destination / 'split_manifest.csv') == tuning['split_manifest_sha256']
    assert ext.sha(destination / 'cv_manifest.csv') == tuning['cv_manifest_sha256']
    tables, _ = ext.raw_tables(archive)
    enriched = ext.aggregate_purchase_features(tables)
    enriched = source[['order_id']].merge(enriched, on='order_id', validate='one_to_one').sort_values('order_id')
    enriched.to_csv(destination / 'purchase_features.csv', index=False)
    checks = json.loads((ROOT / 'config/reproduction_checks.json').read_text())
    assert ext.sha(destination / 'purchase_features.csv') == checks['enrichment_csv_sha256']
    # Match the original CSV parsing, including numeric round trips.
    enriched = pd.read_csv(destination / 'purchase_features.csv')
    outputs = {}
    for role in ['train', 'test']:
        ids = manifest.loc[manifest.split.eq(role), ['order_id', 'split']]
        frame = ids.merge(source.drop(columns='split'), on='order_id', validate='one_to_one')
        frame = ex.add_fixed_features(frame.sort_values('order_id').reset_index(drop=True))
        if role == 'train':
            frame = frame.merge(folds, on='order_id', validate='one_to_one').sort_values('order_id').reset_index(drop=True)
        frame = frame.merge(enriched, on='order_id', validate='one_to_one', how='left')
        assert frame.order_id.tolist() == sorted(ids.order_id)
        outputs[role] = frame
    assert len(outputs['train']) == 64634 and len(outputs['test']) == 31836
    assert set(outputs['train'].order_id).isdisjoint(outputs['test'].order_id)
    ext.save(destination / 'preparation_verification.json', {
        'course_archive_hash_match': True, 'V2_hash_match': True,
        'split_and_cv_manifest_hashes_match': True, 'train_n': 64634, 'test_n': 31836,
        'eligible_n': 96470, 'above_60_days_retained': 306,
        'enrichment_source': 'unchanged aggregate_purchase_features, all original seller/route/product groups',
        'enrichment_csv_hash_match': True,
        'original_enrichment_gzip_sha256': purchase['enrichment_sha256'],
        'stacking': False})
    print('Course ZIP -> V2 -> identical random split/CV -> purchase features verified', flush=True)
    return outputs['train'], outputs['test']


def evaluate(bundle, test, destination):
    prediction = bundle.predict(test)
    rows = ex.metrics(test.lead_time_days, prediction)
    selected = json.loads((ROOT / 'config/selected_model.json').read_text())
    for key, value in selected['params'].items():
        assert bundle.regressor.get_params()[key] == value
    assert bundle.pack == 'time_risk'
    reduced = test.head(120).drop(columns=['lead_time_days', 'delivered_ts', 'estimated_ts', 'order_status'])
    np.testing.assert_allclose(bundle.predict(reduced), prediction[:120], atol=1e-10, rtol=0)
    ext.save(destination / 'test_metrics.json', rows)
    pd.DataFrame({'order_id': test.order_id, 'actual_days': test.lead_time_days,
                  'predicted_days': prediction}).to_csv(destination / 'test_predictions.csv', index=False)
    pd.DataFrame(features.group_rows('leaves63', 'time_risk', 'test', test.lead_time_days.to_numpy(), prediction)).to_csv(
        destination / 'test_duration_groups.csv', index=False)
    print('Frozen selected model test:', rows, flush=True)
    return rows


def fit(train, test, destination, with_cv):
    selected = json.loads((ROOT / 'config/selected_model.json').read_text())
    assert selected['candidate'] == 'leaves63__time_risk'
    params = selected['params']
    risk_oof = np.full(len(train), np.nan)
    regression_oof = np.full(len(train), np.nan)
    cv_rows = []
    features.DEST = destination / 'risk_cache'
    (features.DEST / 'data').mkdir(parents=True)
    for fold in range(1, 6):
        mask = train.oof_fold.eq(fold).to_numpy()
        fitting, scoring = train.loc[~mask].copy(), train.loc[mask].copy()
        assert set(fitting.order_id).isdisjoint(scoring.order_id)
        if with_cv:
            # Inner three-fold training risks exclude each row's own label.
            nested = features.outer_risk(train, fold)
            risk_oof[mask] = nested['score_risk']
            prep = features.prep_for('HGB_best', fitting)
            time = features.TimeEncoder().fit(fitting)
            scale = StandardScaler().fit(nested['fit_risk'].reshape(-1, 1))
            a = features.add_blocks(ext.transform(prep, fitting), fitting, 'time_risk', time, nested['fit_risk'], scale)
            b = features.add_blocks(ext.transform(prep, scoring), scoring, 'time_risk', time, nested['score_risk'], scale)
            learner = HistGradientBoostingRegressor(random_state=33, early_stopping=False, **params).fit(a, fitting.lead_time_days)
            regression_oof[mask] = np.maximum(0, learner.predict(b))
            cv_rows.append({'fold': fold, **ex.metrics(scoring.lead_time_days, regression_oof[mask])})
            del a, b, prep, time, scale, learner
        else:
            risk_oof[mask], _, _ = features.risk_fit(fitting, scoring)
        print('Risk cross-fit fold completed:', fold, flush=True)
    assert np.isfinite(risk_oof).all()
    if with_cv:
        assert np.isfinite(regression_oof).all()
        pd.DataFrame(cv_rows).to_csv(destination / 'cv_folds.csv', index=False)
        pd.DataFrame({'order_id': train.order_id, 'fold': train.oof_fold,
                      'actual_days': train.lead_time_days, 'predicted_days': regression_oof}).to_csv(
                          destination / 'regression_oof.csv', index=False)
    pd.DataFrame({'order_id': train.order_id, 'risk_p30': risk_oof}).to_csv(destination / 'training_risk_oof.csv', index=False)
    _, risk_bundle, _ = features.risk_fit(train, test)
    prep = features.prep_for('HGB_best', train)
    time = features.TimeEncoder().fit(train)
    scale = StandardScaler().fit(risk_oof.reshape(-1, 1))
    a = features.add_blocks(ext.transform(prep, train), train, 'time_risk', time, risk_oof, scale)
    learner = HistGradientBoostingRegressor(random_state=33, early_stopping=False, **params).fit(a, train.lead_time_days)
    bundle = features.InferenceBundle(prep, learner, 'time_risk', time, scale, risk_bundle)
    path = destination / 'selected_model.joblib'
    joblib.dump(bundle, path)
    np.testing.assert_allclose(joblib.load(path).predict(test.head(100)), bundle.predict(test.head(100)), atol=1e-10, rtol=0)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Nonexistent directory under this bundle runs/.')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--with-cv', action='store_true')
    parser.add_argument('--verify-model', type=Path, help='Verify an existing locally generated trusted model; no refitting.')
    args = parser.parse_args()
    verify_bundle(ROOT)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if (ROOT / 'runs').resolve() not in output.parents:
        parser.error('Keep generated data and fitted artifacts under the ignored bundle runs/ directory.')
    if output.exists():
        parser.error('Use a nonexistent output directory; preserve previous results.')
    if args.prepare_only and (args.verify_model or args.with_cv):
        parser.error('--prepare-only cannot be combined with fitting/CV/model validation options.')
    output.mkdir(parents=True)
    with threadpool_limits(limits=2):
        train, test = prepare(args.archive.resolve(), output)
        if args.prepare_only:
            return
        bundle = joblib.load(args.verify_model) if args.verify_model else fit(train, test, output, args.with_cv)
        actual = evaluate(bundle, test, output)
        if args.verify_model:
            reported = json.loads((ROOT / 'results/result_summary.json').read_text())
            np.testing.assert_allclose(actual['MAE_days'], reported['selected_test_MAE_days'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(actual['RMSE_days'], reported['selected_test_RMSE_days'], atol=1e-10, rtol=0)
            ext.save(output / 'model_portability_verification.json', {'full_test_metrics_match': True,
                'bundle_source_model_reload': True, 'outcome_columns_not_required': True, 'n': 31836,
                'refit': False, 'stacking': False})
    print('Completed local reproduction/verification:', output, flush=True)


if __name__ == '__main__':
    main()
