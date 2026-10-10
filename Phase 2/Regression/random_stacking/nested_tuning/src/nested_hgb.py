"""Build HGB inner OOF predictions inside each fixed outer training partition.

Each regression inner fit reconstructs its own three-fold cross-fitted risk
feature. Neither global regression OOF nor global risk OOF is a training input.
Only the existing 64,634-row training frame is loaded; no holdout data is read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
NESTED = Path(__file__).resolve().parents[1]
ROOT = NESTED.parent
REFERENCE = ROOT.parent / "delivery_regression_best"
sys.path.insert(0, str(REFERENCE / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import time_risk_features as features
import tree_extensions as ext


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    value.to_csv(temporary, index=False)
    temporary.replace(path)


def save_model(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def identity(train_path, protocol_path):
    """Pin all code and configuration that can influence fitting or splitting."""
    dependencies = sorted((REFERENCE / "src").glob("*.py"))
    dependencies.extend([REFERENCE / "config/selected_model.json",
                         REFERENCE / "config/random_split_protocol.json"])
    return {
        "source_sha256": sha(__file__),
        "protocol_sha256": sha(protocol_path),
        "train_frame_sha256": sha(train_path),
        "reference_dependencies": {str(p.relative_to(REFERENCE)): sha(p) for p in dependencies},
        "software": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]},
        "risk_params": features.RISK_PARAMS,
        "threadpool_limit": 2,
        "test_data_access": False,
    }


def make_partition(frame, folds, seed):
    assignment = np.zeros(len(frame), dtype=int)
    for fold, (_, score) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(frame), 1):
        assert (assignment[score] == 0).all()
        assignment[score] = fold
    assert (assignment > 0).all()
    return assignment


def check_boundary(fit, score, allowed_ids, outer_validation_ids):
    fit_ids, score_ids = set(fit.order_id), set(score.order_id)
    assert len(fit_ids) == len(fit) and len(score_ids) == len(score)
    assert fit_ids.isdisjoint(score_ids)
    assert fit_ids <= allowed_ids and score_ids <= allowed_ids
    assert fit_ids.isdisjoint(outer_validation_ids)
    assert score_ids.isdisjoint(outer_validation_ids)


def audited_risk_fit(fit, score, allowed_ids, outer_validation_ids, outer, inner, risk_fold):
    check_boundary(fit, score, allowed_ids, outer_validation_ids)
    probability, bundle, audit = features.risk_fit(fit, score)
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    audit.update({"outer_fold": outer, "inner_regression_fold": inner, "risk_fold": risk_fold,
                  "allowed_ids_sha256": ext.ids_sha(sorted(allowed_ids)),
                  "outer_validation_ids_sha256": ext.ids_sha(sorted(outer_validation_ids)),
                  "outer_validation_overlap": 0})
    return probability, bundle, audit


def fit_inner(fit, score, destination, outer, inner, outer_validation_ids, params, seed):
    """Fit one inner regression with risk cross-fitting strictly within fit."""
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    allowed_ids = set(fit.order_id) | set(score.order_id)
    check_boundary(fit, score, allowed_ids, outer_validation_ids)
    risk_fold = make_partition(fit, 3, seed)
    risk_oof = np.full(len(fit), np.nan)
    risk_audits = []
    for subfold in range(1, 4):
        mask = risk_fold == subfold
        a, b = fit.iloc[np.flatnonzero(~mask)], fit.iloc[np.flatnonzero(mask)]
        probability, risk_bundle, audit = audited_risk_fit(
            a, b, set(fit.order_id), outer_validation_ids, outer, inner, subfold)
        risk_oof[mask] = probability
        risk_audits.append(audit)
        save_model(destination / f"risk_model_{subfold}.joblib", risk_bundle)
        print(f"HGB outer {outer}/5 inner {inner}/3 risk {subfold}/3 complete", flush=True)
    assert np.isfinite(risk_oof).all()
    score_risk, full_risk_bundle, audit = audited_risk_fit(
        fit, score, allowed_ids, outer_validation_ids, outer, inner, "full")
    risk_audits.append(audit)
    prep = features.prep_for("HGB_best", fit)
    date = features.TimeEncoder().fit(fit)
    scale = StandardScaler().fit(risk_oof.reshape(-1, 1))
    x = features.add_blocks(ext.transform(prep, fit), fit, "time_risk", date, risk_oof, scale)
    regressor = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **params)
    regressor.fit(x, fit.lead_time_days.to_numpy())
    bundle = features.InferenceBundle(prep, regressor, "time_risk", date, scale, full_risk_bundle)
    prediction = bundle.predict(score)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    save_model(destination / "model.joblib", bundle)
    restored = joblib.load(destination / "model.joblib")
    np.testing.assert_allclose(restored.predict(score), prediction, atol=1e-10, rtol=0)
    reduced = score.head(80).drop(columns=["lead_time_days", "delivered_ts", "estimated_ts", "order_status"])
    np.testing.assert_allclose(restored.predict(reduced), prediction[:80], atol=1e-10, rtol=0)
    output = pd.DataFrame({"order_id": score.order_id.to_numpy(), "inner_fold": inner,
                           "actual_days": score.lead_time_days.to_numpy(), "predicted_days": prediction})
    save_csv(destination / "predictions.csv", output)
    save_csv(destination / "risk_fit_predictions.csv", pd.DataFrame({
        "order_id": fit.order_id.to_numpy(), "risk_fold": risk_fold, "risk_p30": risk_oof}))
    save_csv(destination / "risk_score_predictions.csv", pd.DataFrame({
        "order_id": score.order_id.to_numpy(), "risk_p30": score_risk}))
    save_json(destination / "risk_audit.json", risk_audits)
    record = {
        "outer_fold": outer, "inner_fold": inner,
        "fit_n": len(fit), "score_n": len(score),
        "fit_ids_sha256": ext.ids_sha(fit.order_id), "score_ids_sha256": ext.ids_sha(score.order_id),
        "outer_validation_ids_sha256": ext.ids_sha(sorted(outer_validation_ids)),
        "fit_score_overlap": 0, "outer_validation_overlap": 0,
        "MAE_days": float(np.abs(prediction - score.lead_time_days.to_numpy()).mean()),
        "seconds": time.monotonic() - started,
        "regression_numeric_medians": prep.numeric_.named_steps["impute"].statistics_.tolist(),
        "model_reload_prediction_match": True, "outcome_columns_not_required": True,
        "artifact_hashes": {p.name: sha(p) for p in sorted(destination.iterdir())
                            if p.is_file() and not p.name.endswith(".tmp") and p.name != "checkpoint.json"},
    }
    save_json(destination / "checkpoint.json", record)
    print(f"HGB outer {outer}/5 inner {inner}/3 saved: MAE {record['MAE_days']:.6f}, "
          f"{record['seconds']:.1f}s", flush=True)
    return output


def verify_inner(fit, score, destination, outer, inner, outer_validation_ids, params, seed, replay=False):
    """Independently reconstruct every risk/regression partition for resume/audit."""
    record = json.loads((destination / "checkpoint.json").read_text())
    assert record["outer_fold"] == outer and record["inner_fold"] == inner
    assert record["fit_n"] == len(fit) and record["score_n"] == len(score)
    assert record["fit_ids_sha256"] == ext.ids_sha(fit.order_id)
    assert record["score_ids_sha256"] == ext.ids_sha(score.order_id)
    assert record["outer_validation_ids_sha256"] == ext.ids_sha(sorted(outer_validation_ids))
    for name, expected in record["artifact_hashes"].items():
        assert sha(destination / name) == expected, (destination, name)
    prediction = pd.read_csv(destination / "predictions.csv")
    assert prediction.order_id.tolist() == score.order_id.tolist()
    assert prediction.inner_fold.eq(inner).all()
    np.testing.assert_allclose(prediction.actual_days, score.lead_time_days, atol=1e-10, rtol=0)
    assert np.isfinite(prediction.predicted_days).all() and prediction.predicted_days.ge(0).all()
    np.testing.assert_allclose(np.abs(prediction.predicted_days - prediction.actual_days).mean(),
                               record["MAE_days"], atol=1e-10, rtol=0)
    fit_risk = pd.read_csv(destination / "risk_fit_predictions.csv")
    score_risk = pd.read_csv(destination / "risk_score_predictions.csv")
    assert fit_risk.order_id.tolist() == fit.order_id.tolist()
    assert score_risk.order_id.tolist() == score.order_id.tolist()
    folds = make_partition(fit, 3, seed)
    np.testing.assert_array_equal(folds, fit_risk.risk_fold)
    audits = json.loads((destination / "risk_audit.json").read_text())
    assert len(audits) == 4
    for subfold, audit in enumerate(audits, 1):
        if subfold <= 3:
            mask = folds == subfold
            a, b = fit.iloc[np.flatnonzero(~mask)], fit.iloc[np.flatnonzero(mask)]
            expected_risk_fold = subfold
            allowed_ids = set(fit.order_id)
        else:
            a, b = fit, score
            expected_risk_fold = "full"
            allowed_ids = set(fit.order_id) | set(score.order_id)
        check_boundary(a, b, allowed_ids, outer_validation_ids)
        assert audit["risk_fold"] == expected_risk_fold
        assert audit["outer_fold"] == outer and audit["inner_regression_fold"] == inner
        assert audit["fit_n"] == len(a) and audit["score_n"] == len(b)
        assert audit["fit_ids_sha256"] == ext.ids_sha(a.order_id)
        assert audit["score_ids_sha256"] == ext.ids_sha(b.order_id)
        assert audit["allowed_ids_sha256"] == ext.ids_sha(sorted(allowed_ids))
        assert audit["outer_validation_ids_sha256"] == ext.ids_sha(sorted(outer_validation_ids))
        assert audit["overlap"] == 0 and audit["outer_validation_overlap"] == 0
        assert audit["positive_fit_n"] == int(a.lead_time_days.gt(30).sum())
        numeric, _ = ext.fields(list(ext.GROUPS))
        np.testing.assert_allclose(a[numeric].median().to_numpy(), audit["fit_numeric_medians"], atol=1e-10, rtol=0)
        if replay and subfold <= 3:
            risk_model = joblib.load(destination / f"risk_model_{subfold}.joblib")
            np.testing.assert_allclose(features.risk_predict(risk_model, b), fit_risk.loc[mask, "risk_p30"],
                                       atol=1e-10, rtol=0)
    if replay:
        bundle = joblib.load(destination / "model.joblib")
        assert bundle.pack == "time_risk"
        for name, value in params.items():
            assert bundle.regressor.get_params()[name] == value
        assert bundle.regressor.random_state == seed and bundle.regressor.early_stopping is False
        np.testing.assert_allclose(bundle.predict(score), prediction.predicted_days, atol=1e-10, rtol=0)
        np.testing.assert_allclose(features.risk_predict(bundle.risk_bundle, score), score_risk.risk_p30,
                                   atol=1e-10, rtol=0)
        np.testing.assert_allclose(bundle.risk_scaler.mean_, [fit_risk.risk_p30.mean()], atol=1e-10, rtol=0)
        numeric, _ = ext.fields(list(ext.GROUPS))
        np.testing.assert_allclose(fit[numeric].median().to_numpy(),
                                   bundle.prep.numeric_.named_steps["impute"].statistics_, atol=1e-10, rtol=0)
        elapsed, _ = features.TimeEncoder.fields(fit)
        np.testing.assert_allclose(bundle.time_encoder.scale.mean_, elapsed.mean(axis=0), atol=1e-10, rtol=0)
    return prediction


def run_outer(train, outer, destination, protocol, params, verify_only=False):
    outer_fit = train.loc[train.oof_fold.ne(outer)].copy()
    outer_validation_ids = set(train.loc[train.oof_fold.eq(outer), "order_id"])
    destination.mkdir(parents=True, exist_ok=True)
    inner_folds = make_partition(outer_fit, protocol["inner_folds"], protocol["seed"])
    partition = pd.DataFrame({"order_id": outer_fit.order_id.to_numpy(), "inner_fold": inner_folds})
    partition_path = destination / "partition.csv"
    if partition_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(partition_path), partition)
    else:
        assert not verify_only
        save_csv(partition_path, partition)
    outputs = []
    for inner in range(1, protocol["inner_folds"] + 1):
        mask = inner_folds == inner
        fit, score = outer_fit.iloc[np.flatnonzero(~mask)], outer_fit.iloc[np.flatnonzero(mask)]
        folder = destination / f"inner_{inner}"
        if (folder / "checkpoint.json").exists():
            result = verify_inner(fit, score, folder, outer, inner, outer_validation_ids,
                                  params, protocol["seed"], replay=verify_only)
            print(f"HGB outer {outer}/5 inner {inner}/3 verified checkpoint", flush=True)
        else:
            assert not verify_only, f"Missing checkpoint: {folder}"
            result = fit_inner(fit, score, folder, outer, inner, outer_validation_ids, params, protocol["seed"])
            verify_inner(fit, score, folder, outer, inner, outer_validation_ids, params, protocol["seed"])
        outputs.append(result)
    combined = pd.concat(outputs).set_index("order_id").loc[outer_fit.order_id].reset_index()
    assert combined.order_id.tolist() == outer_fit.order_id.tolist() and combined.order_id.is_unique
    np.testing.assert_array_equal(combined.inner_fold, inner_folds)
    np.testing.assert_allclose(combined.actual_days, outer_fit.lead_time_days, atol=1e-10, rtol=0)
    path = destination / "oof.csv"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), combined, atol=1e-10, rtol=0)
    else:
        assert not verify_only
        save_csv(path, combined)
    completion = {
        "outer_fold": outer, "n": len(combined), "inner_folds": protocol["inner_folds"],
        "fit_ids_sha256": ext.ids_sha(outer_fit.order_id),
        "outer_validation_ids_sha256": ext.ids_sha(sorted(outer_validation_ids)),
        "partition_sha256": sha(partition_path), "oof_sha256": sha(path),
        "checkpoints": {f"inner_{i}/checkpoint.json": sha(destination / f"inner_{i}/checkpoint.json")
                        for i in range(1, protocol["inner_folds"] + 1)},
        "risk_partition_count": 4 * protocol["inner_folds"],
        "outer_validation_never_used_in_fit": True,
    }
    complete_path = destination / "complete.json"
    if complete_path.exists():
        assert json.loads(complete_path.read_text()) == completion
    else:
        assert not verify_only
        save_json(complete_path, completion)
    print(f"HGB outer {outer}/5 complete: {len(combined)} aligned inner OOF rows", flush=True)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nested_tuning_v1/hgb")
    parser.add_argument("--train", type=Path, default=ROOT / "runs/hgb_reference_v1/train_frame.joblib")
    parser.add_argument("--protocol", type=Path, default=NESTED / "config/protocol.json")
    parser.add_argument("--outer", type=int, choices=range(1, 6), nargs="+")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / "runs").resolve() not in output.parents:
        parser.error("Private outputs must remain under random_stacking/runs/.")
    if args.train.resolve() != (ROOT / "runs/hgb_reference_v1/train_frame.joblib").resolve():
        parser.error("Use only the verified reference training frame.")
    protocol = json.loads(args.protocol.read_text())
    assert protocol["outer_folds"] == 5 and protocol["inner_folds"] == 3 and protocol["seed"] == 33
    selected = json.loads((REFERENCE / "config/selected_model.json").read_text())
    assert selected["candidate"] == "leaves63__time_risk" and selected["pack"] == "time_risk"
    current_identity = identity(args.train, args.protocol)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "run_identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == current_identity, "Source/input/protocol changed; use a new run."
    else:
        assert not args.verify_only
        save_json(identity_path, current_identity)
    train = joblib.load(args.train)
    assert len(train) == protocol["train_n"] == 64634
    assert train.order_id.is_unique and train.order_id.tolist() == sorted(train.order_id)
    assert set(train.oof_fold) == set(range(1, 6))
    assert train.split.eq("train").all()
    started = time.monotonic()
    completions = []
    with threadpool_limits(limits=2):
        for outer in (args.outer or range(1, 6)):
            completions.append(run_outer(train, outer, output / f"outer_{outer}", protocol,
                                         selected["params"], args.verify_only))
    if not args.outer:
        report = {"outer_folds": 5, "inner_regression_models": 15, "risk_partition_checks": 60,
                  "test_data_access": False, "identity_sha256": sha(identity_path),
                  "outer_completion_hashes": {str(i): sha(output / f"outer_{i}/complete.json") for i in range(1, 6)},
                  "outer_outputs": completions}
        path = output / ("verification.json" if args.verify_only else "complete.json")
        if path.exists():
            assert json.loads(path.read_text()) == report
        else:
            save_json(path, report)
    print(f"HGB nested {'verification' if args.verify_only else 'fitting'} completed in "
          f"{time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
