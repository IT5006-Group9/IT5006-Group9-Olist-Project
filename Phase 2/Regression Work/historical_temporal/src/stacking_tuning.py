"""Exploratory monthly stacking refinement after the original holdout was seen.

All selection is restricted to historical October 2017--April 2018 windows.
May is a prespecified sensitivity window, observed only after both selections.
June contributes eligible, already-received OOF labels to the July fit but is
not a scoring window. No July-or-later predictions or scores are produced.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data
from threadpoolctl import threadpool_limits

import delivery_regression as base
import baseline_variant_review as previous
import stacking_regression as legacy

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "versions/stacking_tuning_v2"
INPUT = legacy.INPUT
BASE_MODELS = dict(legacy.BASE_MODELS)
META_COLUMNS = list(legacy.META_COLUMNS)
LABEL_CUTOFF = pd.Timestamp("2018-07-01")
MONTHS = pd.date_range("2017-07-01", "2018-06-01", freq="MS").strftime("%Y-%m").tolist()
PRIMARY_MONTHS = pd.date_range("2017-10-01", "2018-04-01", freq="MS").strftime("%Y-%m").tolist()
SENSITIVITY_MONTHS = ["2018-05"]
TIE_TOLERANCE = .01
STAGE1_PREFERENCE = ["convex_two", "convex_four", "ridge_original", "median_four"]
STAGE2_PREFERENCE = ["full_history", "recent180", "gap30"]
REFERENCES = [*BASE_MODELS, "mean_four", "mean_ridge_forest"]


class ConvexMAERegressor(RegressorMixin, BaseEstimator):
    """Minimize absolute error with nonnegative, unit-sum raw-scale weights.

The intercept is fixed at zero. The sparse LP introduces one absolute-error
variable per observation; there is no dense observations-squared allocation.
"""

    def fit(self, X, y):
        values, target = validate_data(self, X, y, dtype=float, y_numeric=True)
        n, m = values.shape
        z = sparse.csc_matrix(values)
        identity = sparse.eye(n, format="csc")
        inequalities = sparse.vstack([
            sparse.hstack([z, -identity]),
            sparse.hstack([-z, -identity]),
        ], format="csc")
        equality = sparse.csc_matrix(np.r_[np.ones(m), np.zeros(n)][None, :])
        solution = linprog(
            np.r_[np.zeros(m), np.full(n, 1. / n)],
            A_ub=inequalities, b_ub=np.r_[target, -target],
            A_eq=equality, b_eq=np.array([1.]),
            bounds=[(0., 1.)] * m + [(0., None)] * n,
            method="highs",
        )
        if not solution.success:
            raise RuntimeError(f"Convex MAE optimization failed: {solution.message}")
        weights = solution.x[:m]
        if weights.min() < -1e-7 or abs(weights.sum()-1.) > 1e-7:
            raise RuntimeError("Convex MAE solver returned infeasible weights.")
        self.coef_ = np.maximum(weights, 0.)
        self.coef_ /= self.coef_.sum()
        self.intercept_ = 0.
        self.solver_success_ = True
        self.solver_message_ = solution.message
        self.objective_ = float(np.abs(values @ self.coef_ - target).mean())
        self.n_iter_ = int(solution.nit)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        values = validate_data(self, X, reset=False, dtype=float)
        return values @ self.coef_


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def directories():
    for folder in ["data", "models", "outputs/tables", "outputs/oof",
                   "outputs/predictions", "outputs/figures"]:
        (DEST / folder).mkdir(parents=True, exist_ok=True)


def stage1_specs():
    common = {"window_days": None, "gap_days": 0}
    return {
        "ridge_original": {**common, "kind": "ridge", "columns": META_COLUMNS},
        "convex_two": {**common, "kind": "convex", "columns": ["ridge_oof_days", "forest_oof_days"]},
        "convex_four": {**common, "kind": "convex", "columns": META_COLUMNS},
        "median_four": {**common, "kind": "median", "columns": META_COLUMNS},
    }


def stage2_specs(chosen):
    return {"full_history": dict(chosen),
            "recent180": {**chosen, "window_days": 180},
            "gap30": {**chosen, "gap_days": 30}}


def build_meta(spec):
    if spec["kind"] == "convex":
        return ConvexMAERegressor()
    if spec["kind"] == "ridge":
        estimator = Ridge(alpha=.1)
    elif spec["kind"] == "median":
        estimator = QuantileRegressor(quantile=.5, alpha=0., solver="highs")
    else:
        raise ValueError(f"Unknown meta model: {spec['kind']}")
    return Pipeline([("scale", StandardScaler()), ("regressor", estimator)])


def meta_fit_rows(oof, cutoff, window_days=None, gap_days=0):
    cutoff = pd.Timestamp(cutoff)
    if window_days is not None and window_days <= 0:
        raise ValueError("window_days must be positive.")
    if gap_days < 0:
        raise ValueError("gap_days must be nonnegative.")
    mask = ((oof.purchase_ts < cutoff) & (oof.delivered_ts < cutoff) &
            (oof.forecast_cutoff < cutoff))
    if window_days is not None:
        mask &= oof.purchase_ts >= cutoff-pd.Timedelta(days=window_days)
    if gap_days:
        mask &= oof.purchase_ts < cutoff-pd.Timedelta(days=gap_days)
    fit = oof.loc[mask].copy()
    if fit.empty:
        raise ValueError("No historical OOF labels meet the meta training rule.")
    return fit


def meta_predict(model, rows, columns):
    return np.maximum(0., model.predict(rows[columns]))


def load_history():
    """Keep only pre-July purchases and July-visible targets for modelling.

For pending pre-July orders only the count contributes to the coverage audit.
No label or future receipt value for those rows is returned to the experiment.
"""
    columns = ["order_id", "purchase_ts", "delivered_ts", "lead_time_days", *base.FEATURES]
    portions, coverage_parts = [], []
    for chunk in pd.read_csv(INPUT, usecols=columns, parse_dates=["purchase_ts", "delivered_ts"], chunksize=20000):
        history = chunk.loc[chunk.purchase_ts < LABEL_CUTOFF].copy()
        known = history.delivered_ts < LABEL_CUTOFF
        coverage_parts.append(pd.DataFrame({
            "month": history.purchase_ts.dt.strftime("%Y-%m"),
            "known": known.to_numpy(),
        }))
        portions.append(history.loc[known])
    frame = pd.concat(portions, ignore_index=True)
    assert len(frame) == 82258 and frame.order_id.is_unique
    assert frame.purchase_ts.max() < LABEL_CUTOFF and frame.delivered_ts.max() < LABEL_CUTOFF
    coverage = pd.concat(coverage_parts, ignore_index=True).groupby("month").known.agg(
        eligible_n="size", known_by_july_n="sum").reset_index()
    coverage = coverage.loc[coverage.month.isin(MONTHS)].copy()
    coverage["pending_n"] = coverage.eligible_n-coverage.known_by_july_n
    coverage["pending_pct"] = 100*coverage.pending_n/coverage.eligible_n
    coverage["role"] = coverage.month.map(lambda month: "primary" if month in PRIMARY_MONTHS
        else "sensitivity" if month in SENSITIVITY_MONTHS else "meta_only" if month == "2018-06" else "warmup")
    return frame, coverage


def protect_v1():
    paths = [ROOT / "src/stacking_regression.py", ROOT / "notebooks/stacking_review.ipynb"]
    paths.extend(path for path in legacy.DEST.rglob("*") if path.is_file() and path.suffix != ".log")
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def write_protocol():
    protocol = {
        "input_path": str(INPUT.relative_to(ROOT)), "input_sha256": sha256(INPUT),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [
            Path(__file__), ROOT / "src/delivery_regression.py",
            ROOT / "src/baseline_variant_review.py", ROOT / "src/stacking_regression.py"]},
        "python": platform.python_version(),
        "environment": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]},
        "seed": base.SEED, "features": base.FEATURES, "base_variants": BASE_MODELS,
        "monthly_oof_windows": MONTHS, "primary_months": PRIMARY_MONTHS,
        "sensitivity_months": SENSITIVITY_MONTHS, "label_cutoff": str(LABEL_CUTOFF),
        "stage1_specs": stage1_specs(),
        "stage2_history_rules": {"full_history": {"window_days": None, "gap_days": 0},
            "recent180": {"window_days": 180, "gap_days": 0}, "gap30": {"window_days": None, "gap_days": 30}},
        "tie_tolerance_days": TIE_TOLERANCE,
        "stage1_preference": STAGE1_PREFERENCE, "stage2_preference": STAGE2_PREFERENCE,
        "selection_rule": "Within 0.01 days of minimum primary equal-month MAE, use the predeclared simplicity preference; May/June never select.",
        "fit_rule": "Both purchase and receipt strictly before monthly origin; OOF origin earlier; window/gap affect meta fit only.",
        "test_scored": False, "test_predictions_generated": False,
        "interpretation": "Exploratory follow-up motivated by an already-observed original test; no fresh independent test claim.",
        "limitations": ["Prior base configurations are non-nested selections.",
            "The two-stage search reuses primary months.",
            "Completed-delivery and finite-maturity conditioning remain.",
            "Monthly refitting is the assumed operating schedule; a model held fixed for two months has a different schedule."],
        "v1_protected_sha256": protect_v1(),
    }
    base.save_json(DEST / "outputs/experiment_protocol.json", protocol)
    return protocol


def generate_monthly_oof(history):
    frames, records = [], []
    with threadpool_limits(limits=4):
        for fold, month in enumerate(MONTHS, 1):
            start = pd.Timestamp(month + "-01")
            end = start + pd.offsets.MonthBegin(1)
            fit = history.loc[(history.purchase_ts < start) & (history.delivered_ts < start)]
            score = history.loc[(history.purchase_ts >= start) & (history.purchase_ts < end)]
            assert len(fit) and len(score) and not set(fit.order_id) & set(score.order_id)
            metadata = score[["order_id", "purchase_ts", "delivered_ts", "lead_time_days"]].copy()
            metadata["fold"], metadata["forecast_cutoff"], metadata["month"] = fold, start, month
            predictions = {}
            for name, spec in legacy.specifications().items():
                model = previous.fit_pipeline(previous.build_pipeline(spec), spec, fit, start)
                values = base.predict_days(model, score[base.FEATURES])
                predictions[name] = pd.DataFrame({"order_id": score.order_id.to_numpy(), "prediction_days": values})
                joblib.dump(model, DEST / f"models/base_{month}_{name}.joblib")
            frames.append(legacy.aligned_oof(metadata, predictions))
            for role, rows in [("fit", fit), ("score", score)]:
                manifest = rows[["order_id"]].copy()
                manifest["month"], manifest["role"] = month, role
                records.append(manifest)
            print(f"Monthly OOF {month}: base fit={len(fit)}, known-label predictions={len(score)}", flush=True)
    oof = pd.concat(frames, ignore_index=True)
    assert len(oof) == 68059 and oof.order_id.is_unique
    oof.to_csv(DEST / "outputs/oof/monthly_oof.csv", index=False)
    pd.concat(records, ignore_index=True).to_csv(DEST / "data/base_fold_manifest.csv", index=False)
    return oof


def coefficient_rows(model, spec, stage, month, candidate):
    if spec["kind"] == "convex":
        weights, intercept = model.coef_, model.intercept_
    else:
        scale, reg = model.named_steps["scale"], model.named_steps["regressor"]
        weights = reg.coef_ / scale.scale_
        intercept = reg.intercept_ - weights @ scale.mean_
    return [{"stage": stage, "month": month, "candidate": candidate, "input": column,
        "weight_in_days_space": float(weight), "intercept_days": float(intercept),
        "weight_sum": float(weights.sum()),
        "solver_success": bool(getattr(model, "solver_success_", True))}
        for column, weight in zip(spec["columns"], weights)]


def evaluate_months(oof, specs, months, stage, role):
    metrics, predictions, manifests, coefficients = [], [], [], []
    for month in months:
        cutoff = pd.Timestamp(month + "-01")
        score = oof.loc[oof.month.eq(month)].copy()
        assert len(score)
        result = score[["order_id", "month", "lead_time_days"]].copy()
        result["role"] = role
        for name, spec in specs.items():
            fit = meta_fit_rows(oof, cutoff, spec["window_days"], spec["gap_days"])
            assert not set(fit.order_id) & set(score.order_id)
            model = build_meta(spec).fit(fit[spec["columns"]], fit.lead_time_days)
            result[name] = meta_predict(model, score, spec["columns"])
            joblib.dump(model, DEST / f"models/{stage}_{month}_{name}.joblib")
            record = fit[["order_id"]].copy()
            record["month"], record["candidate"] = month, name
            manifests.append(record)
            coefficients.extend(coefficient_rows(model, spec, stage, month, name))
            metric = {"candidate": name, "month": month, "role": role, "meta_fit_n": len(fit),
                **base.metric_row(score.lead_time_days, result[name]),
                "within_3_days_pct": float((np.abs(result[name]-score.lead_time_days) <= 3).mean()*100)}
            metrics.append(metric)
            print(f"{stage} {month} {name}: MAE={metric['MAE_days']:.4f}, bias={metric['bias_days']:.4f}", flush=True)
        for name in BASE_MODELS:
            result[name] = score[f"{name}_oof_days"]
        result["mean_four"] = score[META_COLUMNS].mean(axis=1)
        result["mean_ridge_forest"] = score[["ridge_oof_days", "forest_oof_days"]].mean(axis=1)
        for name in REFERENCES:
            metrics.append({"candidate": name, "month": month, "role": role, "meta_fit_n": 0,
                **base.metric_row(score.lead_time_days, result[name]),
                "within_3_days_pct": float((np.abs(result[name]-score.lead_time_days) <= 3).mean()*100)})
        predictions.append(result)
    return {"metrics": pd.DataFrame(metrics), "predictions": pd.concat(predictions, ignore_index=True),
            "manifests": pd.concat(manifests, ignore_index=True), "coefficients": pd.DataFrame(coefficients)}


def summarize(metrics):
    frame = metrics.assign(abs_bias=metrics.bias_days.abs())
    return frame.groupby(["candidate", "role"], sort=True).agg(
        month_n=("month", "nunique"), MAE_month_mean=("MAE_days", "mean"),
        MAE_worst_month=("MAE_days", "max"), bias_abs_month_mean=("abs_bias", "mean"),
        RMSE_month_mean=("RMSE_days", "mean"), scoring_n=("n", "sum")).reset_index()


def choose(primary_metrics, specs, preference, stage):
    summary = summarize(primary_metrics)
    ranked = summary.loc[summary.role.eq("primary") & summary.candidate.isin(specs)]
    minimum = ranked.MAE_month_mean.min()
    eligible = set(ranked.loc[ranked.MAE_month_mean <= minimum+TIE_TOLERANCE, "candidate"])
    selected = next(name for name in preference if name in eligible)
    result = {"selected": selected, "spec": specs[selected],
        "primary_summary": ranked.sort_values("MAE_month_mean").to_dict(orient="records"),
        "minimum_primary_MAE_days": float(minimum), "tie_tolerance_days": TIE_TOLERANCE,
        "eligible_within_tolerance": sorted(eligible), "preference": preference,
        "selection_rule": "primary equal-month MAE within 0.01 days of minimum, then predeclared simplicity preference",
        "May_used_for_selection": False, "June_used_for_selection": False, "test_used_for_selection": False}
    base.save_json(DEST / f"outputs/{stage}_selection.json", result)
    print(f"Selected {stage}: {selected}", flush=True)
    return result


def save_stage(primary, sensitivity, stage):
    combined = {key: pd.concat([primary[key], sensitivity[key]], ignore_index=True) for key in primary}
    combined["metrics"].to_csv(DEST / f"outputs/tables/{stage}_monthly_metrics.csv", index=False)
    summarize(combined["metrics"]).to_csv(DEST / f"outputs/tables/{stage}_summary.csv", index=False)
    combined["predictions"].to_csv(DEST / f"outputs/predictions/{stage}_predictions.csv", index=False)
    combined["manifests"].to_csv(DEST / f"data/{stage}_meta_fit_ids.csv", index=False)
    return combined


def fit_july_bundle(history, oof, design):
    spec = design["spec"]
    meta_fit = meta_fit_rows(oof, LABEL_CUTOFF, spec["window_days"], spec["gap_days"])
    meta = build_meta(spec).fit(meta_fit[spec["columns"]], meta_fit.lead_time_days)
    models = {}
    with threadpool_limits(limits=4):
        for name, base_spec in legacy.specifications().items():
            models[name] = previous.fit_pipeline(previous.build_pipeline(base_spec), base_spec, history, LABEL_CUTOFF)
            joblib.dump(models[name], DEST / f"models/final_{name}.joblib")
    bundle = legacy.DeliveryStack(models, meta, spec["columns"])
    joblib.dump(meta, DEST / "models/meta_final.joblib")
    joblib.dump(bundle, DEST / "models/stacking_tuned.joblib")
    history[["order_id"]].to_csv(DEST / "data/final_base_fit_ids.csv", index=False)
    meta_fit[["order_id"]].to_csv(DEST / "data/final_meta_fit_ids.csv", index=False)
    return bundle, meta_fit, coefficient_rows(meta, spec, "final", "2018-07", design["selected"])


def plot_results(stage1, stage2, design):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first, second = stage1["metrics"], stage2["metrics"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    names = ["ridge_original", "convex_two", "convex_four", "median_four"]
    months = PRIMARY_MONTHS + SENSITIVITY_MONTHS
    for name in names:
        rows = first.loc[first.candidate.eq(name)].set_index("month").loc[months]
        axes[0].plot(months, rows.MAE_days, marker="o", label=name)
        axes[1].plot(months, rows.bias_days, marker="o", label=name)
    axes[0].set_ylabel("MAE (days)")
    axes[1].set_ylabel("Mean prediction minus actual (days)")
    axes[1].axhline(0, color="#64748b", linewidth=1)
    for ax in axes:
        ax.axvspan(6.5, 7.5, color="#e2e8f0", alpha=.65)
        ax.grid(alpha=.15)
    axes[0].legend(ncol=2)
    axes[0].set_title("Monthly rolling evaluation | Oct–Apr selection; May sensitivity only")
    fig.savefig(DEST / "outputs/figures/monthly_comparison.png", dpi=160)
    plt.close(fig)
    summaries = [summarize(first), summarize(second)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    lists = [[*names, "ridge", "forest", "mean_ridge_forest"], [*STAGE2_PREFERENCE, "ridge", "forest", "mean_ridge_forest"]]
    for i, (summary, names) in enumerate(zip(summaries, lists)):
        rows = summary.loc[summary.role.eq("primary")].set_index("candidate").loc[names]
        chosen = design["stage1_selected"] if i == 0 else design["stage2_selected"]
        colors = ["#2563eb" if name == chosen else "#94a3b8" for name in names]
        axes[i].barh(names, rows.MAE_month_mean, color=colors)
        for j, value in enumerate(rows.MAE_month_mean):
            axes[i].text(value+.02, j, f"{value:.3f}", va="center", fontsize=9)
        axes[i].set_xlim(0, rows.MAE_month_mean.max()*1.15)
        axes[i].set_xlabel("Equal-month primary MAE (days)")
        axes[i].set_title("Stage 1: combiner" if i == 0 else "Stage 2: meta history only")
    fig.savefig(DEST / "outputs/figures/tuning_summary.png", dpi=160)
    plt.close(fig)


def run_experiment():
    directories()
    if not INPUT.exists():
        raise FileNotFoundError("Prepare audited v2 course data first; pass --input to its eligible_orders.csv, or reproduce stacking_v1.")
    protocol = write_protocol()
    history, coverage = load_history()
    coverage.to_csv(DEST / "outputs/tables/coverage_by_month.csv", index=False)
    oof = generate_monthly_oof(history)
    primary1 = evaluate_months(oof, stage1_specs(), PRIMARY_MONTHS, "stage1", "primary")
    first = choose(primary1["metrics"], stage1_specs(), STAGE1_PREFERENCE, "stage1")
    specs2 = stage2_specs(first["spec"])
    base.save_json(DEST / "outputs/stage2_protocol.json", {"stage1_selected": first["selected"], "specs": specs2,
        "history_rules_predeclared_in_main_protocol": True})
    primary2 = evaluate_months(oof, specs2, PRIMARY_MONTHS, "stage2", "primary")
    second = choose(primary2["metrics"], specs2, STAGE2_PREFERENCE, "stage2")
    design = {"selected": first["selected"]+"__"+second["selected"],
        "stage1_selected": first["selected"], "stage2_selected": second["selected"], "spec": second["spec"],
        "protocol_sha256": sha256(DEST / "outputs/experiment_protocol.json"),
        "selection_frozen_before_May_sensitivity": True, "test_used_for_selection": False}
    base.save_json(DEST / "outputs/selected_design.json", design)
    # Hold May apart until both choices are frozen. June is not scored here.
    sensitivity1 = evaluate_months(oof, stage1_specs(), SENSITIVITY_MONTHS, "stage1", "sensitivity")
    sensitivity2 = evaluate_months(oof, specs2, SENSITIVITY_MONTHS, "stage2", "sensitivity")
    combined1 = save_stage(primary1, sensitivity1, "stage1")
    combined2 = save_stage(primary2, sensitivity2, "stage2")
    bundle, meta_fit, final_coefficients = fit_july_bundle(history, oof, design)
    pd.concat([combined1["coefficients"], combined2["coefficients"], pd.DataFrame(final_coefficients)],
              ignore_index=True).to_csv(DEST / "outputs/tables/meta_coefficients.csv", index=False)
    plot_results(combined1, combined2, design)
    for filename, digest in protocol["v1_protected_sha256"].items():
        if sha256(ROOT / filename) != digest:
            raise RuntimeError(f"Original stacking artifact changed: {filename}")
    selected_summary = summarize(combined2["metrics"])
    selected_summary = selected_summary.loc[selected_summary.candidate.eq(second["selected"])]
    result = {"selected": design["selected"], "base_fit_n": len(history), "meta_fit_n": len(meta_fit), "oof_n": len(oof),
        "primary_n": int(oof.month.isin(PRIMARY_MONTHS).sum()),
        "sensitivity_n": int(oof.month.isin(SENSITIVITY_MONTHS).sum()),
        "test_scored": False, "test_predictions_generated": False,
        "status": "exploratory_historical_tuning_complete", "v1_unchanged": True,
        "selected_metrics": selected_summary.to_dict(orient="records"),
        "design_sha256": sha256(DEST / "outputs/selected_design.json"),
        "model_sha256": sha256(DEST / "models/stacking_tuned.joblib"),
        "limitations": protocol["limitations"]}
    base.save_json(DEST / "outputs/run_result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    global INPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Existing audited-v2 eligible_orders.csv.")
    args = parser.parse_args()
    if args.input is not None:
        INPUT = args.input.resolve()
    run_experiment()


if __name__ == "__main__":
    import stacking_tuning as runner
    runner.main()
