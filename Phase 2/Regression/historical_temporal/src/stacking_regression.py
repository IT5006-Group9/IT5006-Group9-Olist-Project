"""Chronological stacking of the four reviewed delivery regressors.

Development meta fitting uses March-visible OOF labels only. Historical scoring
also reports a separately named mature cohort; those future labels never enter
an earlier fit. The selected design is frozen before March inspection. Final
refitting regenerates pre-July OOF and requires an explicit --final-test run.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import delivery_regression as base
import baseline_variant_review as previous

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "versions/stacking_v1"
INPUT = DEST / "data/eligible_orders.csv"
BASE_MODELS = {
    "linear": "linear",
    "ridge": "ridge_logx_a1000",
    "tree": "tree_d8",
    "forest": "rf_recent180",
}
META_COLUMNS = [f"{name}_oof_days" for name in BASE_MODELS]
SUBSETS = {
    "four": META_COLUMNS,
    "ridge_forest": ["ridge_oof_days", "forest_oof_days"],
}
ALPHAS = [0.1, 1.0, 10.0, 100.0]
FINAL_FOLDS = base.FOLDS + [
    ("2018-03-01", "2018-04-01"),
    ("2018-04-01", "2018-05-01"),
    ("2018-05-01", "2018-06-01"),
    ("2018-06-01", "2018-07-01"),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def directories():
    for folder in ["data", "models", "outputs/tables", "outputs/predictions",
                   "outputs/oof", "outputs/figures"]:
        (DEST / folder).mkdir(parents=True, exist_ok=True)


def specifications():
    available = previous.configurations()
    return {name: available[variant] for name, variant in BASE_MODELS.items()}


def candidates():
    return {f"stack_{subset}_a{alpha:g}": {"subset": subset,
            "columns": columns, "alpha": alpha}
            for subset, columns in SUBSETS.items() for alpha in ALPHAS}


def build_meta(spec):
    return Pipeline([("scale", StandardScaler()),
                     ("ridge", Ridge(alpha=spec["alpha"]))])


def predict_meta(model, rows, columns):
    return np.maximum(0.0, model.predict(rows[columns]))


def temporal_fit_rows(rows, cutoff):
    """Both the purchase and the receipt label must precede the forecast origin."""
    cutoff = pd.Timestamp(cutoff)
    return rows.loc[(rows.purchase_ts < cutoff) & (rows.delivered_ts < cutoff)].copy()


def meta_fit_rows(oof, cutoff, fold=None):
    fit = temporal_fit_rows(oof, cutoff)
    if fold is not None:
        fit = fit.loc[fit.fold < fold].copy()
    if fit.empty:
        raise ValueError("No earlier OOF labels are available at this forecast origin.")
    return fit


def aligned_oof(metadata, predictions):
    """Join each independently generated prediction column by ID, never position."""
    if not metadata.order_id.is_unique:
        raise ValueError("Duplicate order IDs in OOF metadata.")
    result = metadata.copy()
    expected = set(metadata.order_id)
    for name, values in predictions.items():
        if not values.order_id.is_unique or set(values.order_id) != expected:
            raise ValueError(f"OOF IDs do not align for {name}.")
        result = result.merge(values[["order_id", "prediction_days"]].rename(
            columns={"prediction_days": f"{name}_oof_days"}),
            on="order_id", how="left", validate="one_to_one", sort=False)
    if not np.isfinite(result[META_COLUMNS].to_numpy()).all():
        raise ValueError("Missing or non-finite OOF predictions.")
    return result


class DeliveryStack:
    """Serializable purchase-feature-to-days inference bundle."""

    def __init__(self, base_models, meta_model, columns):
        self.base_models = base_models
        self.meta_model = meta_model
        self.columns = list(columns)
        self.feature_names = list(base.FEATURES)

    def predict_base(self, features):
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing purchase-time features: {sorted(missing)}")
        return pd.DataFrame({f"{name}_oof_days": base.predict_days(model,
                            features[self.feature_names])
                            for name, model in self.base_models.items()},
                            index=features.index)

    def predict(self, features):
        return predict_meta(self.meta_model, self.predict_base(features), self.columns)


def prepare_input(csv_dir=None, archive=None):
    """Build an isolated audited-v2 snapshot without overwriting legacy outputs."""
    directories()
    if not INPUT.exists():
        reference = ROOT / "versions/preparation_v2/data/eligible_orders.csv"
        if reference.exists() and csv_dir is None and archive is None:
            quality = json.loads((reference.parents[1] / "outputs/data_quality.json").read_text())
            if quality.get("preparation_version") != "purchase_inputs_v2_screened_median":
                raise ValueError("Existing preparation is not audited v2.")
            # Keep all provenance with the standalone stacking experiment.
            INPUT.write_bytes(reference.read_bytes())
            source = reference.parents[1] / "outputs/tables/source_inventory.csv"
            (DEST / "outputs/tables/source_inventory.csv").write_bytes(source.read_bytes())
            base.save_json(DEST / "outputs/data_quality.json", quality)
        else:
            base.prepare_data(csv_dir=csv_dir, archive=archive,
                              output_root=DEST, geography_policy="screened_median")
    inventory = pd.read_csv(DEST / "outputs/tables/source_inventory.csv")
    reference_inventory = pd.read_csv(ROOT / "versions/preparation_v2/outputs/tables/source_inventory.csv")
    pd.testing.assert_frame_equal(inventory[["table", "rows", "sha256"]],
                                  reference_inventory[["table", "rows", "sha256"]])
    quality = json.loads((DEST / "outputs/data_quality.json").read_text())
    if quality["preparation_version"] != "purchase_inputs_v2_screened_median":
        raise ValueError("Stacking requires audited-v2 screened median geography.")
    return INPUT


def load_development():
    columns = ["order_id", "purchase_ts", "delivered_ts", "lead_time_days", "split", *base.FEATURES]
    # Keep only development purchases. Locked test labels never reach fit/selection.
    portions = []
    for chunk in pd.read_csv(INPUT, usecols=columns, parse_dates=["purchase_ts", "delivered_ts"], chunksize=20000):
        portions.append(chunk.loc[chunk.purchase_ts < base.VALIDATION_END])
    rows = pd.concat(portions, ignore_index=True)
    train = rows.loc[rows.split.eq("train")].copy()
    validation = rows.loc[rows.split.eq("validation")].copy()
    mature = rows.loc[(rows.purchase_ts < base.VALIDATION_START) &
                      (rows.delivered_ts < base.TEST_START)].copy()
    assert len(train) == 53644 and len(validation) == 7003
    assert rows.order_id.is_unique and mature.order_id.is_unique
    assert train.delivered_ts.max() < base.VALIDATION_START
    assert validation.delivered_ts.max() < base.TEST_START
    return train, validation, mature


def write_protocol():
    environment = {name: importlib.metadata.version(name) for name in [
        "numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "nbformat", "nbclient"]}
    protocol = {
        "input_path": str(INPUT.relative_to(ROOT)), "input_sha256": sha256(INPUT),
        "preparation_version": "purchase_inputs_v2_screened_median", "seed": base.SEED,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [
            Path(__file__), ROOT / "src/delivery_regression.py", ROOT / "src/baseline_variant_review.py"]},
        "environment": environment, "python": platform.python_version(),
        "features": base.FEATURES, "base_variants": BASE_MODELS,
        "base_specs": specifications(), "development_folds": base.FOLDS,
        "final_folds": FINAL_FOLDS, "meta_candidates": candidates(),
        "meta_selection": "lowest unweighted mean mature-history MAE on folds 2 and 3; ties: fewer inputs, lower alpha, name",
        "meta_training": "OOF only; earlier folds; purchase and receipt before each forecast origin",
        "development_meta_cutoff": str(base.VALIDATION_START),
        "mature_history_label_cutoff": str(base.TEST_START),
        "base_selection_nested": False,
        "limits": ["Base configurations were selected in prior non-nested development.",
                   "Conditional OOF and mature scoring cohorts are kept separate.",
                   "March was already inspected in previous development.",
                   "Completed-delivery conditioning and finite label maturity remain."],
        "stacking_fitted": False, "test_scored": False,
    }
    base.save_json(DEST / "outputs/experiment_protocol.json", protocol)
    return protocol


def generate_oof(fit_pool, score_pool, folds, prefix):
    pieces, manifests, summaries = [], [], []
    specs = specifications()
    with threadpool_limits(limits=4):
        for fold, (start, end) in enumerate(folds, 1):
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            fit = temporal_fit_rows(fit_pool, start)
            score = score_pool.loc[(score_pool.purchase_ts >= start) &
                                   (score_pool.purchase_ts < end)].copy()
            if fit.empty or score.empty:
                raise ValueError(f"Empty fit/scoring cohort in {prefix} fold {fold}.")
            assert not set(fit.order_id) & set(score.order_id)
            metadata = score[["order_id", "purchase_ts", "delivered_ts", "lead_time_days"]].copy()
            metadata["fold"] = fold
            metadata["forecast_cutoff"] = start
            predictions = {}
            for name, spec in specs.items():
                began = time.perf_counter()
                model = previous.fit_pipeline(previous.build_pipeline(spec), spec, fit, start)
                pred = base.predict_days(model, score[base.FEATURES])
                predictions[name] = pd.DataFrame({"order_id": score.order_id.to_numpy(), "prediction_days": pred})
                joblib.dump(model, DEST / f"models/{prefix}_fold{fold}_{name}.joblib")
                summaries.append({"phase": prefix, "fold": fold, "start": start, "end": end,
                    "model": name, "fit_n": len(fit), **base.metric_row(score.lead_time_days, pred),
                    "seconds": time.perf_counter() - began})
                print(f"{prefix} fold {fold} {name}: fit={len(fit)}, score={len(score)}, MAE={summaries[-1]['MAE_days']:.3f}", flush=True)
            pieces.append(aligned_oof(metadata, predictions))
            for role, frame in [("fit", fit), ("score", score)]:
                entry = frame[["order_id"]].copy()
                entry["fold"], entry["role"] = fold, role
                manifests.append(entry)
    result = pd.concat(pieces, ignore_index=True)
    assert result.order_id.is_unique
    pd.concat(manifests, ignore_index=True).to_csv(DEST / f"data/{prefix}_base_fold_manifest.csv", index=False)
    pd.DataFrame(summaries).to_csv(DEST / f"outputs/tables/{prefix}_base_cv.csv", index=False)
    return result


def select_meta(oof, mature_backtest):
    rows, prediction_parts, manifest_parts = [], [], []
    options = candidates()
    for fold, (start, end) in enumerate(base.FOLDS, 1):
        if fold == 1:
            continue
        fit = meta_fit_rows(oof, start, fold)
        score = mature_backtest.loc[mature_backtest.fold.eq(fold)].copy()
        assert not set(fit.order_id) & set(score.order_id)
        predictions = score[["order_id", "fold", "lead_time_days"]].copy()
        for name, spec in options.items():
            model = build_meta(spec).fit(fit[spec["columns"]], fit.lead_time_days)
            predictions[name] = predict_meta(model, score, spec["columns"])
            joblib.dump(model, DEST / f"models/meta_fold{fold}_{name}.joblib")
        predictions["mean_four"] = score[META_COLUMNS].mean(axis=1)
        predictions["mean_ridge_forest"] = score[SUBSETS["ridge_forest"]].mean(axis=1)
        for family in BASE_MODELS:
            predictions[family] = score[f"{family}_oof_days"]
        conditional = score.order_id.isin(oof.order_id)
        for cohort, mask in [("mature_history", np.ones(len(score), dtype=bool)),
                             ("March_visible_history", conditional.to_numpy())]:
            for name in [*options, "mean_four", "mean_ridge_forest", *BASE_MODELS]:
                rows.append({"model": name, "fold": fold, "cohort": cohort,
                             "meta_fit_n": len(fit), **base.metric_row(score.lead_time_days[mask], predictions[name][mask])})
        prediction_parts.append(predictions)
        record = fit[["order_id"]].copy()
        record["fold"] = fold
        manifest_parts.append(record)
    cv = pd.DataFrame(rows)
    cv.to_csv(DEST / "outputs/tables/meta_cv_results.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_csv(DEST / "outputs/predictions/meta_cv_predictions.csv", index=False)
    pd.concat(manifest_parts, ignore_index=True).to_csv(DEST / "data/meta_fold_manifest.csv", index=False)
    summary = cv.groupby(["cohort", "model"]).agg(
        CV_MAE_mean=("MAE_days", "mean"), CV_MAE_std=("MAE_days", "std"),
        CV_RMSE_mean=("RMSE_days", "mean"), scoring_n=("n", "sum")).reset_index()
    summary.to_csv(DEST / "outputs/tables/meta_cv_summary.csv", index=False)
    ranked = summary.loc[summary.cohort.eq("mature_history") & summary.model.isin(options)].copy()
    ranked["input_n"] = ranked.model.map(lambda name: len(options[name]["columns"]))
    ranked["alpha"] = ranked.model.map(lambda name: options[name]["alpha"])
    ranked = ranked.sort_values(["CV_MAE_mean", "input_n", "alpha", "model"])
    chosen = ranked.iloc[0].model
    subset_choices = {subset: ranked.loc[ranked.model.map(lambda name: options[name]["subset"] == subset)].iloc[0].model for subset in SUBSETS}
    design = {"selected": chosen, "selected_spec": options[chosen], "subset_choices": subset_choices,
              "selected_CV_MAE_days": float(ranked.iloc[0].CV_MAE_mean),
              "selection_uses_March_targets": False, "selection_uses_test_targets": False,
              "design_frozen_before_March_inspection": True,
              "protocol_sha256": sha256(DEST / "outputs/experiment_protocol.json")}
    base.save_json(DEST / "outputs/frozen_design.json", design)
    print("Frozen stacking design:", chosen, flush=True)
    return design, cv, summary


def fit_bases(train, cutoff, phase):
    models = {}
    with threadpool_limits(limits=4):
        for name, spec in specifications().items():
            models[name] = previous.fit_pipeline(previous.build_pipeline(spec), spec, train, pd.Timestamp(cutoff))
            joblib.dump(models[name], DEST / f"models/{phase}_{name}.joblib")
    return models


def evaluation_predictions(models, meta_models, scoring, train):
    result = scoring[["order_id", "purchase_ts", "lead_time_days", "route_group"]].copy()
    meta_inputs = pd.DataFrame(index=scoring.index)
    for name, model in models.items():
        result[name] = base.predict_days(model, scoring[base.FEATURES])
        meta_inputs[f"{name}_oof_days"] = result[name]
    for name, (model, columns) in meta_models.items():
        result[name] = predict_meta(model, meta_inputs, columns)
    result["mean_four"] = meta_inputs[META_COLUMNS].mean(axis=1)
    result["mean_ridge_forest"] = meta_inputs[SUBSETS["ridge_forest"]].mean(axis=1)
    result["promise"] = scoring.promised_lead_time_days.to_numpy()
    result["train_mean"] = train.lead_time_days.mean()
    result["train_median"] = train.lead_time_days.median()
    return result


def comparison_tables(predictions, phase, selected):
    names = [name for name in predictions.columns if name not in ["order_id", "purchase_ts", "lead_time_days", "route_group"]]
    rows, slices = [], []
    for name in names:
        errors = np.abs(predictions[name] - predictions.lead_time_days)
        role = "selected_stacking" if name == selected else "base" if name in BASE_MODELS else "reference" if name in ["promise", "train_mean", "train_median"] else "fixed_comparator"
        rows.append({"model": name, "role": role, **base.metric_row(predictions.lead_time_days, predictions[name]),
                     "within_3_days_pct": float((errors <= 3).mean() * 100)})
        groups = pd.cut(predictions.lead_time_days, [0, 7, 14, 30, 60, np.inf], include_lowest=True)
        for group, part in predictions.groupby(groups, observed=True):
            slices.append({"model": name, "duration_group": str(group), **base.metric_row(part.lead_time_days, part[name])})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(DEST / f"outputs/tables/{phase}_comparison.csv", index=False)
    pd.DataFrame(slices).to_csv(DEST / f"outputs/tables/{phase}_duration_errors.csv", index=False)
    residuals = predictions[names].subtract(predictions.lead_time_days, axis=0)
    residuals.corr().to_csv(DEST / f"outputs/tables/{phase}_residual_correlations.csv")
    predictions.to_csv(DEST / f"outputs/predictions/{phase}_predictions.csv", index=False)
    return comparison


def coefficient_table(meta_models, phase):
    rows = []
    for name, (model, columns) in meta_models.items():
        scale, reg = model.named_steps["scale"], model.named_steps["ridge"]
        weights = reg.coef_ / scale.scale_
        intercept = reg.intercept_ - np.dot(weights, scale.mean_)
        for column, weight, scaled in zip(columns, weights, reg.coef_):
            rows.append({"model": name, "input": column, "weight_in_days_space": weight,
                         "standardized_coefficient": scaled, "intercept_days": intercept,
                         "weight_sum": float(weights.sum())})
    table = pd.DataFrame(rows)
    table.to_csv(DEST / f"outputs/tables/{phase}_meta_coefficients.csv", index=False)
    return table


def paired_comparison(predictions, candidate, comparator, phase):
    difference = (predictions[candidate]-predictions.lead_time_days).abs() - (predictions[comparator]-predictions.lead_time_days).abs()
    daily = pd.DataFrame({"date": predictions.purchase_ts.dt.normalize(), "difference": difference}).groupby("date").difference.agg(["sum", "count"])
    rng = np.random.default_rng(base.SEED)
    draws = rng.integers(0, len(daily), size=(2000, len(daily)))
    deltas = daily["sum"].to_numpy()[draws].sum(axis=1) / daily["count"].to_numpy()[draws].sum(axis=1)
    result = {"candidate": candidate, "comparator": comparator,
        "MAE_difference_days": float(difference.mean()),
        "day_cluster_bootstrap_95_interval": np.quantile(deltas, [.025, .975]).tolist(),
        "seed": base.SEED, "resamples": 2000,
        "interpretation": "Negative favors stacking; descriptive fixed-month comparison, not a model-selection rule."}
    base.save_json(DEST / f"outputs/{phase}_paired_comparison.json", result)
    return result


def plot_results(predictions, comparison, phase, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [*BASE_MODELS, "mean_four", "mean_ridge_forest", selected]
    metrics = comparison.set_index("model").loc[labels]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = ["#2563eb" if label == selected else "#94a3b8" for label in labels]
    axes[0].barh(labels, metrics.MAE_days, color=colors)
    axes[0].set_xlabel("MAE (days)")
    axes[0].set_title(f"{phase.capitalize()} — same-order comparison")
    for i, value in enumerate(metrics.MAE_days):
        axes[0].text(value+.04, i, f"{value:.3f}", va="center", fontsize=9)
    axes[0].set_xlim(0, max(metrics.MAE_days)*1.2)
    bins = pd.cut(predictions.lead_time_days, [0, 7, 14, 30, 60, np.inf], include_lowest=True)
    for name in ["ridge", "forest", selected]:
        errors = (predictions[name]-predictions.lead_time_days).abs().groupby(bins, observed=True).mean()
        axes[1].plot(range(len(errors)), errors, marker="o", label=name)
    axes[1].set_xticks(range(5), ["0–7", "7–14", "14–30", "30–60", ">60"])
    axes[1].set_xlabel("Actual delivery duration (days)")
    axes[1].set_ylabel("MAE (days)")
    axes[1].set_title("Slow-delivery error")
    axes[1].legend(fontsize=8)
    corr = predictions[list(BASE_MODELS)].subtract(predictions.lead_time_days, axis=0).corr()
    axes[2].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[2].set_xticks(range(4), list(BASE_MODELS), rotation=35)
    axes[2].set_yticks(range(4), list(BASE_MODELS))
    axes[2].set_title("Base-model residual correlation")
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", color="white" if abs(corr.iloc[i,j])>.6 else "black")
    fig.savefig(DEST / f"outputs/figures/{phase}_review.png", dpi=160)
    plt.close(fig)


def run_development(csv_dir=None, archive=None):
    if (DEST / "outputs/final_evaluation.json").exists():
        raise RuntimeError("Final test is already scored for this experiment. Start a new version for further development.")
    prepare_input(csv_dir, archive)
    protocol = write_protocol()
    train, validation, mature = load_development()
    backtest = generate_oof(train, mature, base.FOLDS, "development")
    oof = backtest.loc[backtest.order_id.isin(train.order_id)].copy()
    assert len(oof) == 39445 and len(backtest) == 43114
    oof.to_csv(DEST / "outputs/oof/development_oof.csv", index=False)
    backtest.to_csv(DEST / "outputs/oof/mature_history_backtest.csv", index=False)
    design, cv, summary = select_meta(oof, backtest)
    selected = design["selected"]
    models = fit_bases(train, base.VALIDATION_START, "development")
    meta_models = {}
    for name in design["subset_choices"].values():
        spec = candidates()[name]
        model = build_meta(spec).fit(oof[spec["columns"]], oof.lead_time_days)
        meta_models[name] = (model, spec["columns"])
        joblib.dump(model, DEST / f"models/development_{name}.joblib")
    bundle = DeliveryStack(models, *meta_models[selected])
    joblib.dump(bundle, DEST / "models/stacking_development.joblib")
    predictions = evaluation_predictions(models, meta_models, validation, train)
    comparison = comparison_tables(predictions, "validation", selected)
    coefficients = coefficient_table(meta_models, "development")
    paired = paired_comparison(predictions, selected, "ridge", "validation")
    plot_results(predictions, comparison, "validation", selected)
    protocol["stacking_fitted"] = True
    # Preserve the pre-fit protocol: status is a separate record, not a changed hash.
    result = {"selected": selected, "train_n": len(train), "validation_n": len(validation),
        "oof_n": len(oof), "warmup_n": len(train)-len(oof), "mature_backtest_n": len(backtest),
        "restored_backtest_n": len(backtest)-len(oof), "stacking_fitted": True, "test_scored": False,
        "frozen_design_sha256": sha256(DEST / "outputs/frozen_design.json"),
        "validation_selected": comparison.loc[comparison.model.eq(selected)].iloc[0].to_dict(),
        "validation_ridge": comparison.loc[comparison.model.eq("ridge")].iloc[0].to_dict(),
        "paired_vs_ridge": paired}
    base.save_json(DEST / "outputs/development_result.json", result)
    print(comparison[["model", "MAE_days", "RMSE_days", "bias_days"]].round(3).to_string(index=False), flush=True)
    return result


def run_final():
    """One final evaluation of the already frozen design; never select by test."""
    marker = DEST / "outputs/final_evaluation.json"
    if marker.exists():
        raise RuntimeError("This experiment has already scored the locked test. Read the saved result.")
    design_path = DEST / "outputs/frozen_design.json"
    development = json.loads((DEST / "outputs/development_result.json").read_text())
    design = json.loads(design_path.read_text())
    protocol = json.loads((DEST / "outputs/experiment_protocol.json").read_text())
    if sha256(design_path) != development["frozen_design_sha256"]:
        raise ValueError("Frozen design changed after development.")
    if sha256(DEST / "outputs/experiment_protocol.json") != design["protocol_sha256"]:
        raise ValueError("Protocol changed after design freeze.")
    if sha256(INPUT) != protocol["input_sha256"]:
        raise ValueError("Prepared data changed after design freeze.")
    for filename, digest in protocol["source_sha256"].items():
        if sha256(ROOT / filename) != digest:
            raise ValueError(f"Source changed after design freeze: {filename}")
    # Load historical labels first. Test inputs are loaded without test outcomes.
    columns = ["order_id", "purchase_ts", "delivered_ts", "lead_time_days", *base.FEATURES]
    portions = []
    for chunk in pd.read_csv(INPUT, usecols=columns, parse_dates=["purchase_ts", "delivered_ts"], chunksize=20000):
        portions.append(temporal_fit_rows(chunk, base.TEST_START))
    train = pd.concat(portions, ignore_index=True)
    inputs = pd.read_csv(INPUT, usecols=["order_id", "purchase_ts", *base.FEATURES], parse_dates=["purchase_ts"])
    inputs = inputs.loc[inputs.purchase_ts >= base.TEST_START].copy()
    assert len(inputs) == 12507 and inputs.order_id.is_unique
    assert not set(train.order_id) & set(inputs.order_id)
    oof = generate_oof(train, train, FINAL_FOLDS, "final")
    oof.to_csv(DEST / "outputs/oof/pretest_oof.csv", index=False)
    spec, selected = design["selected_spec"], design["selected"]
    meta = build_meta(spec).fit(oof[spec["columns"]], oof.lead_time_days)
    models = fit_bases(train, base.TEST_START, "final")
    meta_models = {selected: (meta, spec["columns"])}
    bundle = DeliveryStack(models, meta, spec["columns"])
    joblib.dump(bundle, DEST / "models/stacking_final.joblib")
    joblib.dump(meta, DEST / f"models/final_{selected}.joblib")
    coefficient_table(meta_models, "final")
    # All predictions and fit fingerprints are persisted before attaching labels.
    dummy = inputs.copy()
    dummy["lead_time_days"] = np.nan
    predictions = evaluation_predictions(models, meta_models, dummy, train).drop(columns="lead_time_days")
    predictions.to_csv(DEST / "outputs/predictions/locked_test_predictions.csv", index=False)
    fit_record = {"selected": selected, "design_sha256": sha256(design_path),
        "pretest_fit_n": len(train), "pretest_oof_n": len(oof), "test_n": len(inputs),
        "train_label_cutoff": str(base.TEST_START), "test_labels_used_for_fitting": False,
        "prediction_sha256": sha256(DEST / "outputs/predictions/locked_test_predictions.csv"),
        "model_sha256": sha256(DEST / "models/stacking_final.joblib")}
    base.save_json(DEST / "outputs/final_fit_record.json", fit_record)
    labels = pd.read_csv(INPUT, usecols=["order_id", "lead_time_days"])
    scored = predictions.merge(labels, on="order_id", how="left", validate="one_to_one")
    assert scored.lead_time_days.notna().all()
    comparison = comparison_tables(scored, "test", selected)
    paired = paired_comparison(scored, selected, "ridge", "test")
    plot_results(scored, comparison, "test", selected)
    result = {**fit_record, "test_scored": True, "test_used_for_selection": False,
        "selected_test_metrics": comparison.loc[comparison.model.eq(selected)].iloc[0].to_dict(),
        "ridge_test_metrics": comparison.loc[comparison.model.eq("ridge")].iloc[0].to_dict(),
        "paired_vs_ridge": paired,
        "limit": "Historical eligible completed deliveries; finite maturity and tail sample limitations remain."}
    base.save_json(marker, result)
    print(comparison[["model", "MAE_days", "RMSE_days", "bias_days"]].round(3).to_string(index=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, help="Original course CSV directory; read only.")
    parser.add_argument("--archive", type=Path, help="Original course ZIP; read only.")
    parser.add_argument("--final-test", action="store_true", help="Refit the frozen design and evaluate the locked test once.")
    args = parser.parse_args()
    if args.final_test:
        run_final()
    else:
        run_development(args.csv_dir, args.archive)


if __name__ == "__main__":
    # Use a stable import name so the saved bundle can be reloaded elsewhere.
    import stacking_regression as runner
    runner.main()
