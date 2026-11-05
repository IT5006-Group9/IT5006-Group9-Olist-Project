"""Read the published Phase 3 evidence for a dashboard, without fitting models.

Metrics retain their original names and precision. Required main-model evidence
raises a readable ``EvidenceError`` if missing or inconsistent. Optional charts
and supplementary stacking tables return empty DataFrames when unavailable, and
their paths are listed in ``metadata['missing_optional']``. There is deliberately
no Streamlit dependency, caching, model loading, or writes to pipeline outputs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


class EvidenceError(ValueError):
    """Published evidence is missing, malformed, or inconsistent."""


MODEL_LABELS = {
    "HGB_risk": "HGB with past-only risk feature (current main model)",
    "HGB_risk_current_main": "Current main HGB with risk (saved assessment)",
    "HGB_leaves15_l21": "HGB, 15 leaves (without risk)",
    "Ridge_a100": "Ridge, alpha 100",
    "OLS": "Ordinary linear regression",
    "RF_d12_l80": "Random forest, depth 12 / leaf 80",
    "Tree_d6_l30": "Decision tree, depth 6 / leaf 30",
    "Median": "Training-median baseline",
    "Mean": "Training-mean baseline",
    "Promise": "Delivery-promise baseline",
    "Convex_stack": "Original Phase 3 convex stack",
    "random_forest": "Random forest (selected classifier)",
    "catboost": "CatBoost",
    "logistic_tuned": "Logistic regression, tuned",
    "logistic_base": "Logistic regression, fixed baseline",
    "decision_tree": "Decision tree",
    "baseline_majority": "No-skill baseline (positive rate)",
    "baseline_not_delivered_by_T": "Rule: not delivered by T",
    "ridge_fixed": "Phase 2 fixed Ridge, log inputs",
    "forest_fixed": "Phase 2 fixed random forest",
    "ridge_tuned": "Inner-selected Ridge",
    "forest_tuned": "Inner-selected random forest",
    "hgb": "Phase 2 HGB, 63 leaves (chronological rerun)",
    "hgb63_phase2": "Phase 2 HGB, 63 leaves (supplementary assessment)",
    "rf_log": "Random forest, log target / F1",
    "rf_time_risk": "Random forest, log target / time and risk",
    "extra_trees": "ExtraTrees, inner-selected target transform",
    "mean_two_fixed": "Fixed Ridge + RF: equal mean",
    "stack_two_fixed": "Fixed Ridge + RF: MAE stack",
    "mean_three_fixed": "Fixed Ridge + RF + HGB: equal mean",
    "stack_three_fixed": "Fixed Ridge + RF + HGB: MAE stack",
    "mean_two_tuned": "Tuned Ridge + RF: equal mean",
    "stack_two_tuned": "Tuned Ridge + RF: MAE stack",
    "mean_three_tuned": "Tuned Ridge + RF + HGB: equal mean",
    "stack_three_tuned": "Tuned Ridge + RF + HGB: MAE stack",
    "mean_hgb_rf_log": "HGB + log-target RF: equal mean",
    "stack_hgb_rf_log": "HGB + log-target RF: MAE stack",
    "mean_hgb_extra_trees": "HGB + ExtraTrees: equal mean",
    "stack_hgb_extra_trees": "HGB + ExtraTrees: MAE stack",
    "mean_three_rf_risk": "Ridge + RF time/risk + HGB: equal mean",
    "stack_three_rf_risk": "Ridge + RF time/risk + HGB: MAE stack",
}


# Source paths are authoritative; the loader never replaces a missing table with
# a similarly named Phase 2 or other Phase 3 experiment's metrics.
_JSON_FILES = {
    "regression_summary": "outputs/regression/test_summary.json",
    "regression_selection": "outputs/regression/frozen_selection.json",
    "regression_protocol": "outputs/regression/protocol.json",
    "classification_summary": "outputs/classification/test_summary.json",
    "classification_selection": "outputs/classification/selection.json",
}
_TABLE_FILES = {
    "regression_comparison": ("outputs/regression/tables/train_cv_test_comparison.csv", True,
        {"candidate", "test_mae", "train_mae", "cv_mae", "cv_sd"}),
    "regression_months": ("outputs/regression/tables/test_by_month.csv", False,
        {"month", "n", "mae", "rmse"}),
    "regression_importance": ("outputs/regression/tables/permutation_importance_groups.csv", False,
        {"group", "mean", "std"}),
    "regression_error_groups": ("outputs/regression/tables/test_error_groups.csv", False,
        {"group", "n", "mae", "bias", "within3"}),
    "regression_selection_sensitivity": ("outputs/regression/tables/selection_maturity_sensitivity.csv", False,
        {"candidate", "five_month_mae", "jan_apr_mae", "five_month_rank", "jan_apr_rank"}),
    "classification_cv": ("outputs/classification/tables/cv_family_summary.csv", True,
        {"family", "cv_pr_auc_mean", "cv_pr_auc_sd", "cv_recall_at_capacity_mean", "train_pr_auc_mean"}),
    "classification_operating": ("outputs/classification/tables/test_operating_points.csv", True,
        {"rule", "threshold", "flag_rate", "precision", "recall"}),
    "classification_months": ("outputs/classification/tables/test_by_month.csv", False,
        {"t_month", "n", "pr_auc", "recall_at_capacity"}),
    "classification_importance": ("outputs/classification/tables/permutation_importance_groups.csv", False,
        {"group", "features", "drop_mean", "drop_sd"}),
    "classification_criteria": ("outputs/classification/tables/success_criteria.csv", True,
        {"criterion", "target", "achieved", "met"}),
    "stacking_comparison": ("outputs/regression_stacking_holdout/tables/test_comparison.csv", False,
        {"candidate", "cv_mae", "n", "mae", "rmse", "assessment"}),
    "stacking_cv": ("outputs/regression_stacking_chronological/tables/all_14_combinations.csv", False,
        {"candidate", "cv_mae", "cv_sd", "train_mae", "folds", "role"}),
    "stacking_weights": ("outputs/regression_stacking_holdout/tables/frozen_weights.csv", False,
        {"candidate", "member", "weight"}),
}
_CV_FILES = (
    "outputs/regression/tables/cv_all_candidates_folds.csv",
    "outputs/regression/tables/diagnostic_folds.csv",
    "outputs/regression/tables/cv_reference_folds.csv",
    "outputs/regression/tables/stacking_folds.csv",
)

# Enhancements and the strict rerun supplement v1; they never replace its keys.
# Each CSV is read as published, including its metric names and precision.
_UPDATE_JSON = {
    "strict": {
        "summary": "test_summary_strict.json", "selection": "selection_strict.json",
        "rules": "strict_rules.json", "test_lock": "test_lock_strict.json",
    },
    "experiments": {
        "selection": "selection_v2.json", "stacking": "stacking.json", "test_lock": "test_lock_v2.json",
    },
}
_UPDATE_TABLES = {
    "strict": {
        "cv": ("cv_family_summary.csv", {"family", "cv_pr_auc_mean", "cv_pr_auc_sd", "cv_recall_at_capacity_mean", "train_pr_auc_mean"}),
        "operating": ("test_operating_points.csv", {"rule", "threshold", "flag_rate", "precision", "recall"}),
        "months": ("test_by_month.csv", {"t_month", "n", "pr_auc", "recall_at_capacity"}),
        "criteria": ("success_criteria.csv", {"criterion", "target", "achieved", "met"}),
        "paired": ("paired_bootstrap.csv", {"comparison", "d_pr_auc", "d_pr_auc_ci", "d_recall", "d_recall_ci"}),
        "test_results": ("test_results.csv", {"model", "split", "pr_auc", "recall_at_capacity", "n"}),
        "fold_table": ("strict_fold_table.csv", {"fold", "cutoff", "train_rows_v1", "train_rows_strict", "removed_by_R1", "scored_rows_v1", "scored_rows_strict", "removed_by_R2"}),
        "diagnostics": ("diagnostic_rules.csv", {"setting", "scored_rows", "mean_val_pr_auc", "months_better"}),
        "importance": ("importance_groups.csv", {"group", "features", "drop_mean", "drop_sd"}),
    },
    "experiments": {
        "ledger": ("ledger.csv", {"step", "experiment", "mean_val_pr_auc", "mean_val_recall_at_capacity", "decision", "test_pr_auc", "test_recall_at_capacity"}),
        "test_results": ("test_results_v2.csv", {"model", "is_v2", "test_pr_auc", "test_recall_at_capacity", "test_brier"}),
        "daily": ("daily_batch_recall.csv", {"model", "window", "pooled_daily_recall", "window_recall", "days", "orders"}),
        "calibration": ("calibration_decisions.csv", {"model", "method", "months_better", "mean_raw_brier", "mean_calibrated_brier", "mean_d_pr_auc", "decision"}),
        "calibration_test": ("calibration_test_addendum.csv", {"model", "test_raw_brier", "test_sigmoid_brier", "test_pr_auc_raw", "test_pr_auc_sigmoid"}),
    },
}
_UPDATE_DIRECTORIES = {"strict": "strict_cutoff", "experiments": "experiments"}
_UPDATE_CORE = {
    "strict": ("summary", "selection", "rules", "test_lock", "cv", "operating", "criteria"),
    "experiments": ("selection", "ledger", "test_results"),
}


def _fail(message, path=None):
    suffix = f" Source: {path}" if path is not None else ""
    raise EvidenceError(message + suffix)


def _read_json(path):
    if not path.is_file():
        _fail("Required dashboard evidence is missing.", path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _fail("Cannot read dashboard JSON evidence: " + str(error), path)
    if not isinstance(value, dict):
        _fail("Expected a JSON object for dashboard evidence.", path)
    return value


def _read_table(path, required, columns, missing):
    if not path.is_file():
        if required:
            _fail("Required dashboard table is missing.", path)
        missing.append(str(path))
        return pd.DataFrame(columns=sorted(columns))
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        _fail("Cannot read dashboard CSV evidence: " + str(error), path)
    absent = sorted(columns - set(frame.columns))
    if absent:
        _fail("Dashboard evidence columns are missing: " + ", ".join(absent) + ".", path)
    if frame.empty:
        _fail("Published dashboard table exists but has no rows.", path)
    return frame


def _number(value, label):
    try:
        value = float(value)
    except (TypeError, ValueError):
        _fail("Expected a numeric value for " + label + ".")
    if not math.isfinite(value):
        _fail("Nonfinite evidence value for " + label + ".")
    return value


def _same(a, b, label, tolerance=1e-9):
    if not math.isclose(_number(a, label), _number(b, label), rel_tol=1e-10, abs_tol=tolerance):
        _fail("Inconsistent published evidence for " + label + ".")


def _one_row(frame, column, value, label):
    rows = frame[frame[column].eq(value)]
    if len(rows) != 1:
        _fail(f"Expected one {label} row for {value}; found {len(rows)}.")
    return rows.iloc[0]


def _unique(frame, columns, label):
    if frame.duplicated(columns).any():
        _fail("Duplicate evidence identities in " + label + ": " + ", ".join(columns))


def _unit_metrics(frame, columns, label):
    """Check saved probabilities/metrics while allowing documented blank cells."""
    for column in columns:
        if column not in frame:
            continue
        try:
            values = pd.to_numeric(frame[column], errors="raise").dropna()
        except (TypeError, ValueError):
            _fail(f"Nonnumeric {label} metric: {column}")
        if not values.between(0, 1).all():
            _fail(f"{label} metric outside [0,1]: {column}")


def _ci(value, label):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            _fail("Invalid saved confidence interval for " + label)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail("Expected two confidence interval endpoints for " + label)
    endpoints = [_number(item, label) for item in value]
    if endpoints[0] > endpoints[1]:
        _fail("Reversed confidence interval for " + label)
    return endpoints


def _validate_strict_update(strict, original, reference_ids):
    if not strict["available"]:
        return
    summary, selection = strict["summary"], strict["selection"]
    try:
        family = selection["selected_family"]
        selected_hash = selection["selection_sha256"]
        test, train = summary["test"], summary["train"]
        n = int(test["n"])
        if family != summary["family"] or selected_hash != summary["selection_sha256"]:
            _fail("Strict classification summary and selection identify different models.")
        if selection["rules"] != strict["rules"]["rules"]:
            _fail("Strict classification rule manifests disagree.")
        _same(selection["gap_days"], strict["rules"]["gap_days"], "strict classification purge gap")
        _same(n, original["classification_summary"]["test"]["n"], "strict classification test cohort")
        _same(summary["train_rows"], train["n"], "strict classification training count")
        for stage, metrics in (("train", train), ("test", test)):
            for metric in ("pr_auc", "roc_auc", "brier", "positive_rate", "recall_at_capacity", "precision_at_capacity"):
                value = _number(metrics[metric], f"strict {stage} {metric}")
                if not 0 <= value <= 1:
                    _fail(f"Strict {stage} {metric} is outside [0,1].")
        cv = strict["cv"]
        _unique(cv, ["family"], "strict classification family summary")
        chosen = _one_row(cv, "family", family, "strict selected family")
        for metric in ("cv_pr_auc_mean", "cv_pr_auc_sd", "cv_recall_at_capacity_mean", "train_pr_auc_mean"):
            _same(chosen[metric], selection["cv"][metric], "strict selection " + metric)
        _unit_metrics(cv, ("cv_pr_auc_mean", "cv_pr_auc_sd", "cv_recall_at_capacity_mean", "train_pr_auc_mean"), "strict CV")
        locked = strict["test_lock"]
        if reference_ids is not None and locked["test_ids_sha256"] != reference_ids:
            _fail("Strict classification and v1 test order hashes differ.")
        evaluations = [item for item in locked["evaluations"] if item.get("selection_sha256") == selected_hash]
        if len(evaluations) != 1 or evaluations[0].get("family") != family:
            _fail("Strict classification test lock does not identify exactly one matching selection.")
        for metric in ("pr_auc", "recall_at_capacity"):
            _same(evaluations[0][metric], test[metric], "strict test lock " + metric)
            label = metric + "_95ci"
            saved_ci = _ci(summary["bootstrap"][label], "strict " + label)
            locked_ci = _ci(evaluations[0][label], "strict locked " + label)
            for actual, expected in zip(saved_ci, locked_ci):
                _same(actual, expected, "strict test lock " + label)
                if not 0 <= actual <= 1:
                    _fail("Strict bootstrap endpoint is outside [0,1].")
        _same(summary["bootstrap"]["n_boot"], evaluations[0]["n_boot"], "strict bootstrap resamples")
        ops = strict["operating"]
        _unique(ops, ["rule"], "strict classification operating rules")
        _unit_metrics(ops, ("threshold", "flag_rate", "precision", "recall"), "strict operating")
        ranked = _one_row(ops, "rule", "top 20% of the test window (capacity rule)", "strict rank rule")
        fixed = _one_row(ops, "rule", "frozen capacity threshold from validation", "strict frozen rule")
        # The strict CSV retains ceil(20% * n) / n, rather than v1's rounded .2.
        _same(ranked.flag_rate, math.ceil(.2 * n) / n, "strict rank capacity rounding")
        _same(ranked.recall, test["recall_at_capacity"], "strict rank recall")
        _same(ranked.precision, test["precision_at_capacity"], "strict rank precision")
        _same(fixed.threshold, selection["thresholds"]["capacity_threshold"], "strict frozen score threshold")
        criteria = strict["criteria"]
        _unique(criteria, ["criterion"], "strict classification criteria")
        if len(criteria) != len(summary["success"]):
            _fail("Strict classification success summary and criteria have different sizes.")
        for item in summary["success"]:
            row = _one_row(criteria, "criterion", item["criterion"], "strict success criterion")
            for column in ("target", "achieved"):
                _same(row[column], item[column], "strict success " + column)
            if str(row["met"]).lower() not in ("true", "false") or (str(row["met"]).lower() == "true") != item["met"]:
                _fail("Strict classification success attainment disagrees.")
        months = strict["months"]
        if not months.empty:
            _unique(months, ["t_month"], "strict monthly assessment")
            _same(months.n.sum(), n, "strict monthly order counts")
            _unit_metrics(months, ("pr_auc", "recall_at_capacity"), "strict monthly")
        results = strict["test_results"]
        if not results.empty:
            _unique(results, ["model", "split"], "strict test results")
            selected_rows = results[results.model.eq("strict selection: " + family)]
            for split, saved in (("train (in-sample)", train), ("test Jun-Aug 2018", test)):
                row = _one_row(selected_rows, "split", split, "strict result split")
                for metric in ("pr_auc", "recall_at_capacity", "n"):
                    _same(row[metric], saved[metric], "strict CSV " + split + " " + metric)
            cv_row = _one_row(selected_rows, "split", "CV mean (labels known by 1 Jun)", "strict validation split")
            _same(cv_row.pr_auc, chosen.cv_pr_auc_mean, "strict CSV CV PR-AUC")
            _same(cv_row.recall_at_capacity, chosen.cv_recall_at_capacity_mean, "strict CSV CV recall")
            _unit_metrics(results, ("pr_auc", "recall_at_capacity"), "strict results")
        paired = strict["paired"]
        if not paired.empty:
            _unique(paired, ["comparison"], "strict paired bootstrap")
            if len(paired) != len(summary["paired_vs_v1"]):
                _fail("Strict paired CSV and summary have different sizes.")
            for item in summary["paired_vs_v1"]:
                row = _one_row(paired, "comparison", item["comparison"], "strict paired comparison")
                for metric in ("d_pr_auc", "d_recall"):
                    _same(row[metric], item[metric], "strict paired " + metric)
                    csv_ci, json_ci = _ci(row[metric + "_ci"], metric), _ci(item[metric + "_ci"], metric)
                    for actual, expected in zip(csv_ci, json_ci):
                        _same(actual, expected, "strict paired " + metric + " CI")
        folds = strict["fold_table"]
        if not folds.empty:
            _unique(folds, ["fold"], "strict cutoff folds")
            if set(folds.fold) != {"fold1", "fold2", "fold3", "fold4", "fold5", "final"}:
                _fail("Strict classification fold table is incomplete.")
            for prefix, removed in (("train_rows", "removed_by_R1"), ("scored_rows", "removed_by_R2")):
                if not (folds[prefix + "_v1"] - folds[prefix + "_strict"]).eq(folds[removed]).all():
                    _fail("Strict classification fold removals disagree with saved row counts.")
            final = _one_row(folds, "fold", "final", "strict final fold")
            _same(final.train_rows_strict, train["n"], "strict final training rows")
            _same(final.scored_rows_strict, n, "strict final test rows")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, EvidenceError):
            raise
        _fail("Invalid strict classification evidence schema: " + str(error))


def _validate_experiment_update(experiments, original, reference_ids):
    if not experiments["available"]:
        return
    try:
        selection, results, ledger = experiments["selection"], experiments["test_results"], experiments["ledger"]
        if not isinstance(selection["recommended"], bool):
            _fail("The Part B recommendation must be an explicit boolean.")
        _unique(results, ["model"], "Part B test results")
        _unique(ledger, ["step", "experiment"], "Part B experiment ledger")
        _unit_metrics(results, ("test_pr_auc", "test_recall_at_capacity", "test_brier"), "Part B test")
        _unit_metrics(ledger, ("mean_val_pr_auc", "mean_val_recall_at_capacity", "test_pr_auc", "test_recall_at_capacity"), "Part B ledger")
        chosen = _one_row(results, "model", selection["v2_family"], "Part B selected family")
        selected_rows = results[results.is_v2.astype(str).str.lower().eq("true")]
        if len(selected_rows) != 1 or selected_rows.iloc[0].model != selection["v2_family"]:
            _fail("Part B test results mark a different v2 family than selection.")
        headline = _one_row(results, "model", "v1 frozen forest (headline)", "Part B v1 reference")
        for metric in ("pr_auc", "recall_at_capacity", "brier"):
            _same(headline["test_" + metric], original["classification_summary"]["test"][metric], "Part B v1 reference " + metric)
        s0 = _one_row(ledger, "step", "S0", "Part B original reference")
        s6 = _one_row(ledger, "step", "S6", "Part B v2 decision")
        for metric in ("test_pr_auc", "test_recall_at_capacity"):
            _same(s0[metric], headline[metric], "Part B S0 " + metric)
            _same(s6[metric], chosen[metric], "Part B S6 " + metric)
        locked = experiments["test_lock"]
        if locked:
            if reference_ids is not None and locked["test_ids_sha256"] != reference_ids:
                _fail("Part B and v1 test order hashes differ.")
            evaluations = [item for item in locked["evaluations"] if item.get("selection_sha256") == selection["selection_sha256"]]
            if len(evaluations) != 1 or evaluations[0].get("v2_family") != selection["v2_family"]:
                _fail("Part B test lock identifies a different selection.")
            for metric in ("pr_auc", "recall_at_capacity"):
                _same(evaluations[0][metric], chosen["test_" + metric], "Part B test lock " + metric)
        stack = experiments["stacking"]
        if stack:
            if not isinstance(stack["retained"], bool):
                _fail("The classification stack retention decision must be an explicit boolean.")
            weights = stack["final_weights"]
            if set(weights) != set(stack["bases"]) or any(not 0 <= _number(value, "classification stack weight") <= 1 for value in weights.values()):
                _fail("Classification stack weights and base models disagree.")
            _same(sum(weights.values()), 1, "classification stack weight sum")
            best = _one_row(results, "model", stack["best_single"], "classification stack single reference")
            for metric in ("pr_auc", "recall_at_capacity"):
                _same(stack["best_single_test_" + metric], best["test_" + metric], "classification stack single " + metric)
                _number(stack["test_" + metric], "classification stack test " + metric)
            for metric in ("d_pr_auc", "d_recall"):
                _ci(stack["test_bootstrap_vs_best_single"][metric + "_ci"], "stack " + metric)
        daily = experiments["daily"]
        if not daily.empty:
            _unique(daily, ["model", "window"], "Part B daily batches")
            _unit_metrics(daily, ("pooled_daily_recall", "window_recall"), "Part B daily")
            for version, target in (("v1", headline), ("v2", chosen)):
                rows = daily[daily.model.eq(version)]
                row = _one_row(rows, "window", "test Jun-Aug 2018", "Part B daily test window")
                _same(row.orders, original["classification_summary"]["test"]["n"], "Part B daily test orders")
                _same(row.window_recall, target.test_recall_at_capacity, "Part B daily versus window recall")
        calibration = experiments["calibration"]
        if not calibration.empty:
            _unique(calibration, ["model", "method"], "Part B calibration decisions")
            _unit_metrics(calibration, ("mean_raw_brier", "mean_calibrated_brier"), "Part B calibration")
        calibration_test = experiments["calibration_test"]
        if not calibration_test.empty:
            _unique(calibration_test, ["model"], "Part B calibration test")
            _unit_metrics(calibration_test, ("test_raw_brier", "test_sigmoid_brier", "test_pr_auc_raw", "test_pr_auc_sigmoid"), "Part B calibration test")
            for version, target in (("v1", headline), ("v2", chosen)):
                row = _one_row(calibration_test, "model", version, "Part B calibrated model")
                _same(row.test_raw_brier, target.test_brier, "Part B calibration raw Brier")
                _same(row.test_pr_auc_raw, target.test_pr_auc, "Part B calibration raw PR-AUC")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, EvidenceError):
            raise
        _fail("Invalid Part B classification evidence schema: " + str(error))


def _classification_update(root, original):
    update = {"source_paths": {}, "missing_optional": []}
    for version, directory in _UPDATE_DIRECTORIES.items():
        base = root / "outputs/classification" / directory
        section = {"source_paths": {}, "missing_optional": []}
        missing_keys = []
        for key, relative in _UPDATE_JSON[version].items():
            path = base / relative
            section["source_paths"][key] = str(path)
            if path.is_file():
                section[key] = _read_json(path)
            else:
                section[key] = {}
                missing_keys.append(key)
                section["missing_optional"].append(str(path))
        for key, (filename, columns) in _UPDATE_TABLES[version].items():
            path = base / "tables" / filename
            section["source_paths"][key] = str(path)
            section[key] = _read_table(path, False, columns, section["missing_optional"])
            if not path.is_file():
                missing_keys.append(key)
        section["missing_core"] = [key for key in _UPDATE_CORE[version] if key in missing_keys]
        section["available"] = not section["missing_core"]
        update[version] = section
        update["source_paths"].update({version + "." + key: value for key, value in section["source_paths"].items()})
        update["missing_optional"].extend(section["missing_optional"])
    reference_path = root / "outputs/classification/test_lock.json"
    update["source_paths"]["v1_test_lock"] = str(reference_path)
    reference = _read_json(reference_path) if reference_path.is_file() else {}
    if not reference_path.is_file():
        update["missing_optional"].append(str(reference_path))
    reference_ids = reference.get("test_ids_sha256")
    strict_ids = update["strict"]["test_lock"].get("test_ids_sha256")
    experiment_ids = update["experiments"]["test_lock"].get("test_ids_sha256")
    if strict_ids and experiment_ids and strict_ids != experiment_ids:
        _fail("Strict classification and Part B test order hashes differ.")
    update["available"] = update["strict"]["available"] or update["experiments"]["available"]
    update["metadata"] = {
        "headline_version": "v1", "independent_holdout": False,
        "strict_rank_scope": "entire June–August test window",
        "daily_scope": "Part B v1/v2 retrospective daily batches; not strict-cutoff daily results or an intervention study",
        "v1_test_ids_sha256": reference_ids,
    }
    _validate_strict_update(update["strict"], original, reference_ids)
    _validate_experiment_update(update["experiments"], original, reference_ids)
    return update


def _validate(evidence):
    reg = evidence["regression_summary"]
    clf = evidence["classification_summary"]
    try:
        reg_test, clf_test = reg["test"], clf["test"]
        reg_n, clf_n = int(reg_test["n"]), int(clf_test["n"])
        selected_reg = reg["selected_candidate"]
        selected_clf = evidence["classification_selection"]["selected_family"]
    except (KeyError, TypeError, ValueError) as error:
        _fail("Main-model summary or selection has an invalid schema: " + str(error))
    # These are cohort integrity checks, never replacements for source metrics.
    if reg_n != 18603 or clf_n != 21058:
        _fail(f"Unexpected published test cohorts: regression={reg_n}, classification={clf_n}; expected 18603 and 21058.")
    _same(reg.get("test_n"), reg_n, "regression test count")
    if evidence["regression_selection"].get("best_single", {}).get("candidate") != selected_reg:
        _fail("Regression test summary and frozen main selection name different models.")
    _unique(evidence["regression_comparison"], ["candidate"], "main regression comparison")
    row = _one_row(evidence["regression_comparison"], "candidate", selected_reg, "selected regression")
    _same(row.test_mae, reg_test["mae"], "selected regression test MAE")
    _unique(evidence["classification_cv"], ["family"], "classification CV summary")
    row = _one_row(evidence["classification_cv"], "family", selected_clf, "selected classifier")
    selection_cv = evidence["classification_selection"].get("cv", {})
    _same(row.cv_pr_auc_mean, selection_cv.get("cv_pr_auc_mean"), "selected classification CV PR-AUC")
    ops = evidence["classification_operating"]
    _unique(ops, ["rule"], "classification operating rules")
    ranked = _one_row(ops, "rule", "top 20% of the test window (capacity rule)", "rank operating rule")
    frozen = _one_row(ops, "rule", "frozen capacity threshold from validation", "fixed operating threshold")
    _same(ranked.flag_rate, .2, "rank operating capacity")
    # CSV operating points are rounded to four decimals; JSON keeps precision.
    _same(ranked.recall, clf_test["recall_at_capacity"], "rank recall", tolerance=5.1e-5)
    _same(ranked.precision, clf_test["precision_at_capacity"], "rank precision", tolerance=5.1e-5)
    threshold = evidence["classification_selection"].get("thresholds", {}).get("capacity_threshold")
    _same(frozen.threshold, threshold, "frozen validation capacity threshold", tolerance=5.1e-5)
    criteria = evidence["classification_criteria"]
    _unique(criteria, ["criterion"], "classification success criteria")
    success = clf.get("success")
    if not isinstance(success, list) or len(criteria) != len(success):
        _fail("Classification success summary and criteria table differ in size.")
    for item in success:
        criterion = _one_row(criteria, "criterion", item["criterion"], "classification success criterion")
        for column in ("target", "achieved"):
            _same(criterion[column], item[column], "classification success " + column)
        recorded = str(criterion["met"]).lower()
        if recorded not in ("true", "false") or (recorded == "true") != item["met"]:
            _fail("Classification success summary and criteria table disagree on attainment.")
    median = _one_row(evidence["regression_comparison"], "candidate", "Median", "training-median baseline")
    _same(1 - reg_test["mae"] / median.test_mae, reg.get("success", {}).get("mae_reduction_vs_median"),
          "regression improvement over median baseline")
    _same(reg.get("success", {}).get("minimum_reduction"),
          evidence["regression_protocol"].get("success_criteria", {}).get("mae_reduction_vs_training_median"),
          "regression success improvement target")
    for key in ("regression_months", "classification_months"):
        if not evidence[key].empty:
            wanted = reg_n if key.startswith("regression") else clf_n
            _same(pd.to_numeric(evidence[key]["n"], errors="raise").sum(), wanted, key + " order counts")
    for key in ("classification_months", "classification_cv"):
        col = "pr_auc" if key.endswith("months") else "cv_pr_auc_mean"
        if not evidence[key].empty and not evidence[key][col].between(0, 1).all():
            _fail("Classification PR-AUC outside [0,1] in " + key)
    cv = evidence["stacking_cv"]
    if not cv.empty:
        _unique(cv, ["candidate"], "14-combination chronological summary")
        if len(cv) != 14 or not cv.folds.eq(5).all() or not cv.role.eq("combination").all():
            _fail("Supplementary chronological CV must contain 14 combinations with five folds each.")
    supplement = evidence["stacking_comparison"]
    if not supplement.empty:
        _unique(supplement, ["candidate"], "supplementary test comparison")
        if not supplement.n.eq(reg_n).all():
            _fail("Supplementary regression assessment uses a different test cohort.")
        saved_main = supplement[supplement.candidate.eq("HGB_risk_current_main")]
        if len(saved_main) == 1:
            _same(saved_main.iloc[0].mae, reg_test["mae"], "saved main versus supplementary test MAE")
    weights = evidence["stacking_weights"]
    if not weights.empty:
        _unique(weights, ["candidate", "member"], "frozen supplementary weights")
        if weights.weight.isna().any() or not weights.weight.between(0, 1).all():
            _fail("Frozen supplementary weights are invalid.")
        for candidate, rows in weights.groupby("candidate"):
            _same(rows.weight.sum(), 1, "supplementary weight sum: " + candidate)
    return reg_n, clf_n


def load_evidence(phase_root=None):
    """Return JSON objects, original DataFrames and source-aware metadata.

    ``phase_root`` may point to Phase_3 or its repository parent. No metric is
    hardcoded or overwritten. Additional selection/protocol JSON keys and exact
    ``source_paths`` make model identity and threshold provenance available to
    the UI. ``regression_cv_folds`` combines the original main, diagnostic,
    reference and original stacking fold tables, each with an evidence_source
    column; this includes HGB_risk, which is absent from the main search table.
    """
    root = Path(phase_root).expanduser().resolve() if phase_root is not None else Path(__file__).resolve().parents[1]
    if not (root / "outputs").is_dir() and (root / "Phase_3/outputs").is_dir():
        root = root / "Phase_3"
    sources, missing = {}, []
    result = {}
    for key, relative in _JSON_FILES.items():
        path = root / relative
        sources[key] = str(path)
        result[key] = _read_json(path)
    for key, (relative, required, columns) in _TABLE_FILES.items():
        path = root / relative
        sources[key] = str(path)
        result[key] = _read_table(path, required, columns, missing)
    cv_frames = []
    sources["regression_cv_folds"] = [str(root / relative) for relative in _CV_FILES]
    for relative in _CV_FILES:
        frame = _read_table(root / relative, False, {"candidate", "fold", "val_mae", "val_n"}, missing)
        if not frame.empty:
            frame = frame.copy()
            frame["evidence_source"] = relative
            cv_frames.append(frame)
    result["regression_cv_folds"] = (pd.concat(cv_frames, ignore_index=True, sort=False) if cv_frames
        else pd.DataFrame(columns=["candidate", "fold", "val_mae", "val_n", "evidence_source"]))
    if not result["regression_cv_folds"].empty:
        _unique(result["regression_cv_folds"], ["candidate", "fold"], "combined main regression folds")
    reg_n, clf_n = _validate(result)
    protocol = result["regression_protocol"]
    if not isinstance(protocol.get("test_previously_exposed"), bool):
        _fail("Regression protocol must explicitly state historical test exposure.")
    result["classification_update"] = _classification_update(root, result)
    sources.update({"classification_update." + key: path
                    for key, path in result["classification_update"]["source_paths"].items()})
    missing.extend(result["classification_update"]["missing_optional"])
    result["source_paths"] = sources
    result["metadata"] = {
        "phase_root": str(root),
        "regression_prediction_point": "Purchase",
        "classification_prediction_point": "T = min(delivered, estimated)",
        "regression_test_n": reg_n,
        "classification_test_n": clf_n,
        "classification_rank_scope": "entire June–August test window",
        "historical_test_exposure": protocol["test_previously_exposed"],
        "classification_validation_scope": "five monthly expanding-window CV folds, January–May 2018",
        "classification_validation_is_separate_holdout": False,
        "classification_frozen_capacity_threshold": result["classification_selection"]["thresholds"]["capacity_threshold"],
        "regression_importance_unit": "increase in validation MAE, days",
        "classification_importance_unit": "drop in validation PR-AUC",
        "missing_optional": missing,
        "regression_cv_folds_complete": all(str(root / relative) not in missing for relative in _CV_FILES),
        "evidence_notes": [
            "Rank-based top 20% and the frozen validation probability threshold are different operating rules.",
            "The v1 headline ranking covers the whole test window. Part B daily-batch evidence, when available, is retrospective v1/v2 analysis, not strict-cutoff daily performance or a real intervention.",
            "CV is the mean of monthly validation folds, not another independent validation holdout.",
            "Supplementary stacking results retain their separate assessment labels and do not replace the frozen main model.",
            "This loader exposes saved evaluation evidence; it does not provide or refit a predictor.",
            "Strict-cutoff and Part B results supplement the original v1 headline; all reuse historically examined test orders.",
        ],
    }
    return result
