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
            "The published ranking result covers the whole test window; no daily-capacity evaluation is supplied.",
            "CV is the mean of monthly validation folds, not another independent validation holdout.",
            "Supplementary stacking results retain their separate assessment labels and do not replace the frozen main model.",
            "This loader exposes saved evaluation evidence; it does not provide or refit a predictor.",
        ],
    }
    return result
