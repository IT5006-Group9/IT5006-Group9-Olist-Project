"""Audit saved results independently of the training run; never score the holdout."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
from io import BytesIO

import joblib
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delivery_regression import FEATURES, predict_days

e = pd.read_csv(ROOT / "data/eligible_orders.csv", parse_dates=["purchase_ts", "delivered_ts"])
p = pd.read_csv(ROOT / "outputs/predictions/validation_predictions.csv")
oof = pd.read_csv(ROOT / "outputs/oof/base_oof_predictions.csv")
manifest = pd.read_csv(ROOT / "data/cv_fold_manifest.csv")
folds = pd.read_csv(ROOT / "outputs/tables/fold_summary.csv")
comparison = pd.read_csv(ROOT / "outputs/tables/model_comparison.csv").set_index("model")
checks = {}

assert e.order_id.is_unique and len(e) == 96470
assert set(p.order_id) == set(e.loc[e.split.eq("validation"), "order_id"])
assert len(p) == 7003 and (p.lead_time_days > 60).sum() == 37
for model in comparison.index:
    error = p[model].to_numpy() - p.lead_time_days.to_numpy()
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    r2 = 1 - np.sum(error ** 2) / np.sum((p.lead_time_days - p.lead_time_days.mean()) ** 2)
    np.testing.assert_allclose([mae, rmse, r2, error.mean()], comparison.loc[model,
        ["MAE_days", "RMSE_days", "R2", "bias_days"]].to_numpy(float), rtol=1e-10, atol=1e-10)
checks["independent_metric_recalculation"] = "7 rows matched saved per-order predictions"

assert len(oof) == 39445 and oof.order_id.is_unique
assert not set(oof.order_id) & set(e.loc[~e.split.eq("train"), "order_id"])
assert oof[[c for c in oof if c.endswith("_oof_days")]].notna().all().all()
indexed = e.set_index("order_id")
for row in folds.itertuples():
    ids = manifest.loc[manifest.fold.eq(row.fold)]
    fit_ids = set(ids.loc[ids.role.eq("fit"), "order_id"])
    pred_ids = set(ids.loc[ids.role.eq("predict"), "order_id"])
    assert not fit_ids & pred_ids
    assert fit_ids <= set(e.loc[e.split.eq("train"), "order_id"])
    assert (indexed.loc[list(fit_ids), "delivered_ts"] < pd.Timestamp(row.forecast_cutoff)).all()
    assert (indexed.loc[list(fit_ids), "purchase_ts"] < pd.Timestamp(row.forecast_cutoff)).all()
    assert pred_ids == set(oof.loc[oof.fold.eq(row.fold), "order_id"])
    assert (indexed.loc[list(pred_ids), "purchase_ts"] >= pd.Timestamp(row.forecast_cutoff)).all()
    assert (indexed.loc[list(pred_ids), "purchase_ts"] < pd.Timestamp(row.end)).all()
checks["time_oof_and_label_visibility"] = "3 folds, common 39,445 rows; disjoint from validation/test"

v = pd.read_csv(ROOT / "data/stacking_validation.csv")
for name in ["linear", "ridge", "tree", "forest"]:
    loaded = joblib.load(ROOT / f"models/{name}_development.joblib")
    actual = pd.Series(predict_days(loaded, v[FEATURES]), index=v.order_id)
    expected = p.set_index("order_id")[name].reindex(actual.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-9, rtol=1e-9)
checks["reloaded_pipeline_inference"] = "4 pipelines reproduced validation predictions from CSV schema"

roles = pd.read_csv(ROOT / "config/feature_roles.csv")
assert set(roles.loc[roles.role.eq("input"), "field"]) == set(FEATURES)
locked = pd.read_csv(ROOT / "data/locked_test_features.csv")
assert set(locked) == set(FEATURES + ["order_id"]) and len(locked) == 12507
assert not any("test" in f.name for f in (ROOT / "outputs/predictions").glob("*.csv"))
assert json.loads((ROOT / "outputs/selected_models.json").read_text())["test_scored"] is False
checks["feature_allowlist_and_holdout"] = "16 inputs matched role table; test features only, no test scoring artifact"

phase1 = ROOT.parent / "03_配送时长分析/data/order_level_delivery.csv.gz"
if phase1.exists():
    old = pd.read_csv(phase1).set_index("order_id").loc[e.order_id]
    fields = ["lead_time_days", "promised_lead_time_days", "max_distance_km", "total_price",
              "total_freight", "item_count", "seller_count", "avg_product_weight_g",
              "any_distance_missing", "any_weight_missing", "freight_ratio"]
    for field in fields:
        np.testing.assert_allclose(e[field].to_numpy(float), old[field].to_numpy(float),
                                   rtol=1e-9, atol=1e-9, equal_nan=True)
    for field in ["customer_state", "route_group", "category_group"]:
        assert e[field].fillna("NA").tolist() == old[field].fillna("NA").tolist()
    checks["phase1_reconciliation"] = "96,470 target rows and 13 matching inputs reconciled; calendar represented separately"

# Reconstruct item totals directly for a fixed sample rather than trusting a derived table.
archive = ROOT.parent / "01_当前小组项目/课程配套资料/IT5006_Project-Data.zip"
if archive.exists():
    with zipfile.ZipFile(archive) as z:
        items = pd.read_csv(BytesIO(z.read("Olist_CSV/olist_order_items_dataset.csv")))
        orders = pd.read_csv(BytesIO(z.read("Olist_CSV/olist_orders_dataset.csv")))
    sample = e.sample(100, random_state=5006).set_index("order_id")
    raw = items.loc[items.order_id.isin(sample.index)].groupby("order_id").agg(
        total_price=("price", "sum"), total_freight=("freight_value", "sum"),
        item_count=("order_item_id", "size"), seller_count=("seller_id", "nunique"))
    for field in raw:
        np.testing.assert_allclose(sample[field], raw[field].reindex(sample.index), atol=1e-9)
    raw_orders = orders.set_index("order_id").loc[sample.index]
    target = (pd.to_datetime(raw_orders.order_delivered_customer_date) -
              pd.to_datetime(raw_orders.order_purchase_timestamp)).dt.total_seconds() / 86400
    np.testing.assert_allclose(sample.lead_time_days, target, atol=1e-9)
    checks["raw_source_spot_check"] = "100 deterministic order targets and item totals matched original CSV members"

nb = nbformat.read(ROOT / "notebooks/delivery_regression_phase2.ipynb", as_version=4)
nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == "code"]
assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
assert not [o for c in code for o in c.outputs if o.output_type == "error"]
checks["notebook_execution"] = f"{len(code)} code cells executed in order, no errors; nbformat valid"

# Tables used by the narrative: source scope and diagnostics, never fit/tune inputs.
period = e.loc[e.split.isin(["train", "validation"])].groupby("split").lead_time_days.agg(
    n="size", mean_days="mean", median_days="median").reset_index()
period.to_csv(ROOT / "outputs/tables/period_target_summary.csv", index=False)
early_fit = e.loc[e.split.eq("train") & e.purchase_ts.lt("2017-07-01") & e.delivered_ts.lt("2017-07-01")]
early_fit.groupby("purchase_month").size().rename("training_n").reset_index().to_csv(
    ROOT / "outputs/tables/early_fold_month_support.csv", index=False)
duration_groups = pd.cut(p.lead_time_days, [0, 7, 14, 30, 60, np.inf], include_lowest=True)
duration_rows = []
for group in duration_groups.cat.categories:
    subset = p.loc[duration_groups.eq(group)]
    duration_rows.append({"actual_duration_group":str(group), "n":len(subset),
        **{name + "_MAE_days":float(np.abs(subset[name] - subset.lead_time_days).mean())
           for name in comparison.index}})
pd.DataFrame(duration_rows).to_csv(ROOT / "outputs/tables/duration_errors.csv", index=False)

out = {"date":"2026-10-05", "checks":checks, "test_scored":False, "stacking_fitted":False}
(ROOT / "outputs/validation_audit.json").write_text(json.dumps(out, indent=2))
manifest_rows = []
for directory in ["data", "models", "outputs/oof", "outputs/predictions"]:
    for file in sorted((ROOT / directory).glob("*")):
        if file.is_file():
            manifest_rows.append({"path":str(file.relative_to(ROOT)), "bytes":file.stat().st_size,
                                  "sha256":hashlib.sha256(file.read_bytes()).hexdigest()})
pd.DataFrame(manifest_rows).to_csv(ROOT / "outputs/tables/local_handoff_manifest.csv", index=False)
print(json.dumps(out, indent=2))
