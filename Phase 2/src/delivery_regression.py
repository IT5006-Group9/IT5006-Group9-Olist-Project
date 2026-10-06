"""Order-level delivery regression from course CSVs; temporal validation and OOF.

The final holdout remains unscored until the stacking design is frozen.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import zipfile
import importlib.metadata
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
SEED = 5006
NUMERIC = ["promised_lead_time_days", "max_distance_km", "total_price",
           "total_freight", "freight_ratio", "item_count", "seller_count",
           "avg_product_weight_g", "any_distance_missing", "any_weight_missing"]
CATEGORICAL = ["customer_state", "route_group", "category_group",
               "purchase_month", "purchase_weekday", "purchase_hour"]
FEATURES = NUMERIC + CATEGORICAL
VALIDATION_START = pd.Timestamp("2018-03-01")
VALIDATION_END = pd.Timestamp("2018-04-01")
TEST_START = pd.Timestamp("2018-07-01")
FOLDS = [("2017-07-01", "2017-10-01"),
         ("2017-10-01", "2018-01-01"),
         ("2018-01-01", "2018-03-01")]


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def prepare_data(csv_dir=None, archive=None, output_root=None, geography_policy="screened_median"):
    """Read raw course tables; deterministic joins only, no learned preprocessing.

    output_root isolates a prepared revision without overwriting existing fits.
    legacy_mean reproduces the first run; the audited default screens geography
    and uses representative medians. Neither policy fits or scores a model.
    """
    destination = Path(output_root) if output_root is not None else ROOT
    if geography_policy not in ["legacy_mean", "screened_median"]:
        raise ValueError("Use legacy_mean or audited screened_median geography.")
    for folder in ["data", "models", "outputs/tables", "outputs/figures",
                   "outputs/predictions", "outputs/oof"]:
        (destination / folder).mkdir(parents=True, exist_ok=True)
    configured = csv_dir or os.environ.get("OLIST_CSV_DIR")
    csv_dir = Path(configured) if configured else None
    archive = Path(archive or os.environ.get("OLIST_ARCHIVE", ROOT.parent /
        "01_当前小组项目/课程配套资料/IT5006_Project-Data.zip"))
    if csv_dir is None and not archive.is_file():
        raise FileNotFoundError("Set OLIST_CSV_DIR to the course CSV folder or OLIST_ARCHIVE to its ZIP.")
    inventory, tables = [], {}
    names = {"orders":"olist_orders_dataset.csv", "items":"olist_order_items_dataset.csv",
             "customers":"olist_customers_dataset.csv", "sellers":"olist_sellers_dataset.csv",
             "products":"olist_products_dataset.csv", "geo":"olist_geolocation_dataset.csv",
             "translation":"product_category_name_translation.csv"}
    for key, name in names.items():
        if csv_dir:
            content = (csv_dir / name).read_bytes()
        else:
            with zipfile.ZipFile(archive) as z:
                content = z.read("Olist_CSV/" + name)
        from io import BytesIO
        tables[key] = pd.read_csv(BytesIO(content))
        inventory.append({"table":key, "file":name, "rows":len(tables[key]),
                          "sha256":hashlib.sha256(content).hexdigest()})
    pd.DataFrame(inventory).to_csv(destination / "outputs/tables/source_inventory.csv", index=False)
    orders, items = tables["orders"], tables["items"]
    assert orders.order_id.is_unique and not items.duplicated(["order_id", "order_item_id"]).any()
    dates = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    for col in dates:
        orders[col] = pd.to_datetime(orders[col], errors="raise")
    for key, col in [("customers", "customer_id"), ("sellers", "seller_id"),
                     ("products", "product_id"), ("translation", "product_category_name")]:
        assert tables[key][col].is_unique, key
    assert items.order_id.isin(orders.order_id).all()
    assert items.product_id.isin(tables["products"].product_id).all()
    assert items.seller_id.isin(tables["sellers"].seller_id).all()
    assert orders.customer_id.isin(tables["customers"].customer_id).all()
    products = tables["products"].merge(tables["translation"], on="product_category_name",
                                         how="left", validate="many_to_one")
    # Pure source-reference transform, shared with the inspectable audit.
    from data_preparation_audit import geographic_reference
    geo = geographic_reference(tables["geo"], geography_policy)
    frame = (items.merge(products[["product_id", "product_category_name_english", "product_weight_g"]],
                         on="product_id", how="left", validate="many_to_one")
             .merge(orders[["order_id", "customer_id", "order_status", *dates]],
                    on="order_id", how="left", validate="many_to_one")
             .merge(tables["customers"][["customer_id", "customer_unique_id", "customer_state", "customer_zip_code_prefix"]],
                    on="customer_id", how="left", validate="many_to_one")
             .merge(tables["sellers"][["seller_id", "seller_state", "seller_zip_code_prefix"]],
                    on="seller_id", how="left", validate="many_to_one"))
    assert len(frame) == len(items)
    for side in ["customer", "seller"]:
        frame = frame.merge(geo.rename(columns={"geolocation_lat":side+"_lat", "geolocation_lng":side+"_lng"}),
                            left_on=side+"_zip_code_prefix", right_index=True,
                            how="left", validate="many_to_one")
    lat1, lon1, lat2, lon2 = [np.radians(frame[c]) for c in ["customer_lat", "customer_lng", "seller_lat", "seller_lng"]]
    h = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    frame["distance_km"] = (6371*2*np.arcsin(np.sqrt(h.clip(0, 1)))).round(1)
    frame["same_state"] = frame.customer_state.eq(frame.seller_state)
    g = frame.groupby("order_id", sort=True)
    result = g.agg(purchase_ts=(dates[0], "first"), delivered_ts=(dates[1], "first"),
        estimated_ts=(dates[2], "first"), order_status=("order_status", "first"),
        customer_unique_id=("customer_unique_id", "first"), customer_state=("customer_state", "first"),
        total_price=("price", "sum"), total_freight=("freight_value", "sum"),
        item_count=("order_item_id", "size"), seller_count=("seller_id", "nunique"),
        max_distance_km=("distance_km", "max"), all_same_state=("same_state", "all"),
        category_count=("product_category_name_english", "nunique"),
        first_category=("product_category_name_english", "first"))
    result["any_distance_missing"] = g.distance_km.count().lt(g.size()).astype(int)
    positive_weight = frame.product_weight_g.where(frame.product_weight_g.gt(0))
    result["any_weight_missing"] = positive_weight.groupby(frame.order_id).count().lt(g.size()).astype(int)
    result["avg_product_weight_g"] = positive_weight.groupby(frame.order_id).mean().mask(result.any_weight_missing.astype(bool))
    category_missing = g.product_category_name_english.count().lt(g.size())
    result["category_group"] = np.select([category_missing, result.category_count.gt(1)],
        ["Missing/incomplete category", "Multiple categories"], default=result.first_category)
    routes_known = (frame.customer_state.notna() & frame.seller_state.notna()).groupby(frame.order_id).all()
    result["route_group"] = np.select([~routes_known, ~result.all_same_state],
        ["Unknown", "Any seller cross-state"], default="All sellers same-state")
    result["freight_ratio"] = result.total_freight / result.total_price.where(result.total_price.gt(0))
    result["lead_time_days"] = (result.delivered_ts-result.purchase_ts).dt.total_seconds()/86400
    result["promised_lead_time_days"] = (result.estimated_ts-result.purchase_ts).dt.total_seconds()/86400
    for col, values in [("purchase_month", result.purchase_ts.dt.month),
                        ("purchase_weekday", result.purchase_ts.dt.dayofweek),
                        ("purchase_hour", result.purchase_ts.dt.hour)]:
        result[col] = values.astype(int)
    result = result.reset_index()
    valid = result.order_status.eq("delivered") & result.lead_time_days.notna() & result.lead_time_days.ge(0)
    eligible = result.loc[valid].sort_values(["purchase_ts", "order_id"]).reset_index(drop=True)
    assert len(eligible) == 96470, "Course cohort differs from Phase 1; reconcile before modelling."
    assert eligible.order_id.is_unique and np.isfinite(eligible.lead_time_days).all()
    assert eligible.promised_lead_time_days.notna().all() and eligible.promised_lead_time_days.ge(0).all()
    eligible["split"] = "pending_label_train_cutoff"
    eligible.loc[(eligible.purchase_ts < VALIDATION_START) & (eligible.delivered_ts < VALIDATION_START), "split"] = "train"
    eligible.loc[(eligible.purchase_ts >= VALIDATION_START) & (eligible.purchase_ts < VALIDATION_END), "split"] = "validation_label_pending"
    eligible.loc[(eligible.purchase_ts >= VALIDATION_START) & (eligible.purchase_ts < VALIDATION_END) &
                 (eligible.delivered_ts < TEST_START), "split"] = "validation"
    eligible.loc[(eligible.purchase_ts >= VALIDATION_END) & (eligible.purchase_ts < TEST_START), "split"] = "maturation_gap"
    eligible.loc[eligible.purchase_ts >= TEST_START, "split"] = "test_locked"
    version = "purchase_inputs_v2_screened_median" if geography_policy == "screened_median" else "purchase_inputs_v1_legacy_mean"
    eligible.attrs["preparation_version"] = version
    eligible.to_csv(destination / "data/eligible_orders.csv", index=False)
    eligible[["order_id", "purchase_ts", "delivered_ts", "split"]].to_csv(destination / "data/split_manifest.csv", index=False)
    split_summary = eligible.groupby("split").agg(n=("order_id", "size"),
        first_purchase=("purchase_ts", "min"), last_purchase=("purchase_ts", "max"),
        tail_over_60=("lead_time_days", lambda s: int(s.gt(60).sum()))).reset_index()
    split_summary.to_csv(destination / "outputs/tables/split_summary.csv", index=False)
    summary = {"source_orders":len(orders), "item_rows":len(items), "orders_with_items":len(result),
        "orders_without_items":len(orders)-len(result), "eligible_orders":len(eligible),
        "delivered_with_missing_target":int((result.order_status.eq("delivered") & result.lead_time_days.isna()).sum()),
        "retained_over_60_days":int(eligible.lead_time_days.gt(60).sum()),
        "missing_max_distance":int(eligible.max_distance_km.isna().sum()),
        "mean_target_days":float(eligible.lead_time_days.mean()), "median_target_days":float(eligible.lead_time_days.median()),
        "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest() if csv_dir is None else None,
        "timezone":"original timezone-naive timestamps; no conversion",
        "geo_assumption":"static representative zip coordinates from supplied archive; no time-varying geolocation history available",
        "preparation_version":version,
        "geography_policy":geography_policy}
    save_json(destination / "outputs/data_quality.json", summary)
    return eligible, split_summary, summary


def metric_row(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    assert len(y) == len(prediction) and np.isfinite(prediction).all()
    errors = prediction-y
    tail = y > 60
    return {"n":len(y), "MAE_days":mean_absolute_error(y, prediction),
            "RMSE_days":np.sqrt(mean_squared_error(y, prediction)), "R2":r2_score(y, prediction),
            "bias_days":float(errors.mean()), "P90_abs_error_days":float(np.quantile(np.abs(errors), .9)),
            "tail_n":int(tail.sum()), "tail_MAE_days":float(np.abs(errors[tail]).mean()) if tail.any() else np.nan}


def predict_days(model, features):
    """Non-negative operational predictions; unseen levels are audited separately."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories in columns.*", category=UserWarning)
        return np.maximum(0, model.predict(features))


def pipeline(kind, params=None, log_target=False):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                                     drop="first", sparse_output=False))])
    # Calendar domains are known without data: never map an unseen future month
    # to a rare historical month's learned effect.
    calendar = OneHotEncoder(categories=[list(range(1, 13)), list(range(7)), list(range(24))],
                             handle_unknown="error", drop="first", sparse_output=False)
    prep = ColumnTransformer([("numeric", numeric, NUMERIC),
                              ("categorical", categorical, CATEGORICAL[:3]),
                              ("calendar", calendar, CATEGORICAL[3:])], remainder="drop")
    if kind == "linear":
        model = LinearRegression()
    elif kind == "ridge":
        model = Ridge(**(params or {}))
    elif kind == "tree":
        model = DecisionTreeRegressor(random_state=SEED, **(params or {}))
    elif kind == "forest":
        model = RandomForestRegressor(n_estimators=100, n_jobs=4, random_state=SEED,
                                      bootstrap=True, max_samples=.8, **(params or {}))
    else:
        raise ValueError(kind)
    if log_target:
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("preprocess", prep), ("regressor", model)])


def model_grid():
    # Limited candidate grid; Ridge was added after development instability diagnostics.
    # The log version is a controlled tail-handling trial.
    return {"linear":("linear", {}, False),
            "ridge_a10":("ridge", {"alpha":10}, False),
            "ridge_a100":("ridge", {"alpha":100}, False),
            "tree_d8":("tree", {"max_depth":8, "min_samples_leaf":20}, False),
            "tree_d14":("tree", {"max_depth":14, "min_samples_leaf":20}, False),
            "rf_d16_l5":("forest", {"max_depth":16, "min_samples_leaf":5, "max_features":.8}, False),
            "rf_d24_l10":("forest", {"max_depth":24, "min_samples_leaf":10, "max_features":.8}, False),
            "rf_log_d24_l10":("forest", {"max_depth":24, "min_samples_leaf":10, "max_features":.8}, True)}


def train_and_validate(eligible):
    """Tune only on temporal train CV; compare on a separate March validation period."""
    train = eligible.loc[eligible.split.eq("train")].copy()
    validation = eligible.loc[eligible.split.eq("validation")].copy()
    assert train.delivered_ts.max() < VALIDATION_START
    assert train.purchase_ts.max() < validation.purchase_ts.min()
    grid = model_grid()
    cv_rows, cv_predictions, fold_rows, fold_ids, unseen_rows = [], {}, [], [], []
    for fold, (start, end) in enumerate(FOLDS, 1):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        fit = train.loc[(train.purchase_ts < start) & (train.delivered_ts < start)]
        score = train.loc[(train.purchase_ts >= start) & (train.purchase_ts < end)]
        assert len(fit) and len(score) and not set(fit.order_id) & set(score.order_id)
        assert fit.delivered_ts.max() < start and fit.purchase_ts.max() < score.purchase_ts.min()
        fold_rows.append({"fold":fold, "forecast_cutoff":str(start), "end":str(end), "train_n":len(fit),
                          "validation_n":len(score), "latest_training_receipt":str(fit.delivered_ts.max())})
        fold_ids.extend([{"order_id":oid, "fold":fold, "role":"fit"} for oid in fit.order_id])
        fold_ids.extend([{"order_id":oid, "fold":fold, "role":"predict"} for oid in score.order_id])
        for col in CATEGORICAL[:3]:
            unseen_rows.append({"period":f"CV {fold}", "field":col,
                                "unknown_rows":int((~score[col].isin(fit[col])).sum()), "evaluation_rows":len(score)})
        for name, (kind, params, log_target) in grid.items():
            began = time.perf_counter()
            model = pipeline(kind, params, log_target)
            model.fit(fit[FEATURES], fit.lead_time_days)
            pred = predict_days(model, score[FEATURES])
            cv_rows.append({"model":name, "family":"linear" if kind in ["linear", "ridge"] else "tree-based", "algorithm":kind, "fold":fold,
                            **metric_row(score.lead_time_days, pred), "fit_predict_seconds":time.perf_counter()-began})
            if fold == 1 and kind in ["linear", "ridge"]:
                coefficients = pd.DataFrame({"encoded_feature":model.named_steps["preprocess"].get_feature_names_out(),
                                              "coefficient_days":model.named_steps["regressor"].coef_})
                coefficients.loc[coefficients.encoded_feature.str.startswith("calendar__")].to_csv(
                    ROOT / f"outputs/tables/early_fold_{name}_calendar_coefficients.csv", index=False)
            cv_predictions.setdefault(name, []).append(pd.DataFrame({"order_id":score.order_id.values,
                "fold":fold, "prediction_days":pred}))
            print(f"CV {fold}/3 {name}: MAE={cv_rows[-1]['MAE_days']:.3f}", flush=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(ROOT / "outputs/tables/cv_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ROOT / "outputs/tables/fold_summary.csv", index=False)
    pd.DataFrame(fold_ids).to_csv(ROOT / "data/cv_fold_manifest.csv", index=False)
    for col in CATEGORICAL[:3]:
        unseen_rows.append({"period":"March validation", "field":col,
                            "unknown_rows":int((~validation[col].isin(train[col])).sum()), "evaluation_rows":len(validation)})
    pd.DataFrame(unseen_rows).to_csv(ROOT / "outputs/tables/unseen_categories.csv", index=False)
    summary = cv.groupby(["family", "algorithm", "model"]).agg(CV_MAE_mean=("MAE_days", "mean"),
        CV_MAE_std=("MAE_days", "std"), CV_RMSE_mean=("RMSE_days", "mean")).reset_index()
    summary.to_csv(ROOT / "outputs/tables/cv_summary.csv", index=False)
    chosen = {"linear":"linear",
              "ridge":summary.loc[summary.model.str.startswith("ridge")].sort_values(["CV_MAE_mean", "model"]).iloc[0].model}
    chosen.update({family: summary.loc[summary.algorithm.eq(family)].sort_values(["CV_MAE_mean", "model"]).iloc[0].model
                   for family in ["tree", "forest"]})
    models, comparisons, predictions = {}, [], validation[["order_id", "lead_time_days", "route_group", "customer_state"]].copy()
    for name, values in [("promise", validation.promised_lead_time_days.to_numpy()),
                         ("train_mean", np.full(len(validation), train.lead_time_days.mean())),
                         ("train_median", np.full(len(validation), train.lead_time_days.median()))]:
        comparisons.append({"model":name, "role":"reference", "period":"March 2018 validation", **metric_row(validation.lead_time_days, values)})
        predictions[name] = values
    oof = train.loc[train.purchase_ts >= pd.Timestamp(FOLDS[0][0]), ["order_id", "purchase_ts", "lead_time_days"]].copy()
    for family, name in chosen.items():
        kind, params, log_target = grid[name]
        model = pipeline(kind, params, log_target)
        model.fit(train[FEATURES], train.lead_time_days)
        pred = predict_days(model, validation[FEATURES])
        models[family] = model
        joblib.dump(model, ROOT / f"models/{family}_development.joblib")
        comparisons.append({"model":family, "role":"variant_1" if family=="forest" else "regularized_variant" if family=="ridge" else "baseline",
                            "period":"March 2018 validation", **metric_row(validation.lead_time_days, pred)})
        predictions[family] = pred
        this_oof = pd.concat(cv_predictions[name], ignore_index=True).rename(columns={"prediction_days":family+"_oof_days"})
        assert this_oof.order_id.is_unique and set(this_oof.order_id)==set(oof.order_id)
        if "fold" not in oof:
            oof = oof.merge(this_oof, on="order_id", validate="one_to_one")
        else:
            oof = oof.merge(this_oof.drop(columns="fold"), on="order_id", validate="one_to_one")
        print(f"Validation {family}: MAE={comparisons[-1]['MAE_days']:.3f}", flush=True)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(ROOT / "outputs/tables/model_comparison.csv", index=False)
    predictions.to_csv(ROOT / "outputs/predictions/validation_predictions.csv", index=False)
    oof.sort_values(["purchase_ts", "order_id"]).to_csv(ROOT / "outputs/oof/base_oof_predictions.csv", index=False)
    train[FEATURES+ ["order_id", "lead_time_days"]].to_csv(ROOT / "data/stacking_train.csv", index=False)
    validation[FEATURES+ ["order_id", "lead_time_days"]].to_csv(ROOT / "data/stacking_validation.csv", index=False)
    eligible.loc[eligible.split.eq("test_locked"), FEATURES+["order_id"]].to_csv(ROOT / "data/locked_test_features.csv", index=False)
    save_json(ROOT / "outputs/selected_models.json", {"chosen_by_training_CV":chosen,
        "grid":grid, "seed":SEED, "features":FEATURES,
        "test_scored":False, "oof_rows":len(oof), "warmup_train_rows":len(train)-len(oof),
        "preparation_version":eligible.attrs.get("preparation_version", "unspecified_reloaded_schema"),
        "oof_note":"Limited configurations with Ridge added during development; columns selected by training CV, not nested. OOF is for meta-training, not an unbiased final score; March validation remains separate.",
        "versions":{p:importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]}})
    return comparison, cv, models, predictions, oof


def explain(models, eligible, predictions):
    validation = eligible.loc[eligible.split.eq("validation")]
    sample = validation.sample(n=min(2000, len(validation)), random_state=SEED)
    pi = permutation_importance(models["forest"], sample[FEATURES], sample.lead_time_days,
                                scoring="neg_mean_absolute_error", n_repeats=3, random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({"feature":FEATURES, "MAE_increase_days":pi.importances_mean,
                               "repeat_std_days":pi.importances_std}).sort_values("MAE_increase_days", ascending=False)
    importance.to_csv(ROOT / "outputs/tables/permutation_importance.csv", index=False)
    slices=[]
    for name in ["promise", "train_median", "linear", "ridge", "tree", "forest"]:
        for route, part in predictions.groupby("route_group"):
            slices.append({"model":name, "slice":route, **metric_row(part.lead_time_days, part[name])})
    pd.DataFrame(slices).to_csv(ROOT / "outputs/tables/route_errors.csv", index=False)
    return importance


def plot_results(comparison, cv, predictions, importance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    order = ["promise", "train_mean", "train_median", "linear", "ridge", "tree", "forest"]
    view = comparison.set_index("model").loc[order]
    for ax, metric, label in zip(axes, ["MAE_days", "RMSE_days"], ["(a) Mean absolute error", "(b) Root mean squared error"]):
        ax.barh(order, view[metric], color=["#9aa7b7"]*3+["#3268a8"]*3+["#bb8b2c"])
        ax.invert_yaxis();ax.set_xlabel("Days");ax.set_title(label)
        for i,v in enumerate(view[metric]):ax.text(v+.12,i,f"{v:.2f}",va="center")
        ax.set_xlim(0,view[metric].max()*1.2)
    fig.suptitle(f"March 2018 validation: {len(predictions):,} completed orders; final test remains locked")
    fig.tight_layout();fig.savefig(ROOT / "outputs/figures/01_model_comparison.png", dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    errors=predictions.forest-predictions.lead_time_days
    axes[0].hist(errors,bins=50,color="#3268a8");axes[0].axvline(0,color="#444",ls="--")
    axes[0].set(title="(a) Forest prediction error, full range",xlabel="Predicted minus actual days",ylabel="Orders")
    for name in ["train_median","linear","ridge","tree","forest"]:
        bins=pd.cut(predictions.lead_time_days,[0,7,14,30,60,np.inf],include_lowest=True)
        means=(predictions[name]-predictions.lead_time_days).abs().groupby(bins,observed=True).mean()
        axes[1].plot(range(len(means)),means.values,marker="o",label=name)
    axes[1].set(title="(b) Error by observed duration",xlabel="Actual duration group (days)",ylabel="MAE (days)")
    axes[1].set_xticks(range(5),["0-7","7-14","14-30","30-60",">60"]);axes[1].legend()
    fig.suptitle("Validation error analysis; outcome groups are diagnostics, not prediction inputs")
    fig.tight_layout();fig.savefig(ROOT / "outputs/figures/02_error_analysis.png",dpi=180);plt.close(fig)
    top=importance.head(10).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.barh(top.feature,top.MAE_increase_days,xerr=top.repeat_std_days,color="#3268a8")
    ax.axvline(0,color="#444",lw=.7);ax.set(xlabel="Increase in validation MAE after permutation (days)",
        title="Forest feature reliance: 2,000 fixed validation rows, 3 repeats")
    fig.tight_layout();fig.savefig(ROOT / "outputs/figures/03_feature_importance.png",dpi=180);plt.close(fig)


def run():
    eligible, split_summary, quality = prepare_data()
    comparison, cv, models, predictions, oof = train_and_validate(eligible)
    importance = explain(models, eligible, predictions)
    plot_results(comparison, cv, predictions, importance)
    return eligible, split_summary, quality, comparison, cv, importance, oof


if __name__ == "__main__":
    results = run()
    print(results[3].round(3).to_string(index=False), flush=True)
