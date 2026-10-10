"""Reporting diagnostics for fixed models; no tuning or model selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import random_experiments as ex
import tree_extensions as ext
import time_risk_features as features


GROUPS = {
    "Geography and route": ["max_distance_km", "any_distance_missing", "customer_state",
        "route_group", "primary_seller_state", "primary_seller_city", "customer_city",
        "primary_state_route", "primary_city_route", "primary_zip_zone_route"],
    "Purchase date and calendar": ["purchase_ts", "purchase_month", "purchase_weekday", "purchase_hour"],
    "Quoted promise": ["promised_lead_time_days"],
    "Price and freight": ["total_price", "total_freight", "freight_ratio", "mean_unit_price"],
    "Seller identity and composition": ["primary_seller_id", "seller_count", "seller_state_count",
        "primary_seller_price_share"],
    "Product category and composition": ["category_group", "dominant_category", "item_count",
        "distinct_product_count", "category_count_observed", "dominant_category_price_share"],
    "Product weight and volume": ["avg_product_weight_g", "any_weight_missing", "max_product_weight_g",
        "total_known_weight_g", "max_product_volume_cm3", "total_known_volume_cm3", "any_volume_missing"],
}


def metrics(actual, prediction):
    y, p = np.asarray(actual, float), np.asarray(prediction, float)
    assert y.shape == p.shape and np.isfinite(y).all() and np.isfinite(p).all()
    return {"n": len(y), "MAE_days": float(mean_absolute_error(y, p)),
        "RMSE_days": float(np.sqrt(mean_squared_error(y, p))), "R2": float(r2_score(y, p)),
        "bias_days": float((p-y).mean()), "within_3_days_pct": float(100*(np.abs(p-y)<=3).mean())}


def fit_design_prediction(bundle, train, risk_oof):
    """Use the exact cross-fitted risk column used to fit the final regressor.

    Calling bundle.predict(train) would instead use the full-training classifier's
    in-sample probabilities. That is a different design matrix and is not the
    regressor's resubstitution score.
    """
    matrix = features.add_blocks(ext.transform(bundle.prep, train), train, bundle.pack,
        bundle.time_encoder, risk_oof, bundle.risk_scaler)
    return np.maximum(0, bundle.regressor.predict(matrix))


def reference_predictions(train, score, name):
    if name == "promise":
        return score.promised_lead_time_days.to_numpy()
    value = train.lead_time_days.mean() if name == "mean" else train.lead_time_days.median()
    return np.full(len(score), value)


def export_comparison(records, destination):
    """records contain full-training, complete OOF and fixed-test predictions."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    long_rows, fold_rows, wide_rows = [], [], []
    for record in records:
        model, role = record["model"], record["role"]
        training = metrics(record["train_actual"], record["train_prediction"])
        testing = metrics(record["test_actual"], record["test_prediction"])
        oof = record["oof"]
        assert len(oof) == 64634 and oof.order_id.is_unique
        assert sorted(oof.fold.unique()) == [1, 2, 3, 4, 5]
        folds = []
        for fold, part in oof.groupby("fold"):
            values = metrics(part.actual_days, part.predicted_days)
            folds.append(values)
            fold_rows.append({"model": model, "fold": int(fold), **values})
        cv = pd.DataFrame(folds)
        pooled = metrics(oof.actual_days, oof.predicted_days)
        for phase, values in [("train", training), ("oof_pooled", pooled), ("test", testing)]:
            long_rows.append({"model": model, "role": role, "phase": phase, **values})
        wide_rows.append({"model": model, "role": role, "feature_design": record["feature_design"],
            "Train_n": training["n"], "Train_MAE_days": training["MAE_days"],
            "Train_RMSE_days": training["RMSE_days"], "Train_R2": training["R2"],
            "CV_folds": 5, "CV_MAE_mean_days": float(cv.MAE_days.mean()),
            "CV_MAE_SD_sample_days": float(cv.MAE_days.std(ddof=1)),
            "CV_MAE_SD_population_days": float(cv.MAE_days.std(ddof=0)),
            "CV_RMSE_mean_days": float(cv.RMSE_days.mean()), "CV_R2_mean": float(cv.R2.mean()),
            "OOF_pooled_MAE_days": pooled["MAE_days"], "Test_n": testing["n"],
            "Test_MAE_days": testing["MAE_days"], "Test_RMSE_days": testing["RMSE_days"],
            "Test_R2": testing["R2"], "Test_bias_days": testing["bias_days"],
            "Test_within_3_days_pct": testing["within_3_days_pct"],
            "CV_minus_train_MAE_days": float(cv.MAE_days.mean()-training["MAE_days"])})
    pd.DataFrame(long_rows).to_csv(destination / "metrics_by_phase.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(destination / "cv_fold_metrics.csv", index=False)
    table = pd.DataFrame(wide_rows)
    table.to_csv(destination / "train_cv_test_comparison.csv", index=False)
    return table


def canonical_purchase_inputs(frame):
    """Recalculate deterministic calendar/interaction columns after permutation."""
    out = frame.copy()
    dates = pd.to_datetime(out.purchase_ts)
    out["purchase_month"] = dates.dt.month
    out["purchase_weekday"] = dates.dt.dayofweek
    out["purchase_hour"] = dates.dt.hour
    return ex.add_fixed_features(out)


def grouped_permutation(bundle, test, destination, sample_n=8000, repeats=5, seed=33):
    """End-to-end grouped permutation; both risk and regression paths respond.

    Diagnostic only on a fixed random test subset, never used for selecting
    features, parameters, or models. Groups use one row permutation per repeat.
    Correlated groups can share/substitute signal; increments are not additive.
    """
    destination = Path(destination)
    sample = test.sample(n=min(sample_n, len(test)), random_state=seed).sort_values("order_id").copy()
    y = sample.lead_time_days.to_numpy()
    original = bundle.predict(sample)
    baseline = metrics(y, original)["MAE_days"]
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(sample)) for _ in range(repeats)]
    rows = []
    for name, columns in GROUPS.items():
        assert set(columns).issubset(sample.columns), name
        for repeat, positions in enumerate(permutations, 1):
            changed = sample.copy()
            for column in columns:
                changed[column] = sample[column].iloc[positions].to_numpy()
            changed = canonical_purchase_inputs(changed)
            value = metrics(y, bundle.predict(changed))["MAE_days"]
            rows.append({"group": name, "repeat": repeat, "sample_n": len(sample),
                "baseline_MAE_days": baseline, "permuted_MAE_days": value,
                "MAE_increase_days": value-baseline, "input_columns": ";".join(columns)})
        print("Permutation:", name, flush=True)
    raw = pd.DataFrame(rows)
    summary = raw.groupby("group", sort=False).agg(
        MAE_increase_mean_days=("MAE_increase_days", "mean"),
        MAE_increase_SD_days=("MAE_increase_days", "std"),
        permuted_MAE_mean_days=("permuted_MAE_days", "mean"),
        sample_n=("sample_n", "first"), repeats=("repeat", "count"),
        baseline_MAE_days=("baseline_MAE_days", "first"), input_columns=("input_columns", "first"))
    summary = summary.sort_values("MAE_increase_mean_days", ascending=False).reset_index()
    raw.to_csv(destination / "hgb_grouped_permutation_repeats.csv", index=False)
    summary.to_csv(destination / "hgb_grouped_permutation_importance.csv", index=False)
    return summary, {"method": "end-to-end grouped permutation, MAE increase in days",
        "test_n": len(test), "sample_n": len(sample), "seed": seed, "repeats": repeats,
        "baseline_subset_MAE_days": baseline, "subset_ids_sha256": ext.ids_sha(sample.order_id),
        "groups": GROUPS, "risk_recomputed_after_permutation": True,
        "derived_calendar_and_interactions_recomputed": True,
        "refitting_or_selection_performed": False, "causal_interpretation": False,
        "repeat_SD_is_confidence_interval": False, "increments_additive": False}


def plot_comparison(table, importance, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    selected = table.loc[table.model.isin(["Training mean", "Training median", "OLS baseline",
        "Single-tree baseline", "Ridge F1", "RF F1", "RF log-target F1",
        "RF + time/risk", "HGB + time/risk (31 leaves)", "Final HGB (63 leaves)"])].copy()
    positions = np.arange(len(selected))
    for offset, column, label, color in [(-.25,"Train_MAE_days","Train (fit design)","#86b7c8"),
        (0,"CV_MAE_mean_days","5-fold CV","#34718c"),(.25,"Test_MAE_days","Test","#e4a750")]:
        ax.barh(positions+offset, selected[column], height=.24, label=label, color=color)
    ax.set_yticks(positions, selected.model)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("MAE (days); lower is better")
    ax.set_title("Matched random split: training, cross-validation and test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(destination / "train_cv_test_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(importance.group, importance.MAE_increase_mean_days,
        xerr=importance.MAE_increase_SD_days, color="#34718c", capsize=3)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Increase in MAE after shuffling (days); mean ± repeat SD")
    ax.set_title("Final HGB: end-to-end grouped permutation importance\n8,000 fixed test orders; 5 repeats; diagnostic only")
    fig.tight_layout()
    fig.savefig(destination / "hgb_grouped_permutation_importance.png", dpi=160)
    plt.close(fig)


def error_diagnostics(train, test, prediction, raw_orders, destination):
    """Fixed-model test diagnostics and evaluation-only chronology sensitivity."""
    destination = Path(destination)
    target = test.lead_time_days.to_numpy()
    scored = test.assign(predicted_days=prediction)
    groups = pd.cut(scored.lead_time_days, [-np.inf,7,14,30,60,np.inf],
        labels=["0–7 days","7–14 days","14–30 days","30–60 days",">60 days"], right=True)
    rows = []
    for dimension, labels in [("actual duration",groups),("route",scored.route_group)]:
        for label, part in scored.groupby(labels,observed=True):
            rows.append({"dimension":dimension,"group":str(label),
                **metrics(part.lead_time_days,part.predicted_days)})
    pd.DataFrame(rows).to_csv(destination/"hgb_test_error_groups.csv",index=False)
    dates = raw_orders.copy()
    for column in ['order_delivered_customer_date','order_approved_at','order_delivered_carrier_date']:
        dates[column] = pd.to_datetime(dates[column],errors='coerce')
    flagged = dates.loc[(dates.order_delivered_customer_date < dates.order_approved_at) |
        (dates.order_delivered_customer_date < dates.order_delivered_carrier_date),'order_id']
    train_flags,test_flags = train.order_id.isin(flagged),test.order_id.isin(flagged)
    assert int(train_flags.sum()+test_flags.sum())==84
    sensitivity=[]
    for label,mask in [('All fixed test orders',np.ones(len(test),bool)),
        ('Excluding chronology flags at evaluation only',~test_flags.to_numpy()),
        ('Chronology-flagged test orders only',test_flags.to_numpy())]:
        sensitivity.append({'evaluation_cohort':label,**metrics(target[mask],np.asarray(prediction)[mask])})
    pd.DataFrame(sensitivity).to_csv(destination/'chronology_evaluation_sensitivity.csv',index=False)
    cohort=[]
    for role,frame in [('train',train),('test',test)]:
        cohort.append({'split':role,'n':len(frame),'purchase_min':str(frame.purchase_ts.min()),
            'purchase_max':str(frame.purchase_ts.max()),'mean_days':float(frame.lead_time_days.mean()),
            'median_days':float(frame.lead_time_days.median()),'above_30_days_n':int(frame.lead_time_days.gt(30).sum()),
            'above_60_days_n':int(frame.lead_time_days.gt(60).sum()),
            'chronology_flag_n':int(frame.order_id.isin(flagged).sum())})
    pd.DataFrame(cohort).to_csv(destination/'split_population.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    plot_limit=max(float(np.max(target)),float(np.max(prediction)))+5
    axes[0].hexbin(target,prediction,gridsize=55,bins='log',mincnt=1,cmap='Blues')
    axes[0].plot([0,plot_limit],[0,plot_limit],'--',color='#b6583b',linewidth=1)
    axes[0].set(xlim=(0,plot_limit),ylim=(0,plot_limit),xlabel='Observed duration (days)',
        ylabel='Predicted duration (days)',title='a. All test orders; no tail cropping')
    errors=np.asarray(prediction)-target
    axes[1].hist(errors,bins=100,color='#34718c')
    axes[1].axvline(0,color='#b6583b',linestyle='--')
    axes[1].set(xlabel='Predicted − observed duration (days)',ylabel='Orders',
        title=f'b. Residual distribution; mean {errors.mean():.3f} days')
    fig.suptitle('Final HGB (63 leaves): 31,836 fixed random test orders')
    fig.tight_layout()
    figures=destination.parent/'figures'
    figures.mkdir(exist_ok=True)
    fig.savefig(figures/'hgb_test_errors.png',dpi=160)
    plt.close(fig)
    return {'chronology_flags_total':84,'train_flags':int(train_flags.sum()),'test_flags':int(test_flags.sum()),
        'exclusion_scope':'evaluation only; model and training population unchanged',
        'underprediction_share_pct':float(100*(errors<0).mean()),
        'p90_absolute_error_days':float(np.quantile(np.abs(errors),.9))}
