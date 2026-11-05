"""Export the team's fitted complete classification pipelines without fitting.

Run in .venv-classification-export; runtime only needs dashboard NumPy/pandas.
Original model/forest/manifest/output bytes are read and never replaced. The
ignored prepared feature table must be the verified course-source reconstruction
produced by build_classification_demo.py --prepare-only (or pass --csv-dir).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))
sys.path.insert(0, str(PHASE))

from classification import config, search  # noqa: E402
from deployment.classification_pipeline_runtime import load_pipeline, _category  # noqa: E402

OLD = PHASE / "deployment/assets/classification"
DEST = PHASE / "deployment/assets/classification_pipeline"
EVENTS = ["order_approved_at", "order_delivered_carrier_date"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")


def export_history(history, path):
    """Preserve the original pandas sort, cumsum rounding and duplicate ordering."""
    arrays, details = {}, {}
    specs = [("sd", "seller_id", history.seller_delivery_events_, {"delivery_days": "days", "is_late": "late"}),
             ("sr", "seller_id", history.seller_review_events_, {"is_low_review": "low"}),
             ("rd", "route", history.route_delivery_events_, {"delivery_days": "days", "is_late": "late"})]
    for prefix, key, events, sums in specs:
        ev = events.dropna(subset=["known_time"]).sort_values("known_time").copy()
        ev["_count"] = ev.groupby(key).cumcount() + 1
        for field, suffix in sums.items():
            ev["_" + suffix] = ev.groupby(key)[field].cumsum()
        if not ev.known_time.lt(pd.Timestamp(history.event_cutoff)).all():
            raise ValueError("Fitted history contains an event at/after the model cutoff")
        grouped = list(ev.groupby(key, sort=True))
        arrays[prefix + "_keys"] = np.asarray([str(k) for k, _ in grouped], dtype=str)
        arrays[prefix + "_offsets"] = np.r_[0, np.cumsum([len(g) for _, g in grouped])].astype(np.int64)
        ordered = pd.concat([g for _, g in grouped], ignore_index=True)
        arrays[prefix + "_times"] = ordered.known_time.astype("datetime64[ns]").astype("int64").to_numpy()
        arrays[prefix + "_counts"] = ordered._count.to_numpy(dtype=np.int64)
        for suffix in sums.values():
            arrays[prefix + "_" + suffix] = ordered["_" + suffix].to_numpy(dtype=np.float64)
        details[prefix] = {"keys": len(grouped), "events": len(ordered),
                           "latest_known_time": str(ev.known_time.max())}
    np.savez_compressed(path, **arrays)
    return details


def export_parameters(pipeline, path):
    history, pre = pipeline.named_steps["history"], pipeline.named_steps["pre"]
    if list(pipeline.named_steps) != ["history", "pre", "model"]:
        raise ValueError("Unexpected pipeline stages; refusing an approximate conversion")
    numeric, low, high = [pre.named_transformers_[x] for x in ("num", "low", "high")]
    columns = {name: list(cols) for name, _, cols in pre.transformers_}
    if (type(numeric).__name__ != "SimpleImputer" or numeric.strategy != "median"
            or not numeric.add_indicator or numeric.keep_empty_features
            or type(low).__name__ != "OneHotEncoder" or low.handle_unknown != "ignore"
            or low.drop is not None or type(high).__name__ != "TargetEncoder"
            or high.target_type_ != "binary"):
        raise ValueError("The fitted preprocessing does not match the supported exact export")
    params = {
        "numeric": {"columns": columns["num"], "statistics": numeric.statistics_.tolist(),
                    "indicator_features": numeric.indicator_.features_.tolist(),
                    "strategy": "median", "add_indicator": True},
        "low": [], "high": [],
        "history": {"event_cutoff": str(history.event_cutoff), "m": history.m,
                    "rare_min_count": history.rare_min_count,
                    "frequent_sellers": sorted(map(str, history.frequent_sellers_)),
                    "g_delivery": history.g_delivery_, "g_late": history.g_late_, "g_low": history.g_low_},
    }
    sources = columns["num"] + [columns["num"][i] for i in numeric.indicator_.features_]
    for index, column in enumerate(columns["low"]):
        mapping = low._default_to_infrequent_mappings[index]
        mapping = np.arange(len(low.categories_[index])) if mapping is None else mapping
        n = int(low._n_features_outs[index])
        params["low"].append({"column": column, "categories": [_category(x) for x in low.categories_[index]],
                              "mapping": mapping.tolist(), "n_outputs": n,
                              "unknown": "all-zero", "min_frequency": low.min_frequency})
        sources.extend([column] * n)
    for index, column in enumerate(columns["high"]):
        params["high"].append({"column": column, "categories": [_category(x) for x in high.categories_[index]],
                               "encodings": high.encodings_[index].tolist(), "target_mean": float(high.target_mean_),
                               "unknown": "fitted target mean", "smooth": high.smooth})
        sources.append(column)
    params.update(feature_names=pre.get_feature_names_out().tolist(), source_features=sources,
                  feature_groups=[config.GROUP_OF[column] for column in sources])
    if len(sources) != 115 or len(params["feature_names"]) != 115:
        raise ValueError("Unexpected transformed feature count")
    write_json(path, params)
    return params


def reference_input(rows):
    """Fit-only columns are neutral placeholders and are not consumed by transform."""
    frame = rows.copy()
    # The trusted fitted history stores pandas 3 StringDtype keys. Recreate that
    # dtype for merge_asof; the portable runtime itself does not depend on it.
    for field in ("seller_id", "seller_state", "customer_state"):
        frame[field] = frame[field].astype(pd.StringDtype(na_value=np.nan))
    for field in ("order_delivered_customer_date", "review_answer_timestamp"):
        frame[field] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for field in ("delivery_days", "is_on_time", "is_low_review"):
        frame[field] = np.nan
    # Old sklearn/pandas fitted objects used microsecond datetimes. Both sides of
    # merge_asof must use the same resolution; event bytes themselves are unchanged.
    frame.order_purchase_timestamp = pd.to_datetime(frame.order_purchase_timestamp).astype("datetime64[us]")
    return frame[search.INPUT_COLUMNS]


def verify_reference(runtime, fitted, rows, *, already_asof, expected_matrix=None):
    normalized = runtime.validate(rows, already_asof=already_asof)
    ref = reference_input(normalized)
    reference = fitted.named_steps["pre"].transform(fitted.named_steps["history"].transform(ref)).astype(np.float32)
    actual = runtime.transform(rows, already_asof=already_asof)
    if not np.array_equal(reference, actual):
        diff = np.abs(reference.astype(float) - actual.astype(float))
        at = np.unravel_index(np.argmax(diff), diff.shape)
        raise ValueError(f"Complete preprocessing mismatch: {float(diff[at])} at {at}, {runtime.feature_names[at[1]]}")
    if expected_matrix is not None and not np.array_equal(actual, expected_matrix):
        raise ValueError("Exported preprocessing differs from the existing verified demonstration matrix")
    p_expected = fitted.predict_proba(ref)
    p_actual = runtime.predict_proba(rows, already_asof=already_asof)
    error = float(np.max(np.abs(p_expected - p_actual)))
    if error > 1e-12:
        raise ValueError(f"Full pipeline probabilities differ: {error}")
    return {"rows": len(rows), "matrix_exact_float32": True, "max_probability_abs_diff": error}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path)
    parser.add_argument("--prepared", type=Path, default=PHASE / "data/deployment/classification/prepared_features.parquet")
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args(argv)
    old_manifest = json.loads((OLD / "manifest.json").read_text())
    old_protected = {p: sha(p) for folder in (PHASE / "src", PHASE / "outputs", PHASE / "models", OLD)
                     for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if args.csv_dir is not None:
        from deployment.build_classification_demo import prepare_features
        frame, _, provenance = prepare_features(csv_dir=args.csv_dir)
        input_source = "All nine source CSV byte fingerprints verified and features reconstructed by the original source functions; no fitting"
    else:
        frame = pd.read_parquet(args.prepared)
        verification_path = args.prepared.with_name("prepare_verification.json")
        provenance = json.loads(verification_path.read_text())
        input_source = "Existing prepared source reconstruction; full strict feature-value fingerprint reverified"
        # Reconfirm the frozen complete strict feature-value fingerprint rather
        # than trusting the existence or row count of this ignored local cache.
        strict = frame.copy()
        for event, affected in (("order_approved_at", ["approval_lag_hours", "carrier_handover_days"]),
                                ("order_delivered_carrier_date", ["carrier_handover_days", "carrier_transit_days"])):
            strict.loc[strict[event] > strict[config.TIME_COLUMN], affected] = np.nan
        cols = search.INPUT_COLUMNS + [config.TARGET, config.TIME_COLUMN, config.LABEL_TIME_COLUMN]
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(strict[cols], index=False).to_numpy().tobytes()).hexdigest()
        if fingerprint != old_manifest["provenance"]["strict_feature_data_sha256"]:
            raise ValueError("Prepared complete strict features do not match the original frozen fingerprint")
    if not frame.order_id.is_unique:
        raise ValueError("Source rows must have unique order IDs")
    dest = args.dest.resolve()
    if dest == OLD.resolve() or not dest.is_relative_to(PHASE / "deployment/assets"):
        raise ValueError("Write only to a new deployment asset subdirectory")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "direct_features": old_manifest["display_feature_columns"], "metadata_columns": old_manifest["metadata_columns"],
        "source_event_columns": EVENTS, "input_columns": old_manifest["display_feature_columns"] + old_manifest["metadata_columns"] + EVENTS,
        "text_columns": ["order_id", "seller_id", "t_month", *[c for c in config.CATEGORICAL_FEATURES if c != "seller_id_enc"]],
        "region_by_state": config.REGION_BY_STATE, "br_holidays": config.BR_HOLIDAYS,
        "versions": {}, "files": {}, "template_inputs": "new_input_template.csv",
        "history_scope": "Only outcomes stored inside the original fitted pipeline; strict purchase-asof lookup and original fitted priors. No outcomes from new inputs update this frozen history.",
        "seller_count_contract": "seller_prior_orders and seller_tenure_days must be supplied explicitly from operational purchase-time data. The exporter does not fabricate a complete post-training orders ledger.",
        "source_event_contract": "Public rows require approval/carrier source timestamps whenever dependent derived fields are populated; strict R3 masks derived values for source events after T. Current review labels and post-T delivered timestamps are rejected.",
        "editor_recalculation": "Only an explicit action recalculates calendar/price/state/source-event/delivery-at-T derived fields. Shipping-limit duration, city match, product aggregates, distance, seller counts and tenure remain caller supplied.",
        "working_day_calendar": "Original fitted 2016–2018 holiday list; automatic working-day recalculation outside those years is refused.",
        "old_demo_manifest_sha256": sha(OLD / "manifest.json"),
        "export_environment": {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "joblib")}},
        "provenance": {"strict_feature_data_sha256": old_manifest["provenance"]["strict_feature_data_sha256"],
                       "course_sources": provenance["course_sources"], "source_sha256": provenance["source_sha256"],
                       "input_source": input_source,
                       "exporter_sha256": sha(Path(__file__)),
                       "portable_pipeline_runtime_sha256": sha(PHASE / "deployment/classification_pipeline_runtime.py")},
        "verification": {},
    }
    fitted_models = {}
    for version, old in old_manifest["versions"].items():
        source = PHASE / old["source_model"]
        if sha(source) != old["model_sha256"]:
            raise ValueError("Trusted fitted bundle fingerprint changed: " + version)
        fitted = joblib.load(source)["model"]
        fitted_models[version] = fitted
        params_file, history_file = version + "_parameters.json", version + "_history.npz"
        params = export_parameters(fitted, dest / params_file)
        history_details = export_history(fitted.named_steps["history"], dest / history_file)
        original = pd.read_parquet(OLD / old["inputs"])
        event_rows = frame.set_index("order_id").loc[original.order_id, EVENTS].reset_index(drop=True)
        examples = original.reset_index(drop=True)
        for field in EVENTS:
            examples[field] = event_rows[field].to_numpy()
        example_file = version + "_example_inputs.parquet"
        examples[manifest["input_columns"]].to_parquet(dest / example_file, index=False)
        manifest["versions"][version] = {
            "parameters": params_file, "history": history_file, "example_inputs": example_file,
            "forest": "../classification/" + old["forest"], "forest_sha256": old_manifest["files"][old["forest"]],
            "source_model_sha256": old["model_sha256"], "selection_sha256": old["selection_sha256"],
            "capacity_threshold": old["capacity_threshold"], "f1_threshold": old["f1_threshold"],
            "classes": old["classes"], "n_features": old["n_features"], "history_counts": history_details,
            "event_cutoff": params["history"]["event_cutoff"],
        }
        for name in (params_file, history_file, example_file):
            manifest["files"][name] = sha(dest / name)
    # A blank/current-outcome-free example: explicitly illustrative unknown seller
    # and zero prior orders, rather than pretending to query a live seller ledger.
    template = pd.read_parquet(dest / "strict_example_inputs.parquet").iloc[[0]].copy()
    template.order_id = "new-order-example-001"
    template.seller_id = "NEW_SELLER"
    template.seller_prior_orders = 0.
    template.seller_tenure_days = 0.
    template.to_csv(dest / manifest["template_inputs"], index=False)
    manifest["files"][manifest["template_inputs"]] = sha(dest / manifest["template_inputs"])
    write_json(dest / "manifest.json", manifest)
    verifications = {}
    for version, fitted in fitted_models.items():
        runtime = load_pipeline(dest, version)
        examples = runtime.example_inputs()
        checks = {"historical_120": verify_reference(runtime, fitted, examples, already_asof=True,
                  expected_matrix=np.load(OLD / old_manifest["versions"][version]["matrix"], allow_pickle=False))}
        safe = runtime.example_inputs(public_valid_only=True)
        manifest["versions"][version]["public_example_rows"] = len(safe)
        new = runtime.recalculate_editor(runtime.new_input_template())
        if version == "strict":
            checks["historical_public_contract"] = verify_reference(runtime, fitted, safe, already_asof=False)
            rejected = []
            for position in range(len(examples)):
                try:
                    runtime.validate(examples.iloc[[position]])
                except ValueError as error:
                    rejected.append({"order_id": str(examples.order_id.iloc[position]), "reason": str(error)})
            checks["historical_public_rejections"] = {"rows": len(rejected), "orders": rejected,
                "scope": "Source-event sequence anomalies excluded only from new-order editor references; all 120 preserved for exact internal historical replay"}
        checks["new_order"] = verify_reference(runtime, fitted, new, already_asof=False)
        variants = new.copy()
        variants.loc[0, "main_category"] = "UNSEEN_CATEGORY"
        variants.loc[0, "customer_zip2"] = "00"
        variants.loc[0, "main_payment_type"] = "UNSEEN_PAYMENT"
        for name in ("max_weight_g", "total_volume_cm3", "max_distance_km"):
            variants.loc[0, name] = np.nan
        checks["unknown_categories_missing_numeric"] = verify_reference(runtime, fitted, variants, already_asof=False)
        variants.loc[0, "main_category"] = np.nan
        checks["learned_missing_category"] = verify_reference(runtime, fitted, variants, already_asof=False)
        batch = pd.concat([new, variants.assign(order_id="new-order-example-002")], ignore_index=True)
        checks["two_new_orders"] = verify_reference(runtime, fitted, batch, already_asof=False)
        if not np.array_equal(runtime.predict_proba(batch), np.vstack([runtime.predict_proba(batch.iloc[[i]]) for i in range(len(batch))])):
            raise ValueError("Single/batch new-order inference differs")
        verifications[version] = checks
        print(version, json.dumps(checks), flush=True)
    # Persist a public-contract template, including any explicit editor corrections.
    template = load_pipeline(dest, "strict").recalculate_editor(template)
    load_pipeline(dest, "strict").validate(template)
    template.to_csv(dest / manifest["template_inputs"], index=False)
    manifest["files"][manifest["template_inputs"]] = sha(dest / manifest["template_inputs"])
    unchanged = all(sha(path) == original_hash for path, original_hash in old_protected.items())
    if not unchanged:
        raise ValueError("Protected original source/model/output/demo bytes changed during export")
    manifest["verification"] = {"no_fitting": True, "protected_original_bytes_unchanged": unchanged,
                                "historical_matrix_exact": True, "versions": verifications}
    write_json(dest / "manifest.json", manifest)
    print(f"Wrote exact inference-only pipelines to {dest}", flush=True)


if __name__ == "__main__":
    main()
