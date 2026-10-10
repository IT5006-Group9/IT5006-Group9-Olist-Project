"""Independent audit of source/cohort/purchase-time features; no model fitting.

Candidate inputs are a preview, never overwrite the trained development data.
The only geographic reference used for coordinates is the course archive.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/preparation_audit"
PRIVATE = ROOT / "data/preparation_audit"
FILES = {
    "orders":"olist_orders_dataset.csv", "items":"olist_order_items_dataset.csv",
    "customers":"olist_customers_dataset.csv", "sellers":"olist_sellers_dataset.csv",
    "products":"olist_products_dataset.csv", "geo":"olist_geolocation_dataset.csv",
    "translation":"product_category_name_translation.csv",
}


def read_course():
    archive = ROOT.parent / "01_当前小组项目/课程配套资料/IT5006_Project-Data.zip"
    tables, inventory = {}, []
    with zipfile.ZipFile(archive) as z:
        for key, name in FILES.items():
            content = z.read("Olist_CSV/" + name)
            tables[key] = pd.read_csv(BytesIO(content))
            inventory.append({"table":key,"rows":len(tables[key]),"columns":len(tables[key].columns),
                              "sha256":hashlib.sha256(content).hexdigest()})
    return tables, pd.DataFrame(inventory)


def geographic_reference(geo, policy):
    """No outcomes used: conservative country envelope and robust postal proxy.

    [-35,6] latitude / [-75,-25] longitude is a generous screening envelope
    around IBGE's national extremes, including Atlantic islands. It is NOT
    a polygon, state boundary validator, or a newly downloaded coordinate source.
    Inside-envelope anomalies can remain. Exact duplicate reference coordinates
    need not contribute repeated weight; deduplicate in the median candidate.
    """
    g = geo.copy()
    coordinates = ["geolocation_lat", "geolocation_lng"]
    if policy != "legacy_mean":
        inside = g.geolocation_lat.between(-35, 6) & g.geolocation_lng.between(-75, -25)
        g = g.loc[inside]
    if policy in ["screened_median", "screened_state_median"]:
        g = g.drop_duplicates(["geolocation_zip_code_prefix", "geolocation_state", *coordinates])
    keys = ["geolocation_zip_code_prefix"]
    if policy == "screened_state_median":
        keys.append("geolocation_state")
    grouped = g.groupby(keys)[coordinates]
    return grouped.mean() if policy in ["legacy_mean", "screened_mean"] else grouped.median()


def compute_routes(items, orders, customers, sellers, reference, state_key=False):
    route = (items[["order_id", "order_item_id", "seller_id"]]
        .merge(orders[["order_id", "customer_id"]], on="order_id", how="left", validate="many_to_one")
        .merge(customers[["customer_id", "customer_zip_code_prefix", "customer_state"]],
               on="customer_id", how="left", validate="many_to_one")
        .merge(sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]],
               on="seller_id", how="left", validate="many_to_one"))
    assert len(route) == len(items)
    for side in ["customer", "seller"]:
        join_key = [side + "_zip_code_prefix", side + "_state"] if state_key else side + "_zip_code_prefix"
        route = route.merge(reference.rename(columns={"geolocation_lat":side+"_lat",
                            "geolocation_lng":side+"_lng"}), left_on=join_key, right_index=True,
                            how="left", validate="many_to_one")
    a, b, c, d = [np.radians(route[col]) for col in ["customer_lat", "customer_lng", "seller_lat", "seller_lng"]]
    h = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    route["distance_km"] = (6371*2*np.arcsin(np.sqrt(h.clip(0, 1)))).round(1)
    group = route.groupby("order_id").distance_km
    result = group.max().to_frame("max_distance_km")
    result["any_distance_missing"] = group.count().lt(group.size()).astype(int)
    return route, result


def run_audit():
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    tables, inventory = read_course()
    inventory.to_csv(OUT / "source_inventory.csv", index=False)
    old_inventory = pd.read_csv(ROOT / "outputs/tables/source_inventory.csv")
    assert inventory.set_index("table").sha256.equals(old_inventory.set_index("table").sha256)
    orders, items, customers, sellers, products, geo, translation = [tables[k] for k in FILES]
    dates = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date",
             "order_approved_at", "order_delivered_carrier_date"]
    for field in dates:
        orders[field] = pd.to_datetime(orders[field], errors="raise")

    integrity = []
    keys = {"orders":["order_id"], "items":["order_id","order_item_id"],
            "customers":["customer_id"], "sellers":["seller_id"],
            "products":["product_id"], "translation":["product_category_name"]}
    for table, key in keys.items():
        x = tables[table]
        count = int(x.duplicated(key).sum())
        null = int(x[key].isna().any(axis=1).sum())
        integrity.append({"check":table+" primary key", "affected_rows":count+null})
        assert count == 0 and null == 0
    for child, field, parent in [(items,"order_id",orders),(items,"product_id",products),
                                 (items,"seller_id",sellers),(orders,"customer_id",customers)]:
        n = int((~child[field].isin(parent[field])).sum())
        integrity.append({"check":field+" FK coverage", "affected_rows":n})
        assert n == 0
    pd.DataFrame(integrity).to_csv(OUT / "integrity_checks.csv", index=False)

    # Define the cohort independently, starting with all orders rather than item rows.
    target = (orders.order_delivered_customer_date-orders.order_purchase_timestamp).dt.total_seconds()/86400
    with_items = orders.order_id.isin(items.order_id)
    delivered = orders.order_status.eq("delivered")
    observed = orders.order_purchase_timestamp.notna() & orders.order_delivered_customer_date.notna()
    non_negative = target.ge(0)
    eligible_mask = delivered & observed & non_negative & with_items
    cohort = orders.loc[eligible_mask, ["order_id", *dates]].copy()
    cohort["lead_time_days"] = target.loc[eligible_mask].values
    original = pd.read_csv(ROOT / "data/eligible_orders.csv").set_index("order_id")
    assert set(cohort.order_id) == set(original.index)
    np.testing.assert_allclose(cohort.set_index("order_id").lead_time_days,
                               original.lead_time_days.reindex(cohort.order_id), rtol=1e-10)
    status = orders.assign(has_item_records=with_items).groupby(["order_status","has_item_records"]).size().rename("n").reset_index()
    status.to_csv(OUT / "status_item_coverage.csv", index=False)
    flow = pd.DataFrame([
        {"stage":"all source orders", "n":len(orders)},
        {"stage":"delivered status", "n":int(delivered.sum())},
        {"stage":"delivered with purchase/receipt", "n":int((delivered&observed).sum())},
        {"stage":"non-negative observed target", "n":int((delivered&observed&non_negative).sum())},
        {"stage":"also has item records", "n":int(eligible_mask.sum())},
    ])
    flow.to_csv(OUT / "cohort_flow.csv", index=False)
    assert len(cohort) == 96470
    assert int((~with_items & delivered).sum()) == 0

    cohort["receipt_before_approval"] = cohort.order_delivered_customer_date.lt(cohort.order_approved_at)
    cohort["receipt_before_carrier"] = cohort.order_delivered_customer_date.lt(cohort.order_delivered_carrier_date)
    flags = cohort[["order_id","receipt_before_approval","receipt_before_carrier"]]
    flags.loc[flags.drop(columns="order_id").any(axis=1)].to_csv(PRIVATE / "event_chronology_flags.csv", index=False)
    chronology = pd.DataFrame([
        {"check":"receipt before purchase", "n":int((target<0).sum())},
        {"check":"eligible receipt before approval", "n":int(cohort.receipt_before_approval.sum())},
        {"check":"eligible receipt before carrier", "n":int(cohort.receipt_before_carrier.sum())},
        {"check":"eligible union event contradictions", "n":int(flags.drop(columns="order_id").any(axis=1).sum())},
        {"check":"non-delivered status with receipt", "n":int((~delivered&orders.order_delivered_customer_date.notna()).sum())},
    ])
    chronology.to_csv(OUT / "chronology_checks.csv", index=False)

    # Reconstruct all non-geographic purchase inputs directly from the source tables.
    f = (items.merge(products[["product_id","product_category_name","product_weight_g"]],
                     on="product_id", how="left", validate="many_to_one")
        .merge(translation, on="product_category_name", how="left", validate="many_to_one")
        .merge(orders[["order_id","customer_id"]],on="order_id",how="left",validate="many_to_one")
        .merge(customers[["customer_id","customer_state"]],on="customer_id",how="left",validate="many_to_one")
        .merge(sellers[["seller_id","seller_state"]],on="seller_id",how="left",validate="many_to_one"))
    assert len(f) == len(items)
    assert items.price.notna().all() and items.price.gt(0).all()
    assert items.freight_value.notna().all() and items.freight_value.ge(0).all()
    f["positive_weight"] = f.product_weight_g.where(f.product_weight_g.gt(0))
    f["same_state"] = f.customer_state.eq(f.seller_state)
    group = f.groupby("order_id")
    rec = group.agg(total_price=("price","sum"), total_freight=("freight_value","sum"),
                    item_count=("order_item_id","size"), seller_count=("seller_id","nunique"),
                    avg_product_weight_g=("positive_weight","mean"),
                    category_n=("product_category_name_english","nunique"),
                    category_first=("product_category_name_english","first"),
                    all_same_state=("same_state","all"), customer_state=("customer_state","first"))
    rec["any_weight_missing"] = group.positive_weight.count().lt(group.size()).astype(int)
    rec.loc[rec.any_weight_missing.eq(1),"avg_product_weight_g"] = np.nan
    category_missing = group.product_category_name_english.count().lt(group.size())
    rec["category_group"] = np.select([category_missing,rec.category_n.gt(1)],
        ["Missing/incomplete category","Multiple categories"],default=rec.category_first)
    all_states_known = (f.customer_state.notna() & f.seller_state.notna()).groupby(f.order_id).all()
    rec["route_group"] = np.select([~all_states_known, ~rec.all_same_state],
        ["Unknown","Any seller cross-state"],default="All sellers same-state")
    rec["freight_ratio"] = rec.total_freight/rec.total_price
    order_idx = orders.set_index("order_id")
    rec["promised_lead_time_days"] = (order_idx.order_estimated_delivery_date-order_idx.order_purchase_timestamp).dt.total_seconds()/86400
    rec["purchase_month"] = order_idx.order_purchase_timestamp.dt.month
    rec["purchase_weekday"] = order_idx.order_purchase_timestamp.dt.dayofweek
    rec["purchase_hour"] = order_idx.order_purchase_timestamp.dt.hour
    rec = rec.reindex(original.index)
    numeric_cols = ["total_price","total_freight","item_count","seller_count","avg_product_weight_g",
                    "any_weight_missing","freight_ratio","promised_lead_time_days","purchase_month","purchase_weekday","purchase_hour"]
    for col in numeric_cols:
        np.testing.assert_allclose(rec[col],original[col],rtol=1e-9,atol=1e-9,equal_nan=True)
    for col in ["category_group","route_group","customer_state"]:
        assert rec[col].fillna("NA").tolist() == original[col].fillna("NA").tolist()

    profile = []
    roles = pd.read_csv(ROOT / "config/feature_roles.csv")
    fields = roles.loc[roles.role.eq("input"),"field"].tolist()
    assert len(fields) == 16 and len(set(fields)) == 16
    for field in fields:
        s = original[field]
        row = {"field":field,"n":len(s),"missing_n":int(s.isna().sum()),"missing_pct":s.isna().mean()*100,
               "unique_n":int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            assert np.isfinite(s.dropna()).all(), field
            row.update({"min":s.min(),"median":s.median(),"p99":s.quantile(.99),"max":s.max()})
        profile.append(row)
    pd.DataFrame(profile).to_csv(OUT / "feature_profile.csv",index=False)
    # Reveal missingness without looking at final-test prediction errors.
    development = original.loc[original.split.isin(["train","validation"])]
    development.groupby("split")[["max_distance_km","avg_product_weight_g"]].agg(
        lambda x: int(x.isna().sum())).to_csv(OUT / "development_missingness.csv")

    global_valid = geo.geolocation_lat.between(-90,90) & geo.geolocation_lng.between(-180,180)
    screening_valid = geo.geolocation_lat.between(-35,6) & geo.geolocation_lng.between(-75,-25)
    geo_stats = {"source_rows":len(geo),"exact_duplicate_rows":int(geo.duplicated().sum()),
                 "postal_prefixes":int(geo.geolocation_zip_code_prefix.nunique()),
                 "multi_state_prefixes":int(geo.groupby("geolocation_zip_code_prefix").geolocation_state.nunique().gt(1).sum()),
                 "outside_global_domain":int((~global_valid).sum()),
                 "outside_conservative_brazil_envelope":int((~screening_valid).sum()),
                 "flagged_postal_prefixes":int(geo.loc[~screening_valid,"geolocation_zip_code_prefix"].nunique())}
    geographic_rows, distances = [], {}
    for policy in ["legacy_mean","screened_mean","screened_median","screened_state_median"]:
        ref = geographic_reference(geo,policy)
        route, distance = compute_routes(items,orders,customers,sellers,ref,state_key=policy.endswith("state_median"))
        distance = distance.reindex(original.index)
        distances[policy] = distance
        diff = distance.max_distance_km - original.max_distance_km
        geographic_rows.append({"policy":policy,"eligible_n":len(distance),
            "missing_max_distance_n":int(distance.max_distance_km.isna().sum()),
            "any_distance_missing_n":int(distance.any_distance_missing.sum()),
            "max_observed_distance_km":distance.max_distance_km.max(),
            "above_4500_km_n":int(distance.max_distance_km.gt(4500).sum()),
            "changed_ge_50_km_n":int(diff.abs().ge(50).sum()),
            "median_absolute_change_km":diff.abs().median(),
            "new_missing_n":int((distance.max_distance_km.isna() & original.max_distance_km.notna()).sum())})
        if policy == "legacy_mean":
            np.testing.assert_allclose(distance.max_distance_km, original.max_distance_km,
                                      rtol=1e-9, atol=1e-9,equal_nan=True)
            assert distance.any_distance_missing.equals(original.any_distance_missing)
    geo_table = pd.DataFrame(geographic_rows)
    geo_table.to_csv(OUT / "geography_policy_comparison.csv",index=False)
    selected = distances["screened_median"]
    preview = original[fields].copy()
    preview[["max_distance_km","any_distance_missing"]] = selected[["max_distance_km","any_distance_missing"]]
    preview.reset_index().to_csv(PRIVATE / "screened_median_input_preview.csv",index=False)
    comparison = original[["max_distance_km","any_distance_missing"]].join(selected,rsuffix="_candidate")
    comparison.to_csv(PRIVATE / "distance_revision_preview.csv")

    # Availability is a schema/timing judgement, not proved by a timestamp-less archive.
    outcome_fields = {"lead_time_days","delivered_ts","order_status","review_score","payment_total",
                      "order_approved_at","order_delivered_carrier_date","customer_unique_id","order_id"}
    assert not outcome_fields & set(fields)
    availability = []
    for field in fields:
        if field.startswith("purchase_"):
            status, assumption = "direct from prediction timestamp", "purchase timestamp is available when order is placed"
        elif field == "promised_lead_time_days":
            status, assumption = "conditional archived quote", "estimated date must be the original customer quote; no creation/revision history exists"
        elif field in ["max_distance_km","any_distance_missing"]:
            status, assumption = "static reference assumption", "course coordinates approximate historical addresses; revision history absent"
        elif field in ["avg_product_weight_g","any_weight_missing","category_group"]:
            status, assumption = "static catalogue assumption", "archived product attributes are assumed applicable at purchase"
        else:
            status, assumption = "conditional order snapshot", "archived basket, selected sellers, charges and destination represent the placed order"
        availability.append({"field":field,"availability_evidence":status,"required_assumption":assumption,
                             "uses_outcome_for_value":False})
    pd.DataFrame(availability).to_csv(OUT / "feature_availability.csv",index=False)

    # Audit existing chronological assignments, without fitting or scoring a model.
    full = original.copy()
    for field in ["purchase_ts","delivered_ts"]:
        full[field] = pd.to_datetime(full[field])
    expected = pd.Series("pending_label_train_cutoff",index=full.index)
    expected.loc[full.purchase_ts.lt("2018-03-01") & full.delivered_ts.lt("2018-03-01")] = "train"
    expected.loc[full.purchase_ts.ge("2018-03-01") & full.purchase_ts.lt("2018-04-01")] = "validation_label_pending"
    expected.loc[full.purchase_ts.ge("2018-03-01") & full.purchase_ts.lt("2018-04-01") & full.delivered_ts.lt("2018-07-01")] = "validation"
    expected.loc[full.purchase_ts.ge("2018-04-01") & full.purchase_ts.lt("2018-07-01")] = "maturation_gap"
    expected.loc[full.purchase_ts.ge("2018-07-01")] = "test_locked"
    assert expected.equals(full.split)

    summary = {"audit_date":"2026-10-05", "source_unchanged":True,"eligible_orders":len(cohort),
        "target_matched_all_orders":True,"non_geographic_inputs_matched_all_orders":True,
        "no_completed_order_lost_by_item_requirement":True,
        "event_chronology_flagged_orders":int(flags.drop(columns="order_id").any(axis=1).sum()),
        "retained_valid_over_60_days":int(cohort.lead_time_days.gt(60).sum()),
        "geography":geo_stats,"existing_splits_verified":True,
        "candidate_policy":"screened_median; unchanged cohort, all non-geographic inputs unchanged",
        "candidate_applied_to_training":False,"model_fit_performed":False,"test_scoring_performed":False,
        "status":"cohort accepted; screened-median preparation implemented, model artifacts require refit",
        "production_preparation_corrected":True,
        "formal_v2_materialized":"versions/preparation_v2/data/eligible_orders.csv" if (ROOT / "versions/preparation_v2/data/eligible_orders.csv").exists() else None,
        "geographic_reference":"https://biblioteca.ibge.gov.br/visualizacao/periodicos/20/aeb_2022.pdf; table 1.1.1.1 including island notes",
        "reference_use":"geographic domain rationale only; no external coordinates or new Olist data ingested"}
    (OUT / "audit_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    return {"summary":summary,"sources":inventory,"cohort_flow":flow,"chronology":chronology,
            "feature_profile":pd.DataFrame(profile),"availability":pd.DataFrame(availability),
            "geography":geo_table}


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result["summary"],indent=2))
    print(result["geography"].to_string(index=False))
