"""Verify the published regression data without fitting any estimator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regression import audit, config as C, data


def verify(archive=None):
    folder = ROOT / "data" / "regression"
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, spec in manifest["files"].items():
        path = folder / name
        assert path.stat().st_size == spec["size_bytes"], name + ": size differs"
        assert data.digest(path) == spec["sha256"], name + ": fingerprint differs"
    gate = audit.require_pass()
    assert gate["manifest_sha256"] == data.digest(folder / "fold_manifest.csv.gz")
    orders = data.load()
    valid = pd.read_pickle(folder / "all_valid.pkl")
    assert len(orders) == 96203 and len(valid) == 96470
    assert orders.columns.tolist() == manifest["files"]["orders.pkl"]["columns"]
    assert valid.columns.tolist() == manifest["files"]["all_valid.pkl"]["columns"]
    for frame in (orders, valid):
        assert frame.order_id.notna().all() and frame.order_id.is_unique
        assert frame.order_status.eq("delivered").all()
        assert frame.lead_time_days.notna().all() and frame.lead_time_days.ge(0).all()
        actual = (frame.delivered_ts - frame.purchase_ts).dt.total_seconds() / 86400
        np.testing.assert_array_equal(actual.to_numpy(), frame.lead_time_days.to_numpy())
    subset = valid.loc[valid.purchase_ts.ge(C.START) & valid.purchase_ts.lt(C.TEST_END)].reset_index(drop=True)
    pd.testing.assert_frame_equal(subset, orders.drop(columns="chronology_flag"), check_exact=True)
    assert int(valid.lead_time_days.gt(60).sum()) == 306
    common_num, common_cat = C.columns("calendar")
    assert len(common_num + common_cat) == 39
    assert set(common_num + common_cat) <= set(orders.columns)
    assert not set(C.FORBIDDEN) & set(common_num + common_cat)
    assert not {"risk_p30", "risk_unavailable"} & set(orders.columns)

    membership = pd.read_csv(folder / "fold_manifest.csv.gz", parse_dates=["purchase_ts", "delivered_ts"])
    assert len(membership) == 413680
    assert not membership.duplicated(["fold", "role", "order_id"]).any()
    names = ["fold1", "fold2", "fold3", "fold4", "fold5", "final"]
    assert set(membership.fold) == set(names)
    assert set(membership.role) == {"train", "score", "full_window"}
    summaries = []
    test_ids = set(orders.loc[orders.purchase_ts.ge(C.TEST_START) & orders.purchase_ts.lt(C.TEST_END), "order_id"])
    for name, month in zip(names, C.MONTHS + ["2018-06"]):
        start = pd.Timestamp(month + "-01")
        end = C.TEST_END if name == "final" else start + pd.offsets.MonthBegin(1)
        train = orders.purchase_ts.ge(C.START) & orders.purchase_ts.lt(start - pd.Timedelta(days=39)) & orders.delivered_ts.lt(start)
        full = orders.purchase_ts.ge(start) & orders.purchase_ts.lt(end)
        score = full if name == "final" else full & orders.delivered_ts.lt(C.SELECTION_CUTOFF)
        for role, mask in (("train", train), ("score", score), ("full_window", full)):
            expected = orders.loc[mask, ["order_id", "purchase_ts", "delivered_ts"]].reset_index(drop=True)
            actual = membership.loc[membership.fold.eq(name) & membership.role.eq(role), expected.columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        assert not set(orders.loc[train, "order_id"]) & set(orders.loc[score, "order_id"])
        if name != "final":
            assert not (set(orders.loc[train, "order_id"]) | set(orders.loc[full, "order_id"])) & test_ids
        summaries.append({"fold": name, "train_n": int(train.sum()), "score_n": int(score.sum()), "full_window_n": int(full.sum())})
    published = pd.read_csv(ROOT / "outputs/regression/tables/fold_table.csv").set_index("fold")
    for row in summaries:
        for key, original in (("train_n", "train_n"), ("score_n", "score_n"), ("full_window_n", "window_completed_n")):
            assert row[key] == int(published.loc[row["fold"], original])
    lock = json.loads((ROOT / "outputs/regression/test_lock.json").read_text())
    assert len(test_ids) == lock["n"] == 18603
    assert data.ids_hash(test_ids) == lock["ids_sha256"]

    if archive:
        import os
        from unittest.mock import patch
        assert data.digest(archive) == C.ARCHIVE_SHA
        with tempfile.TemporaryDirectory(prefix="it5006-regression-data-verify-") as scratch:
            scratch = Path(scratch)
            with patch.dict(os.environ, {"OLIST_ARCHIVE": str(Path(archive).resolve()), "OLIST_CSV_DIR": "", "REGRESSION_V2_REFERENCE": ""}), \
                 patch.object(C, "OUT", scratch / "outputs"), patch.object(C, "TABLES", scratch / "outputs/tables"), \
                 patch.object(C, "FIGURES", scratch / "figures"), patch.object(C, "PRIVATE", scratch / "data"), \
                 patch.object(C, "MODELS", scratch / "models"):
                rebuilt, raw = data.prepare()
                original = raw["orders"]
                delivered = pd.to_datetime(original.order_delivered_customer_date)
                approved = pd.to_datetime(original.order_approved_at)
                carrier = pd.to_datetime(original.order_delivered_carrier_date)
                flagged = set(original.loc[delivered.lt(approved) | delivered.lt(carrier), "order_id"])
                rebuilt["chronology_flag"] = rebuilt.order_id.isin(flagged)
                pd.testing.assert_frame_equal(rebuilt, orders, check_exact=True)
                pd.testing.assert_frame_equal(pd.read_pickle(C.PRIVATE / "all_valid.pkl"), valid, check_exact=True)
                original_inventory = pd.read_csv(ROOT / "outputs/regression/tables/source_inventory.csv")
                rebuilt_inventory = pd.read_csv(C.TABLES / "source_inventory.csv")
                pd.testing.assert_frame_equal(rebuilt_inventory, original_inventory, check_exact=True)
    return {"passed": True, "cohort_orders": len(orders), "all_valid_orders": len(valid),
            "membership_rows": len(membership), "test_orders": len(test_ids), "folds": summaries,
            "original_zip_rebuild_verified": bool(archive), "model_fits": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optionally rebuild from the original course ZIP in a temporary directory")
    args = parser.parse_args()
    print(json.dumps(verify(args.archive), indent=2))
