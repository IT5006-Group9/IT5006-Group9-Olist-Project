"""Order-level random holdout helpers; no guessed paper parameters or model fitting.

The CLI refuses to export an experimental split until its protocol is verified.
Sorting IDs makes assignments independent of CSV row order. Classification may
reuse a common manifest, or the same procedure on its explicitly different cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def check_parameters(protocol: dict) -> None:
    keys = ("train_fraction", "validation_fraction", "test_fraction")
    if any(protocol.get(key) is None for key in (*keys, "random_seed", "cv_folds")):
        raise ValueError("Split fractions, seed and CV folds require verified method settings; no defaults are assumed.")
    fractions = [protocol[key] for key in keys]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in fractions):
        raise ValueError("Split fractions must be numeric.")
    if not np.isfinite(fractions).all() or not np.isclose(sum(fractions), 1, atol=1e-12, rtol=0):
        raise ValueError("Finite split fractions must sum to one.")
    if fractions[0] <= 0 or fractions[1] < 0 or fractions[2] <= 0:
        raise ValueError("Train and test must be positive; validation can be zero if CV supplies validation.")
    for key in ("random_seed", "cv_folds"):
        if isinstance(protocol[key], bool) or not isinstance(protocol[key], int):
            raise ValueError(f"{key} must be an integer.")
    if not 0 <= protocol["random_seed"] <= 2**32 - 1 or protocol["cv_folds"] < 2:
        raise ValueError("Invalid seed or CV fold count.")
    if protocol.get("regression_stratification") != "none":
        raise ValueError("This helper supports unstratified random regression splitting only; any alternative needs explicit implementation.")


def make_manifest(order_ids, protocol: dict) -> pd.DataFrame:
    """Split unique orders test-first, then validation as a share of the remainder."""
    check_parameters(protocol)
    ids = pd.Series(order_ids, name="order_id")
    if ids.isna().any() or not ids.is_unique or not ids.map(lambda x: isinstance(x, str)).all():
        raise ValueError("One non-null string ID per order is required before splitting.")
    ids = np.array(sorted(ids.tolist()))
    remaining, test = train_test_split(ids, test_size=protocol["test_fraction"],
                                       shuffle=True, random_state=protocol["random_seed"])
    if protocol["validation_fraction"]:
        train, validation = train_test_split(
            remaining,
            test_size=protocol["validation_fraction"] / (1 - protocol["test_fraction"]),
            shuffle=True, random_state=protocol["random_seed"])
    else:
        train, validation = remaining, np.array([], dtype=str)
    manifest = pd.concat([
        pd.DataFrame({"order_id": part, "split": label})
        for label, part in (("train", train), ("validation", validation), ("test", test))
    ], ignore_index=True).sort_values("order_id").reset_index(drop=True)
    if len(manifest) != len(ids) or not manifest.order_id.is_unique:
        raise AssertionError("Every eligible order must occur in exactly one partition.")
    if len(train) < protocol["cv_folds"]:
        raise ValueError("Training population is smaller than the requested fold count.")
    return manifest


def make_cv_manifest(manifest: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    """Each training order appears once as an OOF scoring row; holdouts excluded."""
    check_parameters(protocol)
    if not manifest.order_id.is_unique or manifest.order_id.isna().any():
        raise ValueError("Invalid partition manifest.")
    train_ids = np.array(sorted(manifest.loc[manifest.split.eq("train"), "order_id"]))
    if len(train_ids) < protocol["cv_folds"]:
        raise ValueError("Insufficient training orders for CV.")
    result = []
    folds = KFold(n_splits=protocol["cv_folds"], shuffle=True, random_state=protocol["random_seed"])
    for fold, (fit, score) in enumerate(folds.split(train_ids), 1):
        if set(train_ids[fit]) & set(train_ids[score]):
            raise AssertionError("CV fit/score overlap.")
        result.append(pd.DataFrame({"order_id": train_ids[score], "oof_fold": fold}))
    output = pd.concat(result, ignore_index=True).sort_values("order_id").reset_index(drop=True)
    if len(output) != len(train_ids) or not output.order_id.is_unique:
        raise AssertionError("OOF scoring coverage must be exactly once per training order.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if not protocol.get("reference_method_verified") or not protocol.get("parameter_evidence"):
        raise ValueError("Paper/team adaptation must be documented before experimental split export.")
    check_parameters(protocol)
    source = args.root / protocol["input"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol["input_sha256"]:
        raise ValueError("Prepared input hash differs from the verified V2 data.")
    orders = pd.read_csv(source, usecols=["order_id"])
    manifest = make_manifest(orders.order_id, protocol)
    folds = make_cv_manifest(manifest, protocol)
    destination = args.root / protocol["output_directory"] / "data"
    if destination.exists():
        raise FileExistsError("Use a new experiment directory; existing manifests will not be overwritten.")
    destination.mkdir(parents=True)
    manifest.to_csv(destination / "split_manifest.csv", index=False)
    folds.to_csv(destination / "cv_manifest.csv", index=False)
    (destination / "protocol_snapshot.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(manifest.split.value_counts().to_string())


if __name__ == "__main__":
    main()
