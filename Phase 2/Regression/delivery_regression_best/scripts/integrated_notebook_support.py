"""Execution plumbing for the submission notebook; analytical tables stay visible."""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import redirect_stdout
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import course_notebook_checks as checks
import reproduce_best
from verify_bundle import verify_bundle
from verify_report_supplement import main as verify_report_tables


def start(root, mode="review", with_cv=False):
    root = Path(root)
    if mode not in {"review", "reproduce"}:
        raise ValueError("Mode must be review or reproduce.")
    archive = Path(os.environ["OLIST_ARCHIVE"]).expanduser().resolve() if os.environ.get("OLIST_ARCHIVE") else None
    if archive is not None and not archive.is_file():
        raise FileNotFoundError("OLIST_ARCHIVE does not point to a file.")
    if mode == "reproduce" and archive is None:
        raise ValueError("Reproduce mode requires the course ZIP in OLIST_ARCHIVE.")
    if archive is not None and os.environ.get("OLIST_CSV_DIR"):
        raise ValueError("Clear OLIST_CSV_DIR; this workflow uses the fixed course ZIP.")
    run = root / "runs" / ("integrated_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
    run.mkdir(parents=True, exist_ok=False)
    ctx = dict(root=root, run=run, archive=archive, mode=mode, with_cv=with_cv)
    ctx["thread_limit"] = threadpool_limits(limits=2)
    log = io.StringIO()
    with redirect_stdout(log):
        ctx["manifest_checks"] = verify_bundle(root)
        verify_report_tables()
    (run / "verification.log").write_text(log.getvalue())
    return ctx


def audit_sources(ctx):
    root, archive = ctx["root"], ctx["archive"]
    if archive is not None:
        contracts, inventory, summary = checks.audit_course_archive(archive, ctx["run"] / "source_audit")
    else:
        contracts = pd.read_csv(root / "results/integrated_review/course_data_contracts.csv")
        inventory = pd.read_csv(root / "results/report_supplement/tables/source_table_inventory.csv")
        summary = json.loads((root / "results/integrated_review/course_data_contracts.json").read_text())
    assert summary["hard_checks_passed"]
    ctx["contract_summary"] = summary
    return inventory, contracts


def prepare_orders(ctx):
    if ctx["archive"] is None:
        ctx["preparation_status"] = {"recomputed_in_this_run": False, "reason": "course ZIP not supplied"}
        return None, None
    prepared = ctx["run"] / "prepared_workflow"
    prepared.mkdir()
    log = io.StringIO()
    with redirect_stdout(log):
        train, test = reproduce_best.prepare(ctx["archive"], prepared)
    (ctx["run"] / "preparation.log").write_text(log.getvalue())
    checks.audit_random_frames(train, test).to_csv(ctx["run"] / "fold_audit.csv", index=False)
    ctx["preparation_status"] = json.loads((prepared / "preparation_verification.json").read_text())
    assert len(train) == 64634 and len(test) == 31836
    assert train.lead_time_days.gt(60).sum() + test.lead_time_days.gt(60).sum() == 306
    assert train.max_distance_km.isna().sum() + test.max_distance_km.isna().sum() == 477
    assert train.avg_product_weight_g.isna().sum() + test.avg_product_weight_g.isna().sum() == 22
    return train, test


def refit_models(ctx):
    if ctx["mode"] != "reproduce":
        return
    root, run = ctx["root"], ctx["run"]
    command = [sys.executable, str(root / "scripts/reproduce_report_supplement.py"),
               "--archive", str(ctx["archive"]), "--output", str(run / "fixed_models"), "--skip-importance"]
    if ctx["with_cv"]:
        command.append("--with-cv")
    with (run / "model_fitting.log").open("w") as log:
        subprocess.run(command, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT)
    rebuilt = pd.read_csv(run / "fixed_models/tables/train_test_reproduction.csv")
    published = pd.read_csv(root / "results/report_supplement/tables/metrics_by_phase.csv")
    matched = rebuilt.merge(published, on=["model", "phase"], suffixes=("_new", "_saved"), validate="one_to_one")
    for metric in ["MAE_days", "RMSE_days", "R2"]:
        np.testing.assert_allclose(matched[metric + "_new"], matched[metric + "_saved"], atol=1e-8, rtol=0)


def finish(ctx, final):
    record = {"mode": ctx["mode"], "course_data_rebuilt": ctx["archive"] is not None,
              "source_contracts": ctx["contract_summary"], "preparation": ctx["preparation_status"],
              "fixed_models_refitted": ctx["mode"] == "reproduce",
              "regression_CV_refitted": ctx["mode"] == "reproduce" and ctx["with_cv"],
              "search_repeated": False, "final_stack_fitted": False,
              "source_checks": ctx["manifest_checks"], "final_test_MAE_days": float(final.Test_MAE_days),
              "final_test_RMSE_days": float(final.Test_RMSE_days), "final_test_R2": float(final.Test_R2),
              "CV_SD_definitions": {"population": 0, "sample": 1}, "validation_data_in_regression": False}
    (ctx["run"] / "execution_record.json").write_text(json.dumps(record, indent=2) + "\n")
