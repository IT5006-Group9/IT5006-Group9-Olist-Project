"""Build reading artifacts from the verified supplementary stacking assessment.

Run from the repository root with the assessment environment:
python Phase_3/scripts/build_selected_stacking_report.py

This script reads saved tables and metadata only. It never fits a model,
generates predictions, or selects a model using test results.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/regression_stacking_holdout"
NOTEBOOK = ROOT / "notebooks/regression/selected_stacking_holdout.ipynb"
DOCUMENT = ROOT / "docs/regression/selected_stacking_holdout.md"
TABLE_NAMES = ("test_comparison", "test_by_month", "frozen_weights", "meta_time_audit")
INPUTS = tuple("tables/" + name + ".csv" for name in TABLE_NAMES) + (
    "frozen_selection.json", "verification.json",
)
ORDER = ("stack_two_fixed", "mean_three_fixed", "ridge_fixed", "forest_fixed",
         "hgb63_phase2", "HGB_risk_current_main")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_verified():
    verification = json.loads((OUT / "verification.json").read_text())
    assert verification.get("passed") is True, "Run the assessment verifier before building this report."
    assert verification.get("original_evidence_unchanged") is True
    assert verification.get("historical_test_exposure_disclosed") is True
    frozen = json.loads((OUT / "frozen_selection.json").read_text())
    assert frozen["selected"] == {"stack": "stack_two_fixed", "mean": "mean_three_fixed"}
    frames = {name: pd.read_csv(OUT / "tables" / (name + ".csv")) for name in TABLE_NAMES}
    comparison = frames["test_comparison"]
    assert comparison.candidate.is_unique and set(comparison.candidate) == set(ORDER)
    assert comparison.n.eq(18603).all() and verification["test_n"] == 18603
    assert np.isfinite(comparison[["cv_mae", "mae", "rmse"]]).all().all()
    frames["test_comparison"] = comparison.set_index("candidate").loc[list(ORDER)].reset_index()
    monthly = frames["test_by_month"]
    assert not monthly.duplicated(["candidate", "month"]).any()
    assert set(monthly.candidate) == set(ORDER[:-1])
    assert set(monthly.month) == {"2018-06", "2018-07", "2018-08"}
    for candidate, group in monthly.groupby("candidate", sort=False):
        assert len(group) == 3 and group.n.sum() == 18603
        expected = comparison.set_index("candidate").loc[candidate]
        np.testing.assert_allclose((group.n * group.mae).sum() / group.n.sum(), expected.mae, atol=1e-10)
    weights = frames["frozen_weights"]
    assert weights.candidate.eq(frozen["selected"]["stack"]).all()
    assert weights.member.is_unique and set(weights.member) == set(frozen["stack_members"])
    np.testing.assert_allclose(
        weights.set_index("member").loc[frozen["stack_members"], "weight"], frozen["stack_weights"], atol=1e-12
    )
    for values in (frozen["stack_weights"], frozen["mean_weights"]):
        assert np.isfinite(values).all() and min(values) >= 0
        np.testing.assert_allclose(sum(values), 1, atol=1e-12)
    audit = frames["meta_time_audit"]
    assert len(audit) == 3 and audit.score_n.sum() == frozen["meta_rows"]
    assert audit.future_training_labels.eq(0).all() and audit.overlap.eq(0).all()
    for row in audit.itertuples():
        assert pd.Timestamp(row.latest_train_label) < pd.Timestamp(row.cutoff)
        assert pd.Timestamp(row.latest_train_purchase) < pd.Timestamp(row.cutoff) - pd.Timedelta(days=39)
        assert pd.Timestamp(row.latest_score_label) < pd.Timestamp("2018-06-01")
    hashes = {relative: digest(OUT / relative) for relative in INPUTS}
    return frames, frozen, verification, hashes


def comparison_table(frame):
    return frame[["candidate", "cv_mae", "mae", "rmse", "n"]].rename(columns={
        "cv_mae": "CV mean monthly MAE (days)", "mae": "Test pooled MAE (days)",
        "rmse": "Test pooled RMSE (days)", "n": "Test orders",
    })


def markdown_table(frame):
    # Avoid an optional tabulate dependency in the report environment.
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def document(frames, frozen):
    comparison = frames["test_comparison"]
    selected = comparison.set_index("candidate")
    stack = selected.loc[frozen["selected"]["stack"]]
    mean = selected.loc[frozen["selected"]["mean"]]
    main = selected.loc["HGB_risk_current_main"]
    weights = ", ".join(f"{member}={weight:.6f}" for member, weight in
                        zip(frozen["stack_members"], frozen["stack_weights"]))
    text = f"""# Selected ensembles: supplementary chronological test assessment

At the user's request, two ensembles chosen from the completed January–May 2018 validation results were assessed on the existing June–August 2018 test orders. **Only two ensembles were tested here**, together with their three standalone members: the best learned stack, `stack_two_fixed`, and the best equal-weight mean, `mean_three_fixed`. The earlier chronological development experiment evaluated all 14 combinations; this supplementary assessment does not test all 14 on the final window.

The learned stack's test MAE is **{stack.mae:.4f} days**, with validation MAE **{stack.cv_mae:.4f} days**. The equal mean's test MAE is **{mean.mae:.4f} days**, with validation MAE **{mean.cv_mae:.4f} days**. The existing main model's saved test MAE is **{main.mae:.4f} days**. All test comparisons use the same **18,603 completed delivery orders**, with purchase timestamps from 1 June through 31 August 2018.

## Why validation and test scores are both reported

Validation estimates performance during model development and determines which configurations and combinations to carry forward. Here, CV MAE is the unweighted average of the five January–May monthly MAEs, using outcomes observed before 1 June. The final test period provides a later assessment after the chosen specifications and weights have been frozen. Test MAE pools individual orders across June–August, so larger months contribute more observations.

These values answer different questions and need not be equal. The time periods, available training history, outcome maturity and averaging rules differ. In particular, May validation outcomes are limited to deliveries already observed before 1 June; test scoring uses the archived completed outcomes without excluding an order merely because delivery occurred after August. The score difference cannot be attributed solely to stacking or to the time split.

## Frozen results

{markdown_table(comparison_table(comparison))}

The table retains a fixed presentation order and reports every frozen comparison. `hgb63_phase2` is the migrated Phase 2 HGB member with 63 leaves; `HGB_risk_current_main` is the different 15-leaf main Phase 3 model and reuses its saved assessment without refitting. Their feature representations and specifications also differ. This table is not a matched-input model-family comparison.

The lowest test row is descriptive only. **These test results do not reopen selection or change deployment.** The existing Phase 3 model decision and its original evidence remain unchanged.

## Final training and learned weights

Final base training uses purchases on or after 1 January 2017, strictly before `2018-06-01 - 39 days`, with delivery labels strictly before 1 June. The stack learns nonnegative weights summing to one, without an intercept, from three forward OOF purchase-month windows within that eligible training partition. Those windows are February, March and April; April is partial because of the 39-day embargo. All meta-labels are available before 1 June. The specifications, comparison list and final weights were frozen before any June–August prediction.

The final learned weights are **{weights}**, using **{frozen['meta_rows']:,}** OOF rows. The three-member mean retains fixed one-third weights and learns no second-level parameters. The OOF meta-fit error is a training diagnostic, not another independent evaluation score.

{markdown_table(frames['meta_time_audit'][['month', 'train_n', 'score_n', 'latest_train_purchase', 'latest_train_label']])}

## Historical exposure and reproduction

These orders were exposed during earlier Phase 2 work and the existing Phase 3 assessment. This is a **supplementary historical assessment**, not a new independent unseen test set. Results are conditional on the completed-delivery cohort; they do not establish performance for all placed orders. The original development outputs, test lock, main-model results and notebooks are preserved.

The [executed reading notebook](../../notebooks/regression/selected_stacking_holdout.ipynb) displays this comparison, monthly test results, frozen weights and time audits. It reads only saved assessment artifacts and performs no training, prediction or selection. Run the assessment verifier before rebuilding it:

```bash
python Phase_3/scripts/evaluate_selected_stacking.py --verify
python Phase_3/scripts/build_selected_stacking_report.py
```

To reproduce the complete assessment from the repository root, use the Python 3.12 environment pinned in `Phase_3/requirements-stacking.lock.txt`:

```bash
STACKING_MODEL_THREADS=4 PYTHONPATH=Phase_3/src python Phase_3/scripts/evaluate_selected_stacking.py --csv-dir /path/to/Olist_CSV
python Phase_3/scripts/evaluate_selected_stacking.py --verify
python Phase_3/scripts/build_selected_stacking_report.py
```

Saved artifacts are in [outputs/regression_stacking_holdout](../../outputs/regression_stacking_holdout). The separate [all-14 development report](phase2_stacking_chronological.md) remains the record of the earlier validation-only experiment.
"""
    DOCUMENT.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT.write_text(text)


def reading_notebook():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook()
    nb.cells = [md("""# Selected ensembles: supplementary chronological test assessment

This executed reading notebook loads saved, verified assessment artifacts. It performs no model fitting, prediction or selection. Two ensembles were selected using January–May validation: the best learned stack and the best equal-weight mean. Three standalone members provide context. **The 14-variant development suite has not been rescored in full on the test window.**"""), code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display

root = next(candidate for parent in (Path.cwd(), *Path.cwd().parents)
            for candidate in (parent, parent / 'Phase_3')
            if (candidate / 'outputs/regression_stacking_holdout').is_dir())
out = root / 'outputs/regression_stacking_holdout'
verification = json.loads((out / 'verification.json').read_text())
assert verification['passed'] is True
frozen = json.loads((out / 'frozen_selection.json').read_text())
tables = out / 'tables'
comparison = pd.read_csv(tables / 'test_comparison.csv')
monthly = pd.read_csv(tables / 'test_by_month.csv')
weights = pd.read_csv(tables / 'frozen_weights.csv')
audit = pd.read_csv(tables / 'meta_time_audit.csv')
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda value: f'{value:.4f}')
assert frozen['selected'] == {'stack': 'stack_two_fixed', 'mean': 'mean_three_fixed'}
print('Selected from saved validation, before test prediction:', frozen['selected'])
print('Test orders:', verification['test_n'])
"""), md("""## Validation and test have different roles

Validation supports development and selection. The CV column is the unweighted mean of five January–May monthly MAEs, using labels observed before 1 June. The June–August test column pools 18,603 completed-delivery orders after configurations and weights have been frozen. Orders are retained when their delivery occurs after August.

The periods, training histories, label maturity and averaging rules differ, so scores need not match. May validation excludes outcomes not yet available at the selection cutoff. This prevents unavailable labels from influencing development, but creates incomplete maturity in that month. A lower later score does not isolate an improvement caused by stacking."""), code("""order = ['stack_two_fixed', 'mean_three_fixed', 'ridge_fixed', 'forest_fixed',
         'hgb63_phase2', 'HGB_risk_current_main']
comparison = comparison.set_index('candidate').loc[order].reset_index()
assert comparison['n'].eq(18603).all()
display(comparison[['candidate', 'cv_mae', 'mae', 'rmse', 'r2', 'within3', 'n', 'assessment']].rename(
    columns={'cv_mae': 'CV mean monthly MAE (days)', 'mae': 'Test pooled MAE (days)',
             'rmse': 'Test pooled RMSE (days)'}))
"""), md("""`hgb63_phase2` is the migrated 63-leaf Phase 2 HGB member. `HGB_risk_current_main` is the different 15-leaf main Phase 3 model and uses its saved test result without refitting. Their feature representations and specifications differ as well; this table does not isolate learner complexity.

The lowest test row is a descriptive observation. **The test table does not reopen model selection or deployment.** The existing main-model decision remains unchanged."""), md("""## Monthly test results

Each of the five newly assessed predictors appears for all three purchase months. The main-model row above comes from saved evidence and is not newly scored in this notebook."""), code("""display(monthly[['candidate', 'month', 'n', 'mae', 'rmse', 'bias', 'within3']])
display(monthly.pivot(index='candidate', columns='month', values='mae').reindex(order[:-1]))
"""), md("""## Final weights and time boundaries

Final training requires `purchase >= 2017-01-01`, `purchase < 2018-06-01 - 39 days` and `delivery < 2018-06-01`. Three forward OOF windows within that partition supply the stack's meta-training rows: February, March and a partial April. Each inner fit uses purchases before its own cutoff minus 39 days and labels observed before its cutoff. Meta-labels must be observed before 1 June.

The stack's MAE-optimal nonnegative weights sum to one, with no intercept. The mean uses fixed one-third weights. Every final specification and weight was frozen before any test prediction; the meta-fit score is a training diagnostic."""), code("""display(weights)
display(pd.DataFrame({'member': frozen['mean_members'], 'weight': frozen['mean_weights']}))
display(audit)
print('OOF meta rows:', frozen['meta_rows'])
print('OOF meta-fit MAE (training diagnostic):', f"{frozen['meta_fit_mae']:.4f}")
print('Weights frozen at:', frozen['frozen_at'])
for values in (frozen['stack_weights'], frozen['mean_weights']):
    assert min(values) >= 0
    np.testing.assert_allclose(sum(values), 1, atol=1e-12)
"""), md("""## Interpretation limits and verification

These orders were exposed during Phase 2 and the earlier Phase 3 assessment. This is a supplementary historical assessment, not a new independent unseen holdout. It is conditional on completed deliveries and cannot establish performance for every placed order. The previous main-model results, test lock, development outputs and notebooks remain preserved.

The verifier replays saved ensemble predictions and weights and checks time boundaries and unchanged original evidence. This reading notebook only displays the resulting artifacts."""), code("display(verification)")]
    kernel_name = "selected-stacking-reading-report"
    with tempfile.TemporaryDirectory(prefix="selected-stacking-kernel-") as temporary:
        kernel_dir = Path(temporary) / "kernels" / kernel_name
        kernel_dir.mkdir(parents=True)
        save_json(kernel_dir / "kernel.json", {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Selected stacking reading report", "language": "python",
        })
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = temporary + (os.pathsep + previous if previous else "")
        try:
            NotebookClient(nb, timeout=120, kernel_name=kernel_name,
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(output.output_type == "error" for output in cell.outputs)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK)


def main():
    frames, frozen, verification, input_hashes = load_verified()
    document(frames, frozen)
    reading_notebook()
    assert input_hashes == {relative: digest(OUT / relative) for relative in INPUTS}
    save_json(OUT / "report_verification.json", {
        "built_at": datetime.now(timezone.utc).isoformat(), "passed": True,
        "assessment_verification_passed": verification["passed"],
        "reading_notebook_only": True, "new_model_fits": 0, "new_test_predictions": 0,
        "selected_ensembles": 2, "input_sha256": input_hashes,
        "builder_sha256": digest(__file__), "notebook_sha256": digest(NOTEBOOK),
        "document_sha256": digest(DOCUMENT),
    })
    print("Built verified English report and executed reading notebook; no fitting or prediction.")
    print(DOCUMENT.relative_to(ROOT.parent))
    print(NOTEBOOK.relative_to(ROOT.parent))


if __name__ == "__main__":
    main()
