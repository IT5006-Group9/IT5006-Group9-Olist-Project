# Delivery lead-time regression — Phase 3

The regression branch predicts fractional purchase-to-receipt days using purchase-time inputs. It uses five expanding monthly validation windows (January–May 2018) and a June–August test, with a fixed embargo and strict label-availability checks.

## Reading entries

- [Complete chronological notebook](../../notebooks/regression/delivery_regression_chronological.ipynb)
- [Temporal ensemble notebook](../../notebooks/regression/stacking_chronological.ipynb)
- [Report evidence and supporting documents](../../docs/regression/README.md)
- [Implementation protocol](../../docs/regression/implementation_protocol.md)
- [Saved evidence](../../outputs/regression/)

## Reproduction

From the repository root, use Python 3.12 and install `requirements-regression.lock.txt`. Keep the classification environment separately if it uses a different scikit-learn version.

Check the published snapshots without fitting a model:

```bash
python -m pip install -r requirements-regression.lock.txt
python scripts/verify_regression_data.py
```

To regenerate the raw-data audit, choose an empty output directory:

```bash
OLIST_ARCHIVE=/absolute/path/to/IT5006_Project-Data.zip \
  python src/regression/run_pipeline.py --stage audit \
  --output-root /absolute/path/to/new_regression_run
```

Then run the training stages in that same isolated directory:

```bash
OLIST_ARCHIVE=/absolute/path/to/IT5006_Project-Data.zip \
  python src/regression/run_pipeline.py --stage all \
  --output-root /absolute/path/to/new_regression_run
```

The archive fingerprint is checked before reading. A CSV-directory override must match the same archive byte for byte. `--output-root` isolates a fresh run from the published evidence; the generated reading materials include a source copy for portability.

Stages are `audit`, `select`, `diagnostics`, `stacking`, `freeze`, `test`, `analysis`, `reference`, `figures` and `package`. The audit runs before any estimator fits. `select` uses five monthly mean validation MAEs, with sample SD; eventual validation outcomes are post-selection diagnostics only. `test` rejects a second completed assessment. Reading either notebook displays the saved evidence without launching training.

```bash
python -m unittest discover -s tests -p test_regression.py -v
PYTHONPATH=src python -m regression.notebooks
PYTHONPATH=src python -m regression.verify
PYTHONPATH=src python -m regression.delivery
```

The frozen selected pipeline can also be rebuilt directly from the ZIP in a new directory, without repeating the hyperparameter search:

```bash
PYTHONPATH=src python -m regression.reproduce_frozen \
  --archive /path/to/IT5006_Project-Data.zip --output-root /path/to/new_frozen_run
```

The existing audited order tables and chronological fold manifest are published under [`data/regression`](../../data/regression/README.md). Verify them without model fitting with `python scripts/verify_regression_data.py`; add `--archive /path/to/IT5006_Project-Data.zip` to independently rebuild the tables in a temporary directory. Model checkpoints, transformed matrices, per-order predictions and intermediate caches remain locally generated. Aggregate tables and figures, executed notebooks and explanatory documents form the reviewable deliverable. The [deployment handoff](../../docs/regression/deployment_handoff.md) documents the frozen model and purchase-input scoring interface.

## Interpretation

The evaluation is conditional on observed completed deliveries. Validation labels must exist before June 1 to support the simulated model-selection decision; this particularly limits May coverage. Historical test orders were used in previous experiments, so a new test lock cannot make them untouched external evidence. Archived feature availability is an explicit assumption. See the protocol and report guide for the effect of these limits.

Codex assisted with implementation, verification, figures and documentation under the project direction. The team is responsible for the submitted interpretation.
