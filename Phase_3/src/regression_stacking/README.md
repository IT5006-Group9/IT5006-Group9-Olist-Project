# All Phase 2 regression combinations: chronological rerun

This separate development suite migrates all **14** combinations from the Phase 2 ensemble notebook. It preserves their base specifications and their fixed-versus-tuned distinctions. It does not replace the earlier Phase 3 single-model or one-stack experiment.

- [Executed comparison notebook](../../notebooks/regression/phase2_stacking_chronological.ipynb)
- [Experiment guide and full inventory](../../docs/regression/phase2_stacking_chronological.md)
- [Published results](../../outputs/regression_stacking_chronological/)

## Reproduction

Use Python 3.12 and the dedicated environment, from the repository root:

```bash
python3.12 -m venv .venv-stacking
.venv-stacking/bin/python -m pip install -r Phase_3/requirements-stacking.lock.txt
STACKING_MODEL_THREADS=4 PYTHONPATH=Phase_3/src .venv-stacking/bin/python \
  -m regression_stacking.run --csv-dir /path/to/Olist_CSV
PYTHONPATH=Phase_3/src .venv-stacking/bin/python \
  Phase_3/scripts/verify_stacking_chronological.py
PYTHONPATH=Phase_3/src .venv-stacking/bin/python \
  Phase_3/scripts/build_stacking_chronological_report.py
```

The runner verifies all seven course CSV members against the existing Phase 3 SHA-256 inventory. The original course ZIP is not required when identical extracted CSVs are available. It stores private data, OOF matrices, model checkpoints and order-level predictions under the ignored `Phase_3/data/regression_stacking_chronological/` directory. Published aggregate outputs are under `Phase_3/outputs/regression_stacking_chronological/`.

The protocol is recorded before fitting. Checkpoints include input identities, row order, cutoff, model specification and training-source hashes. Repeating the same command resumes matching checkpoints. A changed protocol or source is rejected rather than mixed with earlier results; run changed experiments in a separate checkout with a new experiment directory.

## Evaluation

Five outer validation months are January-May 2018. Each outer fit requires both purchase before its cutoff minus 39 days and receipt strictly before the cutoff. Inner tuning and stacking use three recent past-only purchase-month windows within that outer training set. The 12 Ridge, six RF and two ExtraTrees candidates are selected separately inside each outer training partition. All 14 combinations use identical outer validation rows.

The original random risk cross-fit is replaced by monthly forward-only fitting inside each regression fitting partition. Initial rows receive probability zero and an explicit unavailable flag, retaining the common training cohort. Original Phase 2 HGB/risk parameters and F0/F1/time/purchase inputs remain documented in `base_models.py` and the frozen protocol.

Outer validation scores use only outcomes observed before June 1. May has incomplete label coverage. This is a retrospective development comparison: the historical dataset and base configurations were used before. The runner never predicts or scores June-August and verifies that all pre-existing classification and regression output files remain unchanged. It does not change the deployed model or reopen the completed final assessment.

## Checks

```bash
PYTHONPATH=Phase_3/src .venv-stacking/bin/python -m unittest discover \
  -s Phase_3/tests -p test_stacking_chronological.py -v
```

The independent verifier replays saved combination predictions, checks their metrics and weight constraints, checks inner selection, inspects base/risk time audits and confirms preservation of the original evidence. Training MAE and the second-level meta-fit MAE are diagnostics, not held-out estimates. Monthly mean MAE and pooled MAE are reported separately.
