# Phase 2 · Regression

Predict purchase-to-receipt delivery lead time. The final selected model remains the 63-leaf HGB; random-test MAE **4.1505 days**. Directory reorganization changes locations and discovery paths, not model selection or reported performance.

| Folder | Purpose | Start here |
|---|---|---|
| `delivery_regression_best/` | Current selected model, full Train–CV–Test evidence, figures and reproduction | [Package](delivery_regression_best/README.md), [best-model notebook](delivery_regression_best/notebooks/best_scheme_review.ipynb), [report evidence](delivery_regression_best/docs/report_evidence_guide.md) |
| `random_stacking/` | Latest nested Ridge/RF and stacking comparison; HGB remains selected | [Guide](random_stacking/README.md), [stacking notebook](delivery_regression_best/notebooks/tuned_stacking_comparison.ipynb) |
| `historical_temporal/` | Earlier time-split experiments, source, notebooks, results and reproduction | [Historical workspace](historical_temporal/README.md) |

Stacking outer-CV MAE is **4.2374** versus HGB **4.2378** days; the latest stack has no final Train/Test score. Historical temporal scores belong to their separate protocols. See [layout migration evidence](layout_migration.json) for preserved files and outputs.

Classification imports the unchanged geographic helpers and historical constants from `historical_temporal/src/`; this dependency does not make the two tasks share targets, split manifests or current random seeds. For historical reproduction set `OLIST_ARCHIVE` or `OLIST_CSV_DIR` explicitly.

[Phase 2 overview](../README.md) · [Classification](../Classification/README.md)
