# Phase 2 · Classification and Regression

Two workstreams, each with its own notebooks, source, results and reproduction instructions. Start with the task README instead of the historical experiment folders.

| Workstream | Target | Entry |
|---|---|---|
| **Classification** | Low-review risk (1–2 stars) at the survey trigger | [Classification](Classification/README.md), [current progress](Classification/docs/current_progress.md) |
| **Regression** | Delivery lead time from purchase to receipt, in days | [Regression](Regression/README.md), [current best model](Regression/delivery_regression_best/README.md) |

```text
Phase 2/
├── README.md
├── Classification/
│   ├── notebooks/
│   ├── src/
│   ├── config/
│   ├── docs/
│   ├── outputs/
│   ├── scripts/
│   └── tests/
└── Regression/
    ├── README.md
    ├── delivery_regression_best/  # selected HGB and report evidence
    ├── random_stacking/          # current ensemble comparison
    └── historical_temporal/      # earlier time-split experiments
```

Directory organization updated 2026-10-11. All original tracked artifacts were retained. Model settings, reported result tables, figures and saved notebook outputs are unchanged. The move updates notebook discovery, cross-package imports, command directories and entry links. Raw/derived order data, fitted models and order-level predictions remain local and ignored.

Regression's current report notebook is [best_scheme_review.ipynb](Regression/delivery_regression_best/notebooks/best_scheme_review.ipynb); its [report supplement](Regression/delivery_regression_best/notebooks/report_supplement_review.ipynb) supplies matched Train–CV–Test results and HGB explanations. The [latest stacking notebook](Regression/delivery_regression_best/notebooks/tuned_stacking_comparison.ipynb) is a development comparison; HGB remains selected. No shared Google Docs report was edited during organization.
