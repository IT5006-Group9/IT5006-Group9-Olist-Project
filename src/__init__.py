"""IT5006 Group 9 - Phase 2 source package.

Modules
-------
config     paths, random seed, chronological windows, column roles
data       load the nine raw Olist CSVs and build the order-level base table
features   feature construction (order, product, geography, timing, payment,
           point-in-time seller / route history, delivery-as-of-T) + leakage checks
split      chronological train / validation / test windows and the label-matured CV folds
pipelines  scikit-learn preprocessing + the model ladders for both problems
ensembles  Family C: OOF predictions, mean / convex / stacked combinations (both problems)
evaluate   metrics, threshold search and result-table helpers

regression_dev/  modules behind notebooks/phase2/regression_dev (development record)
"""
