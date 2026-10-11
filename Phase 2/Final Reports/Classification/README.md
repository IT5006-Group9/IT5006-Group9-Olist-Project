# Classification report evidence

These are copies of the existing result tables and figures; the modelling notebook is unchanged.

- [Model ladder](tables/model_comparison.csv)
- [OOF ensemble comparison](tables/ensemble_test_results.csv)
- [Sampling comparison](tables/sampling_comparison.csv)
- [Weighted RF operating points](tables/rf_operating_points.csv)
- [Permutation importance](tables/permutation_importance.csv)
- [Grouped SHAP](tables/shap_importance.csv)
- [Figures](figures/)

The main ladder/operating-point RF is weighted, whereas the sampling SHAP RF uses no class weighting.
Identify the configuration with every quoted score or threshold. This migration copies the evidence;
it does not reconcile the pending notebook review questions or declare an independently rerun model.
