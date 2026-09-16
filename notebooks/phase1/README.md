# Notebooks

Jupyter/Colab notebooks for data exploration, EDA, feature engineering and modelling.

Run `00` first: it builds `data/smartcommerce_consolidated_v2.csv`, the order-item-grain table the
other notebooks read. Everything is reproducible from the nine raw CSVs of the course archive —
unpack them into `data/Olist_CSV/` at the repository root (that folder is git-ignored), or point
`OLIST_CSV_DIR` at wherever you keep them.

| Notebook | What it covers | Milestone 1 deliverable item |
|---|---|---|
| `00_dataset_overview_eda.ipynb` | Shared foundation: nine-table overview, join integrity, temporal patterns, entity distributions, correlation exploration, and the build of the consolidated table | Dataset overview · temporal patterns · distributions · correlation |
| `01_customer_satisfaction_eda.ipynb` | Review scores vs. category, freight, delivery, location | Candidate problem exploration |
| `02_late_delivery_classification_eda.ipynb` | Whether an order arrives after its promised date | Candidate problem exploration |
| `03_delivery_lead_time_regression_eda.ipynb` | Purchase-to-delivery duration in days | Candidate problem exploration |
| `04_regional_consumer_behaviour_eda.ipynb` | Regional demand, category mix, state x month GMV | Candidate problem exploration |

## Conventions

- **Grain.** `smartcommerce_consolidated_v2.csv` is one row per `(order_id, order_item_id)`.
  Use the `to_order_level()` helper in `00` when you need one row per order, so that every notebook
  aggregates the same way.
- **Identity.** `customer_id` is regenerated per order. Use `customer_unique_id` for anything at the
  person level.
- **Leakage.** `00` section 6.2 records which columns are known at purchase time and which are
  post-outcome. Consult it before adding a feature; do not re-decide this per notebook.

## Milestone 1 Deliverables

- [Submitted Report](reports/phase1/Team9_Phase1_IT5006_AY2627Sem1.pdf)
- [EDA Notebooks](notebooks/phase1/)
- [Interactive Dashboard](https://it5006-group9-olist-project-c5hmxth6jt543wszxqoaqd.streamlit.app/)
