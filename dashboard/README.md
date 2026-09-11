# Dashboard

Interactive Streamlit dashboard for the Olist Brazilian e-commerce dataset — the
Milestone 1 dashboard deliverable.

| File | What it is |
|---|---|
| `app.py` | The Streamlit app: five tabs (Orders, Delivery, Reviews, Categories, Geography) over shared sidebar filters |
| `orders.parquet` | One row per order (98,666 rows, 3.7 MB, zstd). Committed, so the app runs anywhere without the raw CSVs |
| `build_data.py` | Regenerates `orders.parquet` |
| `requirements.txt` | Pinned dependencies for Streamlit Cloud |

## Run it locally

From the repository root:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

It opens at http://localhost:8501. Nothing else is needed — the data file is in
the repository.

## Regenerating the data

`orders.parquet` only changes when the underlying build changes. To rebuild it:

```bash
python dashboard/build_data.py
```

It reads `data/smartcommerce_consolidated_v2.csv` (produced by
`notebooks/00_dataset_overview_eda.ipynb`) and falls back to building straight
from `data/Olist_CSV/` when that file is absent. Commit the regenerated parquet —
Streamlit Cloud has no other way to get the data.

## Deploying to Streamlit Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub
   account that can see this repository.
2. **New app** → pick this repository, branch `main`, main file path
   `dashboard/app.py`.
3. Under **Advanced settings**, set the Python version to 3.11 or later. Leave
   secrets empty — the app needs none.
4. Deploy. The first build takes a few minutes while dependencies install.

Streamlit Cloud picks up `dashboard/requirements.txt` automatically because it
sits beside the app file.

**A private repository is fine** — Streamlit Cloud can deploy from one once the
GitHub account is authorised. The resulting app URL is public unless viewer
access is restricted in the app's settings, so treat the link as shareable.

Put the resulting URL in the Milestone 1 report; the deliverable asks for a live
link, not a screenshot.

## What the numbers mean

- **One row per order.** Item-level attributes are aggregated first; `category`
  is the first non-null category on the order, so a mixed-category order is
  represented by one of its categories.
- **Delivery and review metrics use the delivered cohort only** (orders with an
  actual delivery date), even when the sidebar is not restricted to delivered
  orders. Rates are therefore conditional on the order having completed.
- **The default month window is 2017-01 to 2018-08.** The extract starts and
  ends with months holding a handful of orders; including them makes every time
  series open and close on a misleading spike. Widen the slider to see them.
- **GMV** is item price plus freight, matching the definition used in the
  notebooks.
