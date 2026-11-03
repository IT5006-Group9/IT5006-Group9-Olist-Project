"""Single-order inference using the teammates' frozen classification forests.

Only the fitted preprocessing is prepared offline. The explicit prediction button
evaluates the selected portable forest; it never reads saved prediction scores.
Historical labels are loaded separately, only after the user requests a reveal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets/classification"
VERSIONS = {
    "strict": "Strict cutoff · Availability rerun",
    "v1": "v1 · Frozen headline",
}
OUTCOME_COLUMNS = {"review_score", "review_answer_timestamp", "is_low_review", "y",
                   "delivery_days", "is_on_time", "order_delivered_customer_date"}


def clear_prediction():
    st.session_state.pop("clf_forecast", None)
    st.session_state["clf_reveal"] = False


def change_order(order_ids, direction):
    current = st.session_state.get("clf_order")
    index = order_ids.index(current) if current in order_ids else 0
    if direction == "random":
        alternatives = [order for order in order_ids if order != current]
        target = random.choice(alternatives) if alternatives else current
    else:
        target = order_ids[max(0, min(len(order_ids) - 1, index + direction))]
    clear_prediction()
    st.session_state["clf_order"] = target


def checked_file(base, name, expected_sha):
    if Path(name).name != name:
        raise ValueError("Invalid classification asset filename.")
    path = Path(base) / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Classification asset verification failed: {name}")
    return path


@st.cache_data(show_spinner=False)
def load_inputs(base, manifest_text, version):
    manifest = json.loads(manifest_text)
    spec = manifest["versions"][version]
    def file(name):
        return checked_file(base, name, manifest["files"][name])
    inputs = pd.read_parquet(file(spec["inputs"]))
    matrix = np.load(file(spec["matrix"]), allow_pickle=False)
    if OUTCOME_COLUMNS & set(inputs.columns):
        raise ValueError("Historical outcomes cannot be classification inputs.")
    if not inputs.order_id.is_unique or len(inputs) != manifest["n_examples"]:
        raise ValueError("Classification examples must have unique order IDs.")
    columns = manifest["metadata_columns"] + manifest["display_feature_columns"]
    if set(inputs.columns) != set(columns):
        raise ValueError("Classification input columns do not match the export manifest.")
    ids_sha = hashlib.sha256("\n".join(inputs.order_id.astype(str)).encode()).hexdigest()
    if ids_sha != spec["input_order_ids_sha256"]:
        raise ValueError("Classification example order does not match the matrix.")
    if matrix.shape != (len(inputs), spec["n_features"]) or np.isinf(matrix).any():
        raise ValueError("Invalid classification input matrix.")
    return inputs, matrix


@st.cache_resource(show_spinner="Loading the frozen low-review forest…")
def load_model(base, name, expected_sha):
    from deployment.classification_runtime import load_forest
    return load_forest(checked_file(base, name, expected_sha))


@st.cache_data(show_spinner=False)
def load_outcomes(base, name, expected_sha):
    outcomes = pd.read_parquet(checked_file(base, name, expected_sha))
    if not outcomes.order_id.is_unique:
        raise ValueError("Historical review IDs must be unique.")
    return outcomes.set_index("order_id")


def score_example(matrix, ordinal, spec, files):
    model = load_model(str(ASSETS), spec["forest"], files[spec["forest"]])
    positive = np.flatnonzero(model.classes_ == 1)
    if len(positive) != 1:
        raise ValueError("The fitted classifier must contain the low-review class 1.")
    score = float(model.predict_proba(matrix[[ordinal]])[0, positive[0]])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("The classifier did not produce a valid risk score.")
    return score


def render(header, panel_metrics, chart):
    header("Which orders may need a closer look?", "Run the frozen low-review classifier on a historical order at its survey trigger time. A low review means 1–2 stars.")
    st.caption("1. Choose an order → 2. Run its risk prediction → 3. Optionally reveal the historical review")
    try:
        manifest_text = (ASSETS / "manifest.json").read_text()
        manifest = json.loads(manifest_text)
        version = st.selectbox("Prediction model", list(VERSIONS), format_func=VERSIONS.get,
                               key="clf_version", on_change=clear_prediction)
        spec = manifest["versions"][version]
        inputs, matrix = load_inputs(str(ASSETS), manifest_text, version)
    except (OSError, ValueError, KeyError, ImportError) as error:
        clear_prediction()
        st.warning(f"The classification demonstration package is unavailable: {error}")
        return
    if version == "strict":
        st.info("Strict-cutoff demo · Features from events after T are masked. The project’s published headline remains v1; this demo defaults to the availability-controlled version.")
    else:
        st.warning("v1 retrospective comparison · This original model has label/event availability limitations. Its inputs and threshold differ from the strict-cutoff version. See Model evidence for the timing audit.")
    st.caption("Prediction point T = the earlier of delivery and the promised delivery date (promise if undelivered). This is later in the order lifecycle than the purchase-time delivery forecast.")
    with st.container(key="olist_split_classification_prediction"):
        controls, output = st.columns([1, 1.65], gap="large")
    with controls, st.container(border=True):
        st.subheader("1 · Choose a historical order")
        months = ["All months", *sorted(inputs.t_month.unique())]
        month = st.selectbox("Trigger month", months, key="clf_month", on_change=clear_prediction)
        candidates = inputs if month == "All months" else inputs[inputs.t_month.eq(month)]
        states = ["All states", *sorted(candidates.customer_state.dropna().unique())]
        if st.session_state.get("clf_state") not in states:
            st.session_state["clf_state"] = "All states"
            clear_prediction()
        state = st.selectbox("Customer destination", states, key="clf_state", on_change=clear_prediction)
        if state != "All states":
            candidates = candidates[candidates.customer_state.eq(state)]
        if candidates.empty:
            clear_prediction()
            st.info("No demonstration order matches this selection.")
            return
        indexed = candidates.set_index("order_id", drop=False)
        order_ids = candidates.order_id.tolist()
        if st.session_state.get("clf_order") not in order_ids:
            st.session_state["clf_order"] = order_ids[0]
            clear_prediction()
        def order_label(order_id):
            row = indexed.loc[order_id]
            return f"{pd.Timestamp(row.survey_trigger_time):%d %b} · {row.seller_state} → {row.customer_state} · …{order_id[-6:]}"
        selected = st.selectbox("Historical classification order", order_ids, format_func=order_label,
                                key="clf_order", on_change=clear_prediction)
        signature = (hashlib.sha256(manifest_text.encode()).hexdigest(), version, month, state, selected)
        if st.session_state.get("clf_selection") != signature:
            clear_prediction()
            st.session_state["clf_selection"] = signature
        position = order_ids.index(selected)
        st.caption(f"Order {position + 1} of {len(order_ids)} in the current selection")
        with st.container(key="olist_demo_navigation"):
            previous, following, shuffle = st.columns(3)
        previous.button("← Previous", key="clf_previous", on_click=change_order,
                        args=(order_ids, -1), disabled=position == 0, width="stretch")
        following.button("Next →", key="clf_next", on_click=change_order,
                         args=(order_ids, 1), disabled=position == len(order_ids) - 1, width="stretch")
        shuffle.button("Random", key="clf_random", on_click=change_order,
                       args=(order_ids, "random"), disabled=len(order_ids) < 2, width="stretch")
        record = indexed.loc[selected]
        st.markdown("**Order at the prediction point**")
        st.write(f"Trigger: {pd.Timestamp(record.survey_trigger_time):%d %B %Y, %H:%M}")
        st.write(f"{record.seller_state} → {record.customer_state} · {int(record.n_items)} item(s)")
        st.write(f"Item value R\\$ {record.total_price:,.2f} · Freight R\\$ {record.total_freight:,.2f}")
        st.write("Delivered by T: **" + ("Yes" if record.is_delivered else "No") + "**")
        if st.button("Run low-review prediction", type="primary", width="stretch"):
            clear_prediction()
            try:
                ordinal = int(np.flatnonzero(inputs.order_id.eq(selected).to_numpy())[0])
                score = score_example(matrix, ordinal, spec, manifest["files"])
                threshold = float(spec["capacity_threshold"])
                if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                    raise ValueError("Invalid frozen validation threshold.")
                st.session_state["clf_forecast"] = {
                    "order_id": selected, "version": version, "model": spec["model_label"],
                    "trigger_time": str(record.survey_trigger_time), "risk_score": score,
                    "validation_threshold": threshold, "above_threshold": score >= threshold,
                    "selection_sha256": spec["selection_sha256"], "model_sha256": spec["model_sha256"],
                }
            except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
                clear_prediction()
                st.error(f"Unable to score this order: {error}")
        st.caption(f"{len(inputs)} fixed historical examples · 40 per June–August 2018 trigger month. Within the eligible test cohort, order-ID sampling uses neither review values nor risk scores.")
        st.caption("Changing a model, filter or order, or leaving this page, clears the prediction and revealed review. Browsing never runs inference automatically.")
    with st.expander("See the classification inputs and their meanings"):
        from deployment.classification_input_help import CLASSIFICATION_INPUT_MEANINGS
        st.caption("Read-only recorded inputs for the selected model. Missing may reflect absent source data, information unavailable by T, or version-specific masking; see each field’s definition. Strict and v1 inputs can differ. IDs and timestamps identify the example or retrieve frozen history; they are not direct numeric features.")
        details = record[manifest["metadata_columns"] + manifest["display_feature_columns"]].to_frame("Value")
        details["Value"] = details["Value"].map(lambda value: "Missing / unavailable" if pd.isna(value) else str(value))
        details = details.rename_axis("Input").reset_index()
        details["Meaning / units"] = details.Input.map(CLASSIFICATION_INPUT_MEANINGS)
        with st.container(height=560, border=False, key="olist_clf_input_details"):
            st.markdown(details.to_html(index=False, escape=True, border=0, classes="olist-input-table"), unsafe_allow_html=True)
    with output:
        result = st.session_state.get("clf_forecast")
        if not result or result["order_id"] != selected or result["version"] != version:
            st.markdown('<div class="empty"><div class="eyebrow">LOW-REVIEW RISK</div><h2>Start with an order.</h2><p>Choose a historical order and run its low-review prediction.</p><p class="quiet">The actual review stays hidden until you choose to reveal it.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill">PREDICTION CALCULATED</span>', unsafe_allow_html=True)
            st.subheader("2 · Low-review risk")
            st.caption(f"Selected model: {VERSIONS[version]} · Raw forest score, not a calibrated probability.")
            score, threshold = result["risk_score"], result["validation_threshold"]
            panel_metrics([
                ("Low-review risk score", f"{score:.3f}", "Raw fitted random-forest score for class 1; not a calibrated probability."),
                ("Frozen validation threshold", f"{threshold:.4f}", "This version’s threshold from validation, used unchanged for this single-order check."),
            ], key="classification_prediction")
            if result["above_threshold"]:
                st.warning("Above the frozen validation threshold · This order is flagged for review under this score rule.")
            else:
                st.info("Below the frozen validation threshold · This order is not flagged under this score rule. A low review is still possible.")
            fig = go.Figure(go.Bar(y=["Low-review risk"], x=[score], orientation="h",
                                   marker_color="#4263eb", text=[f"{score:.3f}"], textposition="outside", cliponaxis=False))
            fig.add_vline(x=threshold, line_dash="dash", line_color="#b97728",
                          annotation_text=f"Threshold {threshold:.4f}", annotation_position="top left")
            fig.update_xaxes(range=[0, 1], title="Raw risk score")
            fig.update_yaxes(title=None)
            chart(fig, 210)
            st.caption("This is a fixed score threshold, not a top-20% rank for one order. It does not guarantee a 20% flag rate or the project’s 50% recall target on future orders. Scores are uncalibrated; contacting a customer has not been shown to change their review.")
            st.markdown("**3 · Optional outcome check**")
            if st.checkbox("Reveal the historical review", key="clf_reveal"):
                try:
                    name = manifest["outcomes"]
                    actual = load_outcomes(str(ASSETS), name, manifest["files"][name]).loc[selected]
                    panel_metrics([
                        ("Historical review", f"{int(actual.review_score)} / 5 stars", "Observed review, loaded only after this reveal action."),
                        ("Observed low review", "Yes · 1–2 stars" if actual.y else "No · 3–5 stars", "The observed classification label, not a prediction."),
                    ], key="classification_outcome")
                    st.caption(f"Review answered: {pd.Timestamp(actual.review_answer_timestamp):%d %B %Y, %H:%M}. One example does not establish overall classification performance.")
                except (OSError, ValueError, KeyError, ImportError) as error:
                    st.warning(f"The historical review is unavailable: {error}")
            st.download_button("Download this risk prediction", pd.DataFrame([result]).to_csv(index=False),
                               file_name=f"low_review_{version}_{selected[-6:]}.csv", mime="text/csv")
    with st.expander("How this demonstration runs"):
        st.write("The original fitted history transformer and preprocessor were applied once at export to these fixed examples. Clicking Run evaluates the uploaded model’s fitted forest on the selected prepared row. No training or model selection runs in the app.")
        st.write("Frozen seller and route history comes only from events stored by the fitted pipeline, with each lookup strictly before the order’s purchase. It is not refreshed with later test reviews. Current-order review fields never enter prediction.")
        st.write("The portable forest preserves every tree, split, missing-value route and leaf score. This keeps the existing regression runtime compatible with the teammates’ newer classification training environment.")
        verification = spec["verification"]
        st.caption(f"Verified on all {verification['rows']} examples against the original fitted pipeline; maximum absolute score difference {verification['portable_vs_fitted_pipeline_max_abs_diff']:.2e}. Model SHA-256: {spec['model_sha256']}")
        st.caption("June–August 2018 historical assessment data was already examined during the project. This demonstration does not create a new independent holdout or connect to a live order feed. Aggregate model scores remain on Model evidence.")
        st.caption("The eligible classification cohort contains orders with items and an observed review answered at or after T. Orders without a qualifying review are outside these evaluation results.")
