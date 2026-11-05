"""Complete frozen-pipeline inference and explanations for new feature rows."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from deployment import order_input_ui as inputs_ui
from deployment.classification_input_help import CLASSIFICATION_INPUT_MEANINGS
from deployment.classification_explanations import explain_forest
from deployment.classification_explanation_ui import render_explanation
from deployment.regression_inputs import safe_csv

ASSETS = Path(__file__).resolve().parent / "assets/classification_pipeline"
# Historical explanations retain the original catalogue. Supplied rows use a
# separate availability adapter, including when the fitted forest is V1.
MEANINGS = {**CLASSIFICATION_INPUT_MEANINGS,
    "order_id": "Your order identifier, unique within the supplied batch. It identifies the output row; it does not retrieve order features or a historical review.",
    "survey_trigger_time": "Required local prediction timestamp T: the earlier of delivery and the supplied promise. If delivery has not occurred by the promise, use the promise; both T and the promise must be supplied.",
    "approval_lag_hours": "Hours from purchase to approval. The approval timestamp must be known at or before T; leave missing if approval is unknown. For supplied V1 rows, approval after T is masked.",
    "carrier_handover_days": "Fractional days from approval to carrier handover. Both source timestamps must be known at or before T; leave missing if either is unknown. Supplied rows are masked when either event is after T.",
    "carrier_transit_days": "Fractional days from carrier handover to customer delivery. Requires delivery observed by T and a carrier timestamp known at or before T; otherwise missing. A carrier event after T is masked for supplied rows.",
    "seller_prior_orders": "Your purchase-time count of orders assigned to the representative seller strictly before this purchase. Supply it from your operations records; it is not reconstructed from frozen delivery or review events.",
    "seller_tenure_days": "Your purchase-time elapsed days from the representative seller's first recorded purchase, not account-registration age. Supply it from your operations records; it must be 0 when seller_prior_orders is 0.",
    "order_approved_at": "Local approval timestamp, required when approval_lag_hours or carrier_handover_days is populated. If approval is unknown, leave it and those derived values missing. Approval after T masks both values.",
    "order_delivered_carrier_date": "Local carrier-handover timestamp, required when carrier_shipped is 1 or a handover/transit duration is populated. For an unknown event leave it blank and set carrier_shipped to 0; events after T mask the related durations.",
}


@st.cache_resource(show_spinner="Loading the fitted classification pipeline…")
def _pipeline(version, manifest_text):
    from deployment.classification_pipeline_runtime import load_pipeline
    return load_pipeline(ASSETS, version=version)


def get_pipeline(version):
    text = (ASSETS / "manifest.json").read_text()
    return _pipeline(version, text), hashlib.sha256(text.encode()).hexdigest()


def _values_with_history(pipeline, frame, *, already_asof=False):
    history = pipeline.history_features(frame, already_asof=already_asof)
    values = []
    for position in range(len(frame)):
        value = {**frame.iloc[position].to_dict(), **history.iloc[position].to_dict()}
        value["delivery_vs_seller_prior"] = value["actual_delivery_time"] - value["seller_prior_mean_delivery_days"]
        values.append(value)
    return values


def explain_historical(version, matrix_row, order_id):
    """Use the original prepared row, with the exported column mapping."""
    pipeline, _ = get_pipeline(version)
    explanation = explain_forest(pipeline.forest, matrix_row,
        feature_names=pipeline.feature_names, source_features=pipeline.source_features,
        feature_groups=pipeline.feature_groups)
    examples = pipeline.example_inputs()
    selected = examples.loc[examples.order_id.eq(order_id)].reset_index(drop=True)
    if len(selected) != 1:
        raise ValueError("The historical explanation input is unavailable.")
    return explanation, _values_with_history(pipeline, selected, already_asof=True)[0]


def render(*, version, spec, panel_metrics):
    try:
        pipeline, package_id = get_pipeline(version)
        package_spec = pipeline.manifest["versions"][version]
        if (package_spec["source_model_sha256"] != spec["model_sha256"]
                or package_spec["selection_sha256"] != spec["selection_sha256"]
                or package_spec["capacity_threshold"] != spec["capacity_threshold"]):
            raise ValueError("The inference package and selected model specification differ.")
        examples = pipeline.example_inputs(public_valid_only=True)
    except (OSError, ValueError, KeyError, ImportError) as error:
        inputs_ui.clear_result("classification_" + version)
        st.warning(f"The complete classification inference package is unavailable: {error}")
        return
    st.info("Fill in the blank form with your order details at T, or upload an order table (CSV), to estimate the risk of a 1–2 star review.")
    st.caption("Supplied inputs enforce event availability by T. Seller prior-order count and tenure must be supplied from your operations data; outcome-based history uses only the frozen training events. This historical model does not establish accuracy on today's orders.")
    task = "classification_" + version
    frame = inputs_ui.collect_orders(task=task, examples=examples, meanings=MEANINGS,
                                    validator=pipeline.validate, recalculate=pipeline.recalculate_editor,
                                    seed_identity=package_id)
    if frame is None:
        return
    signature = inputs_ui.identity(frame, spec["model_sha256"] + package_id)
    identity_key, result_key = f"new_{task}_identity", f"new_{task}_result"
    if st.session_state.get(identity_key) != signature:
        inputs_ui.clear_result(task)
        st.session_state[identity_key] = signature
    if st.button("Run new-order risk predictions", type="primary", key=f"new_{task}_run"):
        inputs_ui.clear_result(task)
        st.session_state[identity_key] = signature
        try:
            with st.spinner("Preparing new order features and evaluating the fitted forest…"):
                matrix = pipeline.transform(frame)
                positive = np.flatnonzero(pipeline.forest.classes_ == 1)
                if len(positive) != 1:
                    raise ValueError("The fitted forest must contain the low-review class.")
                scores = pipeline.forest.predict_proba(matrix)[:, positive[0]]
                explanation = explain_forest(pipeline.forest, matrix,
                    feature_names=pipeline.feature_names, source_features=pipeline.source_features,
                    feature_groups=pipeline.feature_groups)
                if not np.allclose(scores, explanation["scores"], atol=1e-12, rtol=0):
                    raise ValueError("The explanation does not reconstruct the fitted risk scores.")
                threshold = float(spec["capacity_threshold"])
                results = pd.DataFrame({"order_id": frame.order_id.to_numpy(), "version": version,
                    "risk_score": scores, "validation_threshold": threshold,
                    "above_threshold": scores >= threshold, "model_sha256": spec["model_sha256"],
                    "selection_sha256": spec["selection_sha256"], "pipeline_manifest_sha256": package_id})
                values = _values_with_history(pipeline, frame)
            st.session_state[result_key] = {"identity": signature, "predictions": results,
                                           "explanation": explanation, "input_values": values}
        except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as error:
            inputs_ui.clear_result(task)
            st.error(f"Unable to predict these orders: {error}")
    result = st.session_state.get(result_key)
    if not result or result["identity"] != signature:
        return
    predictions = result["predictions"]
    st.subheader("New-order low-review risk")
    st.dataframe(predictions[["order_id", "risk_score", "validation_threshold", "above_threshold"]],
                 hide_index=True, use_container_width=True)
    selected = st.selectbox("Inspect a risk prediction", predictions.order_id.tolist(), key=f"new_{task}_selected")
    position = predictions.order_id.tolist().index(selected)
    row = predictions.iloc[position]
    panel_metrics([
        ("Low-review risk score", f"{row.risk_score:.3f}", "Raw forest score, not a calibrated probability."),
        ("Frozen validation threshold", f"{row.validation_threshold:.4f}", "This model's original validation threshold."),
    ], key=task + "_result")
    (st.warning if row.above_threshold else st.info)(
        "Above the frozen validation threshold." if row.above_threshold else "Below the frozen validation threshold. A low review is still possible.")
    st.caption("A fixed threshold is not a top-20% ranking and does not guarantee a 20% contact rate. No historical review is read or revealed for these supplied orders.")
    st.download_button("Download new risk predictions", safe_csv(predictions),
                       file_name=f"new_low_review_{version}.csv", mime="text/csv", key=f"new_{task}_download")
    render_explanation(result["explanation"], position, key=task,
                       input_values=result["input_values"][position])
