"""New-order delivery forecasts from the existing fitted regression bundle."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from deployment import order_input_ui as inputs_ui
from deployment import regression_inputs
from deployment.input_help import INPUT_MEANINGS


def render(*, examples, manifest, deployment_id, load_bundle, panel_metrics):
    st.info("New-order forecast · Supply purchase-time features. This model was fitted on historical Olist data; predictions for new periods do not establish accuracy on today's marketplace.")
    frame = inputs_ui.collect_orders(
        task="delivery", examples=examples, meanings=INPUT_MEANINGS,
        validator=lambda rows: regression_inputs.validate_orders(rows, manifest),
        recalculate=regression_inputs.derive_editor_fields,
        seed_identity=deployment_id,
    )
    if frame is None:
        return
    signature = inputs_ui.identity(frame, deployment_id)
    if st.session_state.get("new_delivery_identity") != signature:
        inputs_ui.clear_result("delivery")
        st.session_state["new_delivery_identity"] = signature
    if st.button("Run delivery predictions", type="primary", key="new_delivery_run"):
        inputs_ui.clear_result("delivery")
        st.session_state["new_delivery_identity"] = signature
        try:
            with st.spinner("Running the fitted delivery model…"):
                results = regression_inputs.predict_orders(frame, load_bundle(), manifest)
            results["model_name"] = manifest["model_name"]
            results["model_sha256"] = manifest["artifacts"]["regression_model.joblib"]["sha256"]
            results["selection_sha256"] = manifest["selection_sha256"]
            results["deployment_manifest_sha256"] = deployment_id
            st.session_state["new_delivery_result"] = {"identity": signature, "predictions": results}
        except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as error:
            inputs_ui.clear_result("delivery")
            st.error(f"Unable to predict these orders: {error}")
    result = st.session_state.get("new_delivery_result")
    if not result or result["identity"] != signature:
        return
    predictions = result["predictions"]
    st.subheader("2 · New-order delivery predictions")
    st.dataframe(predictions[["order_id", "predicted_days", "quoted_days", "predicted_exceeds_quote", "estimated_arrival"]],
                 hide_index=True, use_container_width=True)
    chosen = st.selectbox("Inspect a delivery prediction", predictions.order_id.tolist(), key="new_delivery_selected")
    row = predictions.set_index("order_id").loc[chosen]
    panel_metrics([
        ("Predicted lead time", f"{row.predicted_days:.1f} days", "The fitted model's purchase-to-delivery point estimate."),
        ("Recorded promise", f"{row.quoted_days:.1f} days", "The supplied delivery promise at purchase."),
    ], key="new_delivery_result")
    st.write(f"Expected arrival · **{pd.Timestamp(row.estimated_arrival):%d %B %Y}**")
    st.caption("These are point estimates. No actual delivery outcome or calibrated interval is available for the supplied rows.")
    st.download_button("Download new delivery predictions", regression_inputs.safe_csv(predictions),
                       file_name="new_delivery_predictions.csv", mime="text/csv", key="new_delivery_download")
