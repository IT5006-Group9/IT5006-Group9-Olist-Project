"""Session-only editors and bounded CSV input for feature-ready new orders."""
from __future__ import annotations

import csv
import hashlib
import io

import pandas as pd
import streamlit as st

from deployment.regression_inputs import safe_csv


def clear_result(task):
    st.session_state.pop(f"new_{task}_result", None)
    st.session_state.pop(f"new_{task}_identity", None)


def identity(frame, model_identity):
    content = frame.to_csv(index=False)
    return hashlib.sha256((model_identity + "\n" + content).encode()).hexdigest()


def csv_frame(upload, *, max_rows=100, max_bytes=2 * 1024 * 1024):
    """Keep identifier strings and original CSV line numbers for validation."""
    content = upload.getvalue() if hasattr(upload, "getvalue") else upload
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("Upload a UTF-8 CSV file.")
    if len(content) > max_bytes:
        raise ValueError("The CSV must be no larger than 2 MiB.")
    try:
        reader = csv.reader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        header = next(reader)
        if not header or len(set(header)) != len(header):
            raise ValueError("CSV headers must be present and unique.")
        records, numbers = [], []
        while True:
            start_line = reader.line_num + 1
            try:
                row = next(reader)
            except StopIteration:
                break
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV row {start_line}: expected {len(header)} cells, received {len(row)}.")
            records.append(row)
            numbers.append(start_line)
            if len(records) > max_rows:
                raise ValueError(f"Upload at most {max_rows} order rows.")
    except (UnicodeError, csv.Error, StopIteration) as error:
        raise ValueError(f"Cannot read UTF-8 CSV: {error or 'the file is empty'}.") from error
    if not records:
        raise ValueError("The template contains headers only. Add at least one order row.")
    result = pd.DataFrame(records, columns=header)
    result.attrs["csv_row_numbers"] = numbers
    return result


def _value(value):
    return "" if pd.isna(value) else str(value)


def collect_orders(*, task, examples, meanings, validator, recalculate=None, seed_identity=""):
    """Return validated rows; editing is separate from the explicit Run action."""
    st.subheader("1 · Supply new order features")
    st.write("Use a complete feature row at the task’s prediction time. An order ID alone cannot supply the order, product, route and history information the fitted model requires.")
    method = st.radio("Input method", ["Edit one order", "Upload CSV"], horizontal=True,
                      key=f"new_{task}_method", on_change=clear_result, args=(task,))
    seed = examples.iloc[[0]].copy().reset_index(drop=True)
    seed["order_id"] = f"new-{task}-001"
    with st.expander("CSV templates and field definitions"):
        left, right = st.columns(2)
        left.download_button("Download empty template", safe_csv(seed.iloc[:0]),
                             file_name=f"{task}_new_orders_template.csv", mime="text/csv", key=f"new_{task}_template")
        right.download_button("Download filled example", safe_csv(seed),
                              file_name=f"{task}_new_order_example.csv", mime="text/csv", key=f"new_{task}_example")
        catalogue = pd.DataFrame({"Field": examples.columns,
                                  "Meaning / units": [meanings.get(name, "Prediction-time input; see the model contract.") for name in examples.columns]})
        st.markdown(catalogue.to_html(index=False, escape=True, border=0,
                                     classes="olist-input-table"), unsafe_allow_html=True)
        st.caption("All template headers are required. Leave a value blank only where the field allows missing information. Review labels and outcomes unavailable at the task’s prediction point are rejected. CSV limit: 100 orders and 2 MiB.")
    if method == "Upload CSV":
        uploaded = st.file_uploader("Upload feature-ready orders", type=["csv"],
                                    key=f"new_{task}_upload", on_change=clear_result, args=(task,))
        if uploaded is None:
            clear_result(task)
            st.info("Download a template, fill the required values, then upload the CSV.")
            return None
        try:
            frame = validator(csv_frame(uploaded))
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            clear_result(task)
            st.error(f"Please correct these input fields:\n\n{error}")
            return None
    else:
        st.caption("The editor starts with recorded feature values and a new order ID. Change the values to describe your order; no historical review is copied. Any history fields and multi-item statistics must describe your supplied order.")
        seed_key, draft_key, revision_key, base_key = (
            f"new_{task}_{name}" for name in ("seed", "draft", "revision", "editor_base")
        )
        reference_key = f"new_{task}_reference"
        references = examples.order_id.tolist()
        previous_seed = st.session_state.get(seed_key)
        # The reference widget disappears on another method/source/page. Restore
        # its choice from durable state before it can reset a second-row draft.
        if (reference_key not in st.session_state and previous_seed
                and previous_seed[0] == seed_identity
                and previous_seed[2] == tuple(examples.columns)
                and previous_seed[1] in references):
            st.session_state[reference_key] = previous_seed[1]
        reference = st.selectbox("Reference example for initial values", references,
                                 format_func=lambda value: f"Reference …{str(value)[-8:]}",
                                 key=reference_key, on_change=clear_result, args=(task,))
        signature = (seed_identity, reference, tuple(examples.columns))
        if st.session_state.get(seed_key) != signature:
            frame = examples.loc[examples.order_id.eq(reference)].iloc[[0]].copy().reset_index(drop=True)
            frame["order_id"] = f"new-{task}-001"
            st.session_state[draft_key] = frame
            st.session_state[seed_key] = signature
            st.session_state[revision_key] = st.session_state.get(revision_key, 0) + 1
            clear_result(task)
        editor_key = f"new_{task}_editor_{st.session_state[revision_key]}"
        # Keep the grid's base stable while Streamlit owns its edit deltas.
        # Re-entry after widget cleanup starts a clean grid from the latest
        # durable draft, including edits that have not passed validation yet.
        if editor_key not in st.session_state or base_key not in st.session_state:
            st.session_state[base_key] = st.session_state[draft_key].copy(deep=True)
        initial = st.session_state[base_key].iloc[0]
        details = pd.DataFrame({"Field": initial.index, "Value": [_value(value) for value in initial],
                                "Meaning / units": [meanings.get(name, "Prediction-time input.") for name in initial.index]})
        edited = st.data_editor(details, hide_index=True, height=540, use_container_width=True,
                                disabled=["Field", "Meaning / units"], num_rows="fixed",
                                column_config={"Field": st.column_config.TextColumn(width="medium"),
                                               "Value": st.column_config.TextColumn("Your value", width="medium"),
                                               "Meaning / units": st.column_config.TextColumn(width="large")},
                                key=editor_key)
        draft = pd.DataFrame([dict(zip(edited.Field, edited.Value))])
        st.session_state[draft_key] = draft.copy(deep=True)
        if recalculate is not None:
            if st.button("Recalculate derived fields", key=f"new_{task}_recalculate"):
                try:
                    st.session_state[draft_key] = recalculate(draft)
                    st.session_state[revision_key] += 1
                    clear_result(task)
                    st.rerun()
                except (ValueError, KeyError, TypeError, OverflowError) as error:
                    clear_result(task)
                    st.error(f"Unable to recalculate: {error}")
                    return None
            st.caption("Use Recalculate after changing dates, prices or routes. It updates fields derivable from the supplied values; it does not invent seller history, item measurements or basket shares. Uploaded CSV values are validated as supplied.")
        try:
            frame = validator(draft)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            clear_result(task)
            st.error(f"Please correct these input fields:\n\n{error}")
            return None
    st.success(f"{len(frame)} order row(s) validated. Click Run to calculate fresh predictions.")
    with st.expander("Preview the validated prediction inputs"):
        st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption("Uploaded or edited orders are processed in this app session and are not added to the repository. No review outcome is available for these inputs.")
    return frame
