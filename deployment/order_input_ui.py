"""Session-only editors and bounded CSV input for feature-ready new orders."""
from __future__ import annotations

import csv
import hashlib
import io
import re

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


def invalidate_order(task):
    clear_result(task)
    for suffix in ("checked", "check_error", "csv_checked", "csv_check_error"):
        st.session_state.pop(f"new_{task}_{suffix}", None)


def _strings(frame):
    return frame.reset_index(drop=True).map(_value)


def _blank(columns):
    return pd.DataFrame([[""] * len(columns)], columns=columns, dtype=object)


def _untouched_legacy_example(task, draft, examples):
    legacy = st.session_state.get(f"new_{task}_seed")
    if not legacy or len(legacy) != 3:
        return False
    match = examples.loc[examples.order_id.eq(legacy[1])]
    if len(match) != 1:
        return False
    expected = match.copy().reset_index(drop=True)
    expected["order_id"] = f"new-{task}-001"
    return _strings(draft).equals(_strings(expected))


def _form_error(task, error, columns):
    from deployment.order_input_labels import field_label
    message = str(error)
    for field in sorted(columns, key=len, reverse=True):
        message = re.sub(r"\b" + re.escape(field) + r"\b", field_label(task, field), message)
    return message


def collect_orders(*, task, examples, meanings, validator, recalculate=None, seed_identity=""):
    """Collect a genuinely blank order form or a complete, bounded CSV batch.

    Example values are read-only guidance. Checking is an explicit action for
    both input methods; only unchanged, checked inputs can reach prediction.
    """
    from deployment.order_form import render_blank_order
    from deployment.order_form_prepare import derived_fields, prepare_order
    from deployment.order_input_labels import field_label, group_fields

    canonical = [name for _, fields in group_fields(task) for name in fields if name in examples.columns]
    ordered = canonical + [name for name in examples.columns if name not in canonical]
    form_groups = group_fields(task, ordered)
    st.subheader("Enter your order details")
    st.write("Fill in one order from a blank form, or upload a completed order table. The model needs order, product and delivery details; an order ID alone is not enough.")
    methods = ["Fill in one order", "Upload an order table (CSV)"]
    method_key = f"new_{task}_method"
    legacy_methods = {"Edit one order": methods[0], "Enter one order": methods[0], "Upload CSV": methods[1]}
    if st.session_state.get(method_key) in legacy_methods:
        st.session_state[method_key] = legacy_methods[st.session_state[method_key]]
        invalidate_order(task)
    if st.session_state.get(method_key, methods[0]) not in methods:
        st.session_state[method_key] = methods[0]
        invalidate_order(task)
    method_column, form_actions = st.columns([4, 1], vertical_alignment="bottom")
    with method_column:
        method = st.radio("Input method", methods, horizontal=True, key=method_key,
                          index=methods.index(st.session_state.get(method_key, methods[0])),
                          on_change=invalidate_order, args=(task,))
    example = examples.iloc[[0]].copy().reset_index(drop=True)
    example["order_id"] = "example-order-001"

    if method == methods[1]:
        st.write("A CSV is a table file you can edit in Excel. Download the blank table below, fill one order per row, keep its column headings, and save it as CSV UTF-8 before uploading.")
        st.caption("Upload your file → Check order details → Run predictions. Uploading a file does not start a prediction.")
        with st.expander("Order table templates and field guide"):
            left, right = st.columns(2)
            left.download_button("Download blank order table", safe_csv(example.iloc[:0]),
                                 file_name=f"{task}_new_orders_template.csv", mime="text/csv", key=f"new_{task}_template")
            right.download_button("Download example order table", safe_csv(example),
                                  file_name=f"{task}_new_order_example.csv", mime="text/csv", key=f"new_{task}_example")
            catalogue = pd.DataFrame({"Field": examples.columns,
                                      "Meaning / units": [meanings.get(name, "Supply the value for this order.") for name in examples.columns]})
            with st.container(height=360, border=False):
                st.markdown(catalogue.to_html(index=False, escape=True, border=0, classes="olist-input-table"), unsafe_allow_html=True)
            st.caption("All template headers are required. Leave a value blank only where the field allows missing information. Review labels and outcomes unavailable at the task’s prediction point are rejected. Limit: 100 orders and 2 MiB. Uploaded values are checked as supplied; they are not automatically recalculated.")
        uploaded = st.file_uploader("Upload your completed order table (CSV)", type=["csv"],
                                    key=f"new_{task}_upload", on_change=invalidate_order, args=(task,))
        check_clicked = st.button("Check order details", type="primary", key=f"new_{task}_csv_check",
                                  disabled=uploaded is None)
        st.caption("This checks the uploaded fields and values. After they pass, use the Run button below to start the predictions.")
        if uploaded is None:
            invalidate_order(task)
            st.info("Upload your filled CSV table to check the orders before prediction.")
            return None
        content = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
        upload_identity = hashlib.sha256(seed_identity.encode() + b"\n" + content).hexdigest()
        checked_key, error_key = f"new_{task}_csv_checked", f"new_{task}_csv_check_error"
        if st.session_state.get(checked_key, {}).get("identity") != upload_identity:
            clear_result(task)
            st.session_state.pop(checked_key, None)
        if st.session_state.get(error_key, {}).get("identity") != upload_identity:
            st.session_state.pop(error_key, None)
        if check_clicked:
            invalidate_order(task)
            try:
                validated = validator(csv_frame(content))
                st.session_state[checked_key] = {"identity": upload_identity, "frame": validated.copy(deep=True)}
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                st.session_state[error_key] = {"identity": upload_identity, "message": str(error)}
        if error_key in st.session_state:
            st.error("Please correct these table fields:\n\n" + st.session_state[error_key]["message"])
            return None
        if checked_key not in st.session_state:
            st.info("File uploaded. Click Check order details to validate it before running predictions.")
            return None
        frame = st.session_state[checked_key]["frame"].copy(deep=True)
    else:
        seed_key, draft_key, revision_key, checked_key = (
            f"new_{task}_{name}" for name in ("form_seed", "draft", "revision", "checked")
        )
        signature = (seed_identity, tuple(examples.columns), "blank-form-v1")
        if st.session_state.get(seed_key) != signature:
            previous = st.session_state.get(draft_key)
            usable = isinstance(previous, pd.DataFrame) and len(previous) == 1 and list(previous.columns) == list(examples.columns)
            keep = usable and not _untouched_legacy_example(task, previous, examples)
            st.session_state[draft_key] = _strings(previous) if keep else _blank(examples.columns)
            st.session_state[seed_key] = signature
            st.session_state[revision_key] = st.session_state.get(revision_key, 0) + 1
            invalidate_order(task)
        has_entries = st.session_state[draft_key].map(lambda value: bool(_value(value).strip())).any().any()
        start = "Your current draft is shown below." if has_entries else "Start with the blank form below."
        st.caption(start + " Complete the sections, then click Check order details. Grey example text is guidance only and is not entered for you.")
        with st.expander("View a completed example"):
            st.caption("This filled order shows how to write the values. Opening it does not fill your form. Use only values that describe your own order.")
            example_derived = set(derived_fields(task, example))
            fields = [name for _, members in form_groups for name in members if name not in example_derived]
            guide = pd.DataFrame({"Order detail": [field_label(task, name) for name in fields],
                                  "Example value": [_value(example.iloc[0][name]) for name in fields],
                                  "Meaning / units": [meanings.get(name, "Supply the value for this order.") for name in fields]})
            with st.container(height=380, border=False, key=f"olist_{task}_completed_example"):
                st.markdown(guide.to_html(index=False, escape=True, border=0, classes="olist-input-table"), unsafe_allow_html=True)
            st.caption("Calendar terms, ratios and other fields that can be calculated from your entries are prepared when you check the form.")
        derivable = set(derived_fields(task, st.session_state[draft_key]))
        automatic = [name for name in examples.columns if name in derivable]
        draft = render_blank_order(task, st.session_state[draft_key], meanings,
                                   st.session_state[revision_key], form_groups,
                                   derived_fields=automatic, example_values=example)
        st.session_state[draft_key] = draft.copy(deep=True)
        with form_actions:
            if st.button("Clear form", key=f"new_{task}_reset", width="stretch",
                         help="Clear your entered fields and any prediction for this form.",
                         disabled=not draft.map(lambda value: bool(_value(value).strip())).any().any()):
                st.session_state[draft_key] = _blank(examples.columns)
                st.session_state[revision_key] += 1
                invalidate_order(task)
                st.rerun()
        raw_identity = identity(draft, seed_identity)
        if st.session_state.get(checked_key) != raw_identity:
            clear_result(task)
            st.session_state.pop(checked_key, None)
        error_key = f"new_{task}_check_error"
        if st.session_state.get(error_key, {}).get("identity") != raw_identity:
            st.session_state.pop(error_key, None)
        has_manual_entries = draft.drop(columns=automatic).map(
            lambda value: bool(_value(value).strip())
        ).any().any()
        if st.button("Check order details", type="primary", key=f"new_{task}_check",
                     disabled=not has_manual_entries):
            invalidate_order(task)
            try:
                prepared = prepare_order(task, draft, recalculate)
                validator(prepared)
                normalized = _strings(prepared)
                st.session_state[draft_key] = normalized
                st.session_state[revision_key] += 1
                st.session_state[checked_key] = identity(normalized, seed_identity)
                st.rerun()
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                st.session_state[error_key] = {"identity": raw_identity,
                                              "message": _form_error(task, error, examples.columns)}
        st.caption("This checks the required fields and calculates values that follow from your entries. Product measurements, seller history and multi-item summaries must come from your own order records.")
        if error_key in st.session_state:
            st.error("Please check these details:\n\n" + st.session_state[error_key]["message"])
        if st.session_state.get(checked_key) != raw_identity:
            st.info("Fill in the sections and click Check order details to enable the prediction."
                    if has_manual_entries else "Enter your order details to enable Check order details.")
            return None
        try:
            frame = validator(draft)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            invalidate_order(task)
            st.error("Please check these details:\n\n" + _form_error(task, error, examples.columns))
            return None
        if automatic:
            with st.expander("Values calculated from your entries"):
                values = pd.DataFrame({"Order detail": [field_label(task, name) for name in automatic],
                                       "Value": [_value(draft.iloc[0][name]) for name in automatic]})
                st.dataframe(values, hide_index=True, use_container_width=True)
    st.success(f"{len(frame)} order(s) checked. Click Run below to calculate the prediction.")
    with st.expander("Preview the checked order data"):
        st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption("Your order data stays in this app session and is not added to the project dataset. Entering or checking details does not run a prediction.")
    return frame
