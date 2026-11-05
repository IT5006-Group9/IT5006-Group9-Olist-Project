"""Render a blank/durable one-order draft with friendly native text inputs.

The shared page owns the durable draft, revision, preparation and validation.
This renderer only reads that draft and returns the currently entered strings.
It never loads examples, fills required values, derives fields, runs a model or
writes durable session state. Optional example values appear as placeholders.
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import streamlit as st

from deployment.order_input_labels import field_info, group_fields


def _text(value):
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return ""
    return str(value)


def _placeholder(info, field, examples):
    example = examples.get(field) if examples is not None else None
    if pd.api.types.is_scalar(example) and not pd.isna(example) and str(example).strip():
        return "e.g. " + str(example)
    return {
        "datetime": "YYYY-MM-DD HH:MM:SS",
        "flag": "0 = No, 1 = Yes",
        "count": "Enter a whole number",
        "number": "Enter a number",
        "text": "Enter the value for your order",
    }[info["kind"]]


def render_blank_order(task, draft, meanings, revision, fields_grouped=None,
                       derived_fields=(), example_values=None):
    """Return one row in the unchanged raw-column order, with string values.

    ``draft`` must be the caller's one-row blank/durable DataFrame. The caller
    must explicitly pass derived keys to hide; their existing draft values are
    retained for caller-owned preparation. ``fields_grouped`` accepts the
    ``group_fields`` list of pairs or an ordered mapping of group to raw keys.
    Examples may be a mapping, Series or one-row DataFrame, for grey placeholder
    text only; they never become input defaults or returned values.
    """
    if not isinstance(draft, pd.DataFrame) or len(draft) != 1 or not draft.columns.is_unique:
        raise ValueError("The order form requires one draft row with unique raw input columns.")
    columns = list(draft.columns)
    hidden = set(derived_fields or ())
    groups = group_fields(task, columns) if fields_grouped is None else fields_grouped
    groups = list(groups.items()) if isinstance(groups, Mapping) else list(groups)
    visible = [(group, [field for field in fields if field not in hidden]) for group, fields in groups]
    visible = [(group, fields) for group, fields in visible if fields]
    rendered = [field for _, fields in visible for field in fields]
    required = set(columns) - hidden
    if len(rendered) != len(set(rendered)) or set(rendered) != required:
        raise ValueError("Form groups must cover each manual draft field exactly once.")
    if isinstance(example_values, pd.DataFrame):
        if len(example_values) != 1:
            raise ValueError("Placeholder examples must contain exactly one row.")
        example_values = example_values.iloc[0]
    values = {field: _text(draft.iloc[0][field]) for field in columns}
    for group_index, (group, fields) in enumerate(visible):
        with st.expander(str(group), expanded=group_index == 0):
            # The existing opt-in mobile rules stack these two columns only.
            with st.container(key=f"olist_split_{task}_order_form_{revision}_{group_index}"):
                left, right = st.columns(2)
                for index, field in enumerate(fields):
                    info = field_info(task, field)
                    meaning = meanings.get(field, "Supply the value for this order.")
                    help_text = f"{meaning}\n\nTable column: {field}"
                    with (left if index % 2 == 0 else right):
                        values[field] = st.text_input(
                            info["label"], value=values[field], help=help_text,
                            placeholder=_placeholder(info, field, example_values),
                            key=f"{task}_field_{field}_{revision}",
                        )
    result = pd.DataFrame([[values[field] for field in columns]], columns=columns, dtype=object)
    result.attrs = dict(draft.attrs)
    return result
