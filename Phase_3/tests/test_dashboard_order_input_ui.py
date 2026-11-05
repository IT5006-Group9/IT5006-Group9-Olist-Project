"""CSV parsing, input identities and real editor lifecycle, without models."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

PHASE = Path(__file__).resolve().parents[1]
if str(PHASE) not in sys.path:
    sys.path.insert(0, str(PHASE))

from deployment.order_input_ui import csv_frame, identity  # noqa: E402


class OrderCSVParserTests(unittest.TestCase):
    def test_utf8_bom_preserves_leading_zero_identifiers_and_literal_na(self):
        content = ("\ufefforder_id,zip,city,note\r\n"
                   "000001,00123,São Paulo,NA\r\n"
                   "000002,00007,杭州,\r\n").encode("utf-8")
        frame = csv_frame(io.BytesIO(content))
        self.assertEqual(frame.columns.tolist(), ["order_id", "zip", "city", "note"])
        self.assertEqual(frame.order_id.tolist(), ["000001", "000002"])
        self.assertEqual(frame.zip.tolist(), ["00123", "00007"])
        self.assertEqual(frame.city.tolist(), ["São Paulo", "杭州"])
        self.assertEqual(frame.note.tolist(), ["NA", ""])
        self.assertEqual(frame.attrs["csv_row_numbers"], [2, 3])

    def test_duplicate_headers_and_absent_header_are_rejected(self):
        for content in (b"id,id\n001,002\n", b"\n001,002\n"):
            with self.subTest(csv=content), self.assertRaisesRegex(ValueError, "headers.*unique"):
                csv_frame(content)

    def test_empty_and_headers_only_uploads_are_rejected_but_blank_cells_are_data(self):
        for content in (b"", b"\xef\xbb\xbf", b"id,note\n", b"id,note\n\n\n"):
            with self.subTest(csv=content), self.assertRaises(ValueError):
                csv_frame(content)
        frame = csv_frame(b"id,note\n,\n")
        self.assertEqual(frame.shape, (1, 2))
        self.assertEqual(frame.iloc[0].tolist(), ["", ""])
        self.assertEqual(frame.attrs["csv_row_numbers"], [2])

    def test_wrong_cell_count_and_malformed_quotes_report_readable_errors(self):
        for content, count in ((b"id,note\n001\n", 1), (b"id,note\n001,note,extra\n", 3)):
            with self.subTest(csv=content), self.assertRaisesRegex(ValueError, f"CSV row 2: expected 2 cells, received {count}"):
                csv_frame(content)
        for content in (b'id,note\n001,"unterminated\n', b'id,note\n"bad"x,value\n'):
            with self.subTest(csv=content), self.assertRaisesRegex(ValueError, "Cannot read UTF-8 CSV"):
                csv_frame(content)

    def test_multiline_cells_and_blank_lines_retain_record_start_line_numbers(self):
        content = b'id,note\n\n001,"first\nsecond"\n\n002,last\n'
        frame = csv_frame(content)
        self.assertEqual(frame.id.tolist(), ["001", "002"])
        self.assertEqual(frame.note.tolist(), ["first\nsecond", "last"])
        self.assertEqual(frame.attrs["csv_row_numbers"], [3, 6])
        with self.assertRaisesRegex(ValueError, "CSV row 3: expected 2 cells, received 3"):
            csv_frame(b'id,note\n\n"first\nsecond",note,extra\n')

    def test_row_and_byte_limits_include_the_exact_boundary(self):
        content = b"id\n001\n002\n"
        self.assertEqual(len(csv_frame(content, max_rows=2)), 2)
        with self.assertRaisesRegex(ValueError, "at most 2 order rows"):
            csv_frame(content + b"003\n", max_rows=2)
        hundred = ("id\n" + "\n".join(f"{row:03d}" for row in range(100)) + "\n").encode()
        self.assertEqual(len(csv_frame(hundred)), 100)
        with self.assertRaisesRegex(ValueError, "at most 100 order rows"):
            csv_frame(hundred + b"100\n")
        tiny = b"id\nx\n"
        self.assertEqual(len(csv_frame(tiny, max_bytes=len(tiny))), 1)
        with self.assertRaisesRegex(ValueError, "no larger"):
            csv_frame(tiny, max_bytes=len(tiny) - 1)
        with self.assertRaisesRegex(ValueError, "no larger than 2 MiB"):
            csv_frame(b"x" * (2 * 1024 * 1024 + 1))

    def test_parser_is_generic_text_only_and_does_not_evaluate_formula_like_values(self):
        frame = csv_frame(bytearray(b'id,text\n001,=1+1\n002,@SUM(1)\n'))
        self.assertEqual(frame.text.tolist(), ["=1+1", "@SUM(1)"])
        self.assertTrue(all(isinstance(value, str) for value in frame.to_numpy().ravel()))
        # Formula rejection belongs to the task validator, after this parser.
        # No regression/classification feature contract is required to parse.
        for content in (b"id\n\xff\n", "id\n001\n", None):
            with self.subTest(value=content), self.assertRaises(ValueError):
                csv_frame(content)


class OrderIdentityTests(unittest.TestCase):
    def test_identity_tracks_values_row_order_model_and_nanosecond_timestamps(self):
        frame = pd.DataFrame({"order_id": ["new-001", "new-002"], "price": [10.0, 20.0],
                              "purchase_ts": [pd.Timestamp("2026-01-01 00:00:00.000000001"),
                                              pd.Timestamp("2026-01-02 00:00:00")]})
        original = frame.copy(deep=True)
        first = identity(frame, "model-sha-A")
        self.assertRegex(first, r"^[0-9a-f]{64}$")
        self.assertEqual(first, identity(frame.copy(deep=True), "model-sha-A"))
        self.assertEqual(first, identity(frame.set_axis([10, 20]), "model-sha-A"))
        changed = frame.copy(deep=True)
        changed.loc[0, "price"] = 11
        self.assertNotEqual(first, identity(changed, "model-sha-A"))
        changed = frame.copy(deep=True)
        changed.loc[0, "purchase_ts"] += pd.Timedelta(1, unit="ns")
        self.assertNotEqual(first, identity(changed, "model-sha-A"))
        self.assertNotEqual(first, identity(frame.iloc[::-1], "model-sha-A"))
        self.assertNotEqual(first, identity(frame, "model-sha-B"))
        pd.testing.assert_frame_equal(frame, original)


EDITOR_APP = """
import sys
import pandas as pd
import streamlit as st
sys.path.insert(0, PHASE_DIRECTORY)
from deployment.order_input_ui import collect_orders, clear_result, identity

def clear_scores():
    clear_result(TASK)

page = st.radio('Page', ['Orders', 'Other page'], key='test_page', on_change=clear_scores)
if page == 'Orders':
    source = st.radio('Order source', ['New order', 'Historical'], key='test_source', on_change=clear_scores)
    if source == 'New order':
        examples = pd.DataFrame({'order_id': ['reference-first', 'reference-second'],
                                 'price': [10., 25.], 'category': ['books', 'toys']})
        def validate(rows):
            typed = rows.copy(deep=True)
            typed['price'] = pd.to_numeric(typed.price, errors='raise')
            if typed.price.lt(0).any():
                raise ValueError('price must be nonnegative')
            return typed
        frame = collect_orders(task=TASK, examples=examples,
            meanings={'order_id': 'New order identifier', 'price': 'Item price', 'category': 'Category'},
            validator=validate, seed_identity='frozen-package-a')
        if frame is not None:
            current = identity(frame, 'frozen-model')
            key = 'new_' + TASK + '_identity'
            if st.session_state.get(key) != current:
                clear_result(TASK)
                st.session_state[key] = current
            if st.button('Calculate test result'):
                st.session_state['new_' + TASK + '_result'] = frame.copy(deep=True)
            if 'new_' + TASK + '_result' in st.session_state:
                st.success('Calculated result is available')
    else:
        st.write('Historical order controls')
else:
    st.write('Another page')
"""


class EditorDraftLifecycleTests(unittest.TestCase):
    def launch(self, task):
        source = ("PHASE_DIRECTORY = " + repr(str(PHASE)) + "\nTASK = " + repr(task)
                  + "\n" + EDITOR_APP)
        app = AppTest.from_string(source).run(timeout=30)
        self.healthy(app)
        return app

    def healthy(self, app, *, invalid=False):
        self.assertEqual([error.message for error in app.exception], [])
        if not invalid:
            self.assertEqual([error.value for error in app.error], [])

    def editor_details(self, app):
        return next(table.value for table in app.dataframe if 'Field' in table.value)

    def preview(self, app):
        return next(table.value for table in app.dataframe if 'order_id' in table.value)

    def edit(self, app, task, changes, *, invalid=False):
        # Exercise Streamlit's actual delta state and widget cleanup, rather
        # than mocking data_editor and accidentally preserving values in a mock.
        details = self.editor_details(app)
        revision = app.session_state[f'new_{task}_revision']
        key = f'new_{task}_editor_{revision}'
        old = app.session_state[key]
        rows = {int(row): dict(value) for row, value in old['edited_rows'].items()}
        for field, value in changes.items():
            row = details.Field.tolist().index(field)
            rows[row] = {'Value': str(value)}
        app.session_state[key] = {'edited_rows': rows, 'added_rows': [], 'deleted_rows': []}
        app.run(timeout=30)
        self.healthy(app, invalid=invalid)

    def calculate(self, app, task):
        next(button for button in app.button if button.label == 'Calculate test result').click().run(timeout=30)
        self.healthy(app)
        self.assertIn(f'new_{task}_result', app.session_state)

    def test_second_reference_draft_survives_method_source_and_page_cleanup_but_scores_do_not(self):
        for task in ('delivery', 'classification_v1'):
            with self.subTest(task=task):
                app = self.launch(task)
                app.selectbox(key=f'new_{task}_reference').set_value('reference-second').run(timeout=30)
                self.edit(app, task, {'order_id': 'my-new-order', 'category': 'my-category'})
                # A normal rerun must also keep a stable grid base so edits do
                # not disappear before any navigation occurs.
                app.run(timeout=30)
                self.assertEqual(self.preview(app).iloc[0].to_dict(),
                                 {'order_id': 'my-new-order', 'price': 25., 'category': 'my-category'})
                transitions = (
                    ('method', f'new_{task}_method', 'Upload CSV', 'Edit one order'),
                    ('source', 'test_source', 'Historical', 'New order'),
                    ('page', 'test_page', 'Other page', 'Orders'),
                )
                for name, key, away, back in transitions:
                    with self.subTest(task=task, transition=name):
                        self.calculate(app, task)
                        app.radio(key=key).set_value(away).run(timeout=30)
                        self.healthy(app)
                        self.assertNotIn(f'new_{task}_result', app.session_state)
                        app.radio(key=key).set_value(back).run(timeout=30)
                        self.healthy(app)
                        self.assertEqual(app.selectbox(key=f'new_{task}_reference').value, 'reference-second')
                        self.assertEqual(self.preview(app).iloc[0].to_dict(),
                                         {'order_id': 'my-new-order', 'price': 25., 'category': 'my-category'})
                        self.assertNotIn(f'new_{task}_result', app.session_state)
                        self.assertFalse(any(message.value == 'Calculated result is available' for message in app.success))
                # A later edit must replace the restored delta, not replay the
                # old editor base or accidentally start from the first example.
                self.edit(app, task, {'category': 'revised-category'})
                self.assertEqual(self.preview(app).category.tolist(), ['revised-category'])

    def test_invalid_draft_is_preserved_until_corrected_and_reference_change_explicitly_replaces_it(self):
        for task in ('delivery', 'classification_v1'):
            with self.subTest(task=task):
                app = self.launch(task)
                self.calculate(app, task)
                self.edit(app, task, {'order_id': 'needs-correction', 'price': '-1'}, invalid=True)
                self.assertNotIn(f'new_{task}_result', app.session_state)
                self.assertTrue(any('price must be nonnegative' in error.value for error in app.error))
                app.radio(key=f'new_{task}_method').set_value('Upload CSV').run(timeout=30)
                app.radio(key=f'new_{task}_method').set_value('Edit one order').run(timeout=30)
                self.healthy(app, invalid=True)
                details = self.editor_details(app).set_index('Field')
                self.assertEqual(details.loc['order_id', 'Value'], 'needs-correction')
                self.assertEqual(details.loc['price', 'Value'], '-1')
                self.assertNotIn('Calculate test result', [button.label for button in app.button])
                self.edit(app, task, {'price': '15'})
                self.assertEqual(self.preview(app).order_id.tolist(), ['needs-correction'])
                self.assertEqual(self.preview(app).price.tolist(), [15])
                self.calculate(app, task)
                app.selectbox(key=f'new_{task}_reference').set_value('reference-second').run(timeout=30)
                self.healthy(app)
                self.assertEqual(self.preview(app).iloc[0].to_dict(),
                                 {'order_id': f'new-{task}-001', 'price': 25., 'category': 'toys'})
                self.assertNotIn(f'new_{task}_result', app.session_state)


if __name__ == "__main__":
    unittest.main()
