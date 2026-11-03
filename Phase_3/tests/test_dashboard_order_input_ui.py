"""Standalone CSV parsing and input identities, without task/model UI flows."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
