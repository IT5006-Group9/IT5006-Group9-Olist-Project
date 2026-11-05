"""V1-only fixed-example classification flows using the exported fitted forest.

The tests never train, load a pickle, or require the classification training
environment. Expected scores come from the portable forest's numeric trees;
guards reject outcome/saved-score reads before the explicit reveal action.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


PHASE = Path(__file__).resolve().parents[1]
ENTRY = PHASE / "dashboard/app.py"
for directory in (PHASE, PHASE / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deployment import classification_page as page  # noqa: E402
from deployment.classification_input_help import CLASSIFICATION_INPUT_MEANINGS  # noqa: E402
from deployment.classification_runtime import PortableForest, load_forest  # noqa: E402


ASSETS = PHASE / "deployment/assets/classification"
MANIFEST = json.loads((ASSETS / "manifest.json").read_text())
ACTUAL_PREDICT = PortableForest.predict_proba


class ClassificationPredictionTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def assert_healthy(self, app, *, allow_error=False):
        self.assertEqual([error.message for error in app.exception], [])
        if not allow_error:
            self.assertEqual([error.value for error in app.error], [])

    def widget(self, widgets, label):
        matches = [widget for widget in widgets if widget.label == label]
        self.assertEqual(len(matches), 1, f"Expected exactly one {label!r} widget")
        return matches[0]

    def metric(self, app, label):
        matches = [metric.value for metric in app.metric if metric.label == label]
        self.assertEqual(len(matches), 1, f"Expected exactly one {label!r} metric")
        return matches[0]

    def navigate(self, app, destination):
        next(widget for widget in app.sidebar.radio if widget.key == "workspace").set_value(
            destination
        ).run(timeout=60)
        self.assert_healthy(app)

    def launch(self, *, expect_model=True):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        self.assert_healthy(app)
        self.navigate(app, "Low-review prediction")
        if expect_model:
            self.assertEqual(app.session_state["clf_version"], "v1")
        self.assertNotIn("Prediction model", [widget.label for widget in app.selectbox])
        return app

    def run_prediction(self, app):
        self.widget(app.button, "Run low-review prediction").click().run(timeout=60)
        self.assert_healthy(app)
        self.assertIn("clf_forecast", app.session_state)
        return app.session_state["clf_forecast"]

    def reveal(self, app):
        self.widget(app.checkbox, "Reveal the historical review").check().run(timeout=60)
        self.assert_healthy(app)

    def assert_cleared(self, app):
        self.assert_healthy(app)
        self.assertNotIn("clf_forecast", app.session_state)
        for key in ("clf_explanation", "clf_explanation_values", "clf_explanation_error"):
            self.assertNotIn(key, app.session_state)
        if "clf_reveal" in app.session_state:
            self.assertFalse(app.session_state["clf_reveal"])
        self.assertNotIn("Historical review", [metric.label for metric in app.metric])
        self.assertNotIn("Low-review risk score", [metric.label for metric in app.metric])
        self.assertFalse(app.checkbox)
        self.assertFalse(any(item.proto.label in ("Download this risk prediction", "Download new risk predictions")
                             for item in app.get("download_button")))

    @contextmanager
    def real_inference_spy(self):
        # autospec preserves the bound self argument, then calls the real trees.
        with patch.object(PortableForest, "predict_proba", autospec=True,
                          side_effect=ACTUAL_PREDICT) as predict:
            yield predict

    def test_final_v1_runs_real_forest_on_exact_row_without_outcomes_or_score_lookup(self):
        original_read = pd.read_parquet
        read_paths = []

        def prediction_only_read(path, *args, **kwargs):
            name = str(path)
            read_paths.append(name)
            if Path(name).name == MANIFEST["outcomes"] or "/predictions/" in name or "scores" in Path(name).stem:
                raise AssertionError("Prediction must not read outcomes or saved scores: " + name)
            return original_read(path, *args, **kwargs)

        with patch.object(pd, "read_parquet", side_effect=prediction_only_read), \
                patch.object(page, "load_outcomes", side_effect=AssertionError("Outcome read before reveal")) as outcomes, \
                self.real_inference_spy() as predict:
            app = self.launch()
            self.assertEqual(predict.call_count, 0, "Opening the page cannot run inference")
            # Version-aware field meanings remain an accurate technical audit
            # inside the folded input catalogue; primary notices expose one model.
            notices = [item.value for group in (app.info, app.warning, app.caption, app.subheader) for item in group]
            self.assertTrue(any("Final model" in text and "Random Forest V1" in text for text in notices))
            self.assertFalse(any(re.search(r"\bstrict\b", text, flags=re.IGNORECASE) for text in notices))
            self.assertTrue(any("historical" in text.lower() or "retrospective" in text.lower() for text in notices))
            self.assertFalse(any("availability limitations" in item.value.lower() for item in app.warning))
            selected = self.widget(app.selectbox, "Historical order").value
            spec = MANIFEST["versions"]["v1"]
            frozen_selection = json.loads((PHASE / spec["selection_file"]).read_text())
            self.assertEqual(spec["capacity_threshold"], frozen_selection["thresholds"]["capacity_threshold"])
            self.assertEqual(spec["selection_sha256"], frozen_selection["selection_sha256"])
            inputs = original_read(ASSETS / spec["inputs"])
            picker = self.widget(app.selectbox, "Historical order")
            self.assertEqual(picker.key, "clf_order")
            self.assertEqual(len(picker.options), len(inputs))
            for number, (label, source) in enumerate(zip(picker.options, inputs.itertuples()), start=1):
                with self.subTest(example=number):
                    self.assertEqual(label, f"Example {number:03d} · {pd.Timestamp(source.survey_trigger_time):%d %b %Y}")
                    self.assertNotIn(source.order_id, label)
                    self.assertNotRegex(label, r"[0-9a-f]{16,}|[→←…]")
            self.assertIn("São Paulo (SP)", self.widget(app.selectbox, "Customer destination").options)
            self.assertIn("Pernambuco (PE)", self.widget(app.selectbox, "Customer destination").options)
            self.assertIn("June 2018", self.widget(app.selectbox, "Trigger month").options)
            self.assertIn("Example 001 · Order details", [title.value for title in app.subheader])
            details = "\n".join(item.value for item in app.markdown if "<table" not in item.value)
            self.assertIn("**Prediction point T:** 01 June 2018, 00:00", details)
            self.assertNotIn("25 April 2018", details, "The selector and summary date are T, not the purchase date")
            self.assertIn("**Seller state:** São Paulo (SP)", details)
            self.assertIn("**Customer destination:** Pernambuco (PE)", details)
            first = inputs.iloc[0]
            for label, value in (("Items", str(int(first.n_items))),
                                 ("Item value", f"R\\$ {first.total_price:,.2f}"),
                                 ("Freight", f"R\\$ {first.total_freight:,.2f}")):
                self.assertIn(f"**{label}:** {value}", details)
            self.assertIn("**Delivered by prediction point T:** No · not yet delivered at T", details)
            ordinal = inputs.order_id.tolist().index(selected)
            matrix = np.load(ASSETS / spec["matrix"], allow_pickle=False)
            forest = load_forest(ASSETS / spec["forest"])
            expected = float(ACTUAL_PREDICT(forest, matrix[[ordinal]])[0, 1])
            result = self.run_prediction(app)
            self.assertEqual(predict.call_count, 1)
            np.testing.assert_array_equal(predict.call_args.args[1], matrix[[ordinal]])
            self.assertEqual(predict.call_args.args[1].shape, (1, spec["n_features"]))
            self.assertAlmostEqual(result["risk_score"], expected, places=14)
            self.assertEqual(result["validation_threshold"], spec["capacity_threshold"])
            self.assertEqual(result["above_threshold"], expected >= spec["capacity_threshold"])
            self.assertEqual(result["version"], "v1")
            self.assertEqual(self.metric(app, "Low-review risk score"), f"{expected:.3f}")
            self.assertEqual(self.metric(app, "Frozen validation threshold"), f"{spec['capacity_threshold']:.4f}")
            self.assertFalse(self.widget(app.checkbox, "Reveal the historical review").value)
            self.assertNotIn("Historical review", [metric.label for metric in app.metric])
            self.assertEqual(len(app.get("download_button")), 1)
            outcomes.assert_not_called()
        self.assertFalse(any(Path(path).name == "strict_inputs.parquet" for path in read_paths))
        self.assertTrue(any(Path(path).name == "inputs.parquet" for path in read_paths))

    def test_outcome_is_loaded_only_after_reveal_and_never_enters_download(self):
        actuals = pd.read_parquet(ASSETS / MANIFEST["outcomes"]).set_index("order_id")
        with patch.object(page, "load_outcomes", wraps=page.load_outcomes) as load_outcomes, \
                patch.object(st, "download_button", wraps=st.download_button) as download, \
                self.real_inference_spy() as predict:
            app = self.launch()
            result = self.run_prediction(app)
            load_outcomes.assert_not_called()
            self.assertEqual(predict.call_count, 1)
            self.reveal(app)
            self.assertEqual(load_outcomes.call_count, 1)
            self.assertEqual(predict.call_count, 1, "Revealing a review cannot rerun the forest")
            actual = actuals.loc[result["order_id"]]
            self.assertEqual(self.metric(app, "Historical review"), f"{int(actual.review_score)} / 5 stars")
            self.assertEqual(self.metric(app, "Observed low review"),
                             "Yes · 1–2 stars" if actual.y else "No · 3–5 stars")
            self.assertTrue(any(pd.Timestamp(actual.review_answer_timestamp).strftime("%d %B %Y, %H:%M")
                                in caption.value for caption in app.caption))
            csv_text = download.call_args.args[1]
            downloaded = pd.read_csv(io.StringIO(csv_text))
            self.assertEqual(downloaded.order_id.tolist(), [result["order_id"]])
            self.assertAlmostEqual(downloaded.risk_score.iloc[0], result["risk_score"])
            self.assertEqual(downloaded.version.tolist(), ["v1"])
            for field in ("model_sha256", "selection_sha256"):
                self.assertEqual(downloaded[field].iloc[0], MANIFEST["versions"]["v1"][field])
            self.assertFalse(set(downloaded.columns) & (page.OUTCOME_COLUMNS | {"y"}))
            # A second explicit Run also closes a previously revealed review.
            self.run_prediction(app)
            self.assertFalse(self.widget(app.checkbox, "Reveal the historical review").value)
            self.assertNotIn("Historical review", [metric.label for metric in app.metric])
            self.assertEqual(load_outcomes.call_count, 1)
            self.assertEqual(predict.call_count, 2)

    def test_matching_filter_and_source_changes_clear_prediction_and_download(self):
        inputs = pd.read_parquet(ASSETS / MANIFEST["versions"]["v1"]["inputs"])
        with self.real_inference_spy() as predict:
            app = self.launch()
            # Pick the last full-sample row: a filtered ordinal must never
            # replace Example 120 with its position inside the smaller cohort.
            chosen = inputs.iloc[-1].order_id
            self.widget(app.selectbox, "Historical order").set_value(chosen).run(timeout=60)
            row = inputs.set_index("order_id").loc[chosen]
            expected_label = f"Example {len(inputs):03d} · {pd.Timestamp(row.survey_trigger_time):%d %b %Y}"
            picker = self.widget(app.selectbox, "Historical order")
            self.assertEqual(picker.options[picker.index], expected_label)
            self.assertEqual(predict.call_count, 0)
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.selectbox, "Trigger month").set_value(row.t_month).run(timeout=60)
            self.assert_cleared(app)
            picker = self.widget(app.selectbox, "Historical order")
            self.assertIn(expected_label, picker.options)
            # Streamlit may reset selection when the filtered options change;
            # the same underlying order must retain its full-sample label.
            picker.set_value(chosen).run(timeout=60)
            self.assert_cleared(app)
            picker = self.widget(app.selectbox, "Historical order")
            self.assertEqual(picker.value, chosen)
            self.assertEqual(picker.options[picker.index], expected_label)
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.selectbox, "Customer destination").set_value(row.customer_state).run(timeout=60)
            self.assert_cleared(app)
            picker = self.widget(app.selectbox, "Historical order")
            self.assertIn(expected_label, picker.options)
            picker.set_value(chosen).run(timeout=60)
            self.assert_cleared(app)
            picker = self.widget(app.selectbox, "Historical order")
            self.assertEqual(picker.value, chosen)
            self.assertEqual(picker.options[picker.index], expected_label)
            self.assertEqual(self.widget(app.selectbox, "Customer destination").value, "RS")
            self.assertLess(len(picker.options), len(inputs))
            self.assertIn("Example 120 · Order details", [title.value for title in app.subheader])
            details = "\n".join(item.value for item in app.markdown if "<table" not in item.value)
            self.assertIn("**Seller state:** Paraná (PR)", details)
            self.assertIn("**Customer destination:** Rio Grande do Sul (RS)", details)
            self.assertIn("**Delivered by prediction point T:** Yes", details)
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.radio, "Order source").set_value("Enter an order or upload a table").run(timeout=60)
            self.assert_cleared(app)
            self.widget(app.radio, "Order source").set_value("Historical examples").run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 3)
            matrix = np.load(ASSETS / MANIFEST["versions"]["v1"]["matrix"], allow_pickle=False)
            for call in predict.call_args_list:
                np.testing.assert_array_equal(call.args[1], matrix[[-1]])

    def test_stale_strict_session_migrates_to_v1_without_running_a_model(self):
        with self.real_inference_spy() as predict:
            app = self.launch()
            app.session_state["clf_version"] = "strict"
            app.session_state["clf_forecast"] = {"version": "strict", "risk_score": .999, "order_id": "stale-strict"}
            app.session_state["clf_reveal"] = True
            for key in ("clf_explanation", "clf_explanation_values", "clf_explanation_error",
                        "new_classification_strict_result", "new_classification_strict_identity"):
                app.session_state[key] = {"stale": "strict"}
            app.run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(app.session_state["clf_version"], "v1")
            for key in ("new_classification_strict_result", "new_classification_strict_identity"):
                self.assertNotIn(key, app.session_state)
            self.assertNotIn("Prediction model", [widget.label for widget in app.selectbox])
            self.assertEqual(predict.call_count, 0)
            result = self.run_prediction(app)
            self.assertEqual(result["version"], "v1")
            self.assertEqual(predict.call_count, 1)
            # An old strict forecast can also coexist with a current v1 marker.
            # The result's own version must prevent this second stale state.
            app.session_state["clf_forecast"] = {"version": "strict", "risk_score": .999, "order_id": "stale-strict"}
            app.run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(app.session_state["clf_version"], "v1")
            self.assertEqual(predict.call_count, 1)

    def test_order_navigation_manual_selection_and_page_leave_never_restore_results(self):
        with self.real_inference_spy() as predict:
            app = self.launch()
            original = self.widget(app.selectbox, "Historical order").value
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.button, "Next →").click().run(timeout=60)
            self.assert_cleared(app)
            self.assertNotEqual(self.widget(app.selectbox, "Historical order").value, original)
            self.widget(app.button, "← Previous").click().run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(self.widget(app.selectbox, "Historical order").value, original)
            self.run_prediction(app)
            self.reveal(app)
            other = next(order for order in pd.read_parquet(ASSETS / "inputs.parquet").order_id if order != original)
            self.widget(app.selectbox, "Historical order").set_value(other).run(timeout=60)
            self.assert_cleared(app)
            self.run_prediction(app)
            self.reveal(app)
            self.navigate(app, "Delivery prediction")
            self.assertNotIn("clf_forecast", app.session_state)
            self.navigate(app, "Low-review prediction")
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 3)

    def test_random_stays_in_filter_and_stale_state_resets_for_new_month(self):
        inputs = pd.read_parquet(ASSETS / "inputs.parquet")
        with self.real_inference_spy() as predict:
            app = self.launch()
            self.widget(app.selectbox, "Trigger month").set_value("2018-06").run(timeout=60)
            self.widget(app.selectbox, "Customer destination").set_value("SP").run(timeout=60)
            before = self.widget(app.selectbox, "Historical order").value
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.button, "Random").click().run(timeout=60)
            self.assert_cleared(app)
            after = self.widget(app.selectbox, "Historical order").value
            self.assertNotEqual(after, before)
            candidates = inputs.loc[inputs.t_month.eq("2018-06") & inputs.customer_state.eq("SP"), "order_id"].tolist()
            self.assertIn(after, candidates)
            # MA has exactly one June example and no July example: a month
            # change must reset the now-invalid state before its widget exists.
            self.widget(app.selectbox, "Customer destination").set_value("MA").run(timeout=60)
            self.assert_cleared(app)
            for label in ("← Previous", "Next →", "Random"):
                self.assertTrue(self.widget(app.button, label).disabled)
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.selectbox, "Trigger month").set_value("2018-07").run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(self.widget(app.selectbox, "Customer destination").value, "All states")
            selected = self.widget(app.selectbox, "Historical order").value
            self.assertEqual(inputs.set_index("order_id").loc[selected, "t_month"], "2018-07")
            self.assertEqual(predict.call_count, 2)

    def test_missing_manifest_and_corrupt_input_assets_fail_gracefully(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.object(page, "ASSETS", base):
                app = self.launch(expect_model=False)
                self.assert_cleared(app)
                self.assertTrue(any("package is unavailable" in warning.value for warning in app.warning))
                shutil.copytree(ASSETS, base, dirs_exist_ok=True)
                (base / "v1_matrix.npy").write_bytes(b"corrupt matrix")
                st.cache_data.clear()
                app = self.launch()
                self.assert_cleared(app)
                self.assertTrue(any("verification failed" in warning.value for warning in app.warning))

    def test_missing_or_corrupt_forest_clears_stale_prediction_without_reveal(self):
        for mode in ("missing", "corrupt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                shutil.copytree(ASSETS, base, dirs_exist_ok=True)
                with patch.object(page, "ASSETS", base):
                    app = self.launch()
                    self.run_prediction(app)
                    self.reveal(app)
                    forest = base / MANIFEST["versions"]["v1"]["forest"]
                    if mode == "missing":
                        forest.unlink()
                    else:
                        forest.write_bytes(b"corrupt forest")
                    st.cache_resource.clear()
                    self.widget(app.button, "Run low-review prediction").click().run(timeout=60)
                    self.assert_healthy(app, allow_error=True)
                    self.assertEqual(len(app.error), 1)
                    self.assertIn("Unable to score this order", app.error[0].value)
                    self.assertNotIn("clf_forecast", app.session_state)
                    self.assertFalse(app.checkbox)
                    self.assertFalse(app.get("download_button"))

    def test_corrupt_strict_only_assets_do_not_affect_the_final_v1_page(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shutil.copytree(ASSETS, base, dirs_exist_ok=True)
            strict = MANIFEST["versions"]["strict"]
            for key in ("inputs", "matrix", "forest"):
                (base / strict[key]).write_bytes(b"unavailable audit-only strict asset")
            with patch.object(page, "ASSETS", base), self.real_inference_spy() as predict:
                app = self.launch()
                result = self.run_prediction(app)
                self.assertEqual(result["version"], "v1")
                self.assertEqual(predict.call_count, 1)

    def test_unavailable_outcome_does_not_break_or_change_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shutil.copytree(ASSETS, base, dirs_exist_ok=True)
            (base / MANIFEST["outcomes"]).write_bytes(b"corrupt historical reviews")
            with patch.object(page, "ASSETS", base), self.real_inference_spy() as predict:
                app = self.launch()
                forecast = dict(self.run_prediction(app))
                self.reveal(app)
                self.assertTrue(any("historical review is unavailable" in warning.value for warning in app.warning))
                self.assertEqual(dict(app.session_state["clf_forecast"]), forecast)
                self.assertNotIn("Historical review", [metric.label for metric in app.metric])
                self.assertEqual(predict.call_count, 1)
                self.assertEqual(len(app.get("download_button")), 1)

    def test_all_49_read_only_inputs_have_meanings_and_exclude_current_review(self):
        columns = MANIFEST["metadata_columns"] + MANIFEST["display_feature_columns"]
        self.assertEqual(len(columns), 49)
        self.assertEqual(len(set(columns)), 49)
        self.assertEqual(set(columns), set(CLASSIFICATION_INPUT_MEANINGS))
        self.assertTrue(all(CLASSIFICATION_INPUT_MEANINGS[name].strip() for name in columns))
        forbidden = page.OUTCOME_COLUMNS | {"y", "risk_score", "prediction", "score"}
        self.assertFalse(set(columns) & forbidden)
        for version, spec in MANIFEST["versions"].items():
            with self.subTest(version=version):
                inputs = pd.read_parquet(ASSETS / spec["inputs"])
                self.assertEqual(set(inputs.columns), set(columns))
                self.assertEqual(inputs.shape, (120, 49))
                self.assertTrue(inputs.order_id.is_unique)
                self.assertEqual(inputs.groupby("t_month").size().to_dict(),
                                 {"2018-06": 40, "2018-07": 40, "2018-08": 40})
                ids_sha = hashlib.sha256("\n".join(inputs.order_id.astype(str)).encode()).hexdigest()
                self.assertEqual(ids_sha, spec["input_order_ids_sha256"])
        app = self.launch()
        expander = self.widget(app.expander, "See the classification inputs and their meanings")
        self.assertFalse(expander.proto.expanded)
        # Retain the full-width folded catalogue outside both prediction columns.
        column_rows = [[child for child in node.children.values() if child.type == "column"]
                       for node in app.main if hasattr(node, "children")]
        prediction_columns = [columns for columns in column_rows if len(columns) == 2
                              and any(widget.key == "clf_order" for widget in columns[0].selectbox)]
        self.assertEqual(len(prediction_columns), 1)
        self.assertFalse(prediction_columns[0][0].expander)
        self.assertFalse(prediction_columns[0][1].expander)
        details = next(text.value for text in app.markdown if 'class="dataframe olist-input-table"' in text.value)
        for column in columns:
            self.assertIn(f"<td>{column}</td>", details)
            self.assertIn(CLASSIFICATION_INPUT_MEANINGS[column].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), details)
        self.assertFalse(app.text_input)
        self.assertFalse(app.number_input)
        self.assertFalse(app.get("file_uploader"))


if __name__ == "__main__":
    unittest.main()
