"""Fixed-example classification flows using the exported fitted forests.

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
ENTRY = PHASE.parent / "dashboard/app.py"
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

    def launch(self):
        app = AppTest.from_file(str(ENTRY)).run(timeout=60)
        self.assert_healthy(app)
        self.navigate(app, "Low-review prediction")
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
        if "clf_reveal" in app.session_state:
            self.assertFalse(app.session_state["clf_reveal"])
        self.assertNotIn("Historical review", [metric.label for metric in app.metric])
        self.assertNotIn("Low-review risk score", [metric.label for metric in app.metric])
        self.assertFalse(app.checkbox)
        self.assertFalse(app.get("download_button"))

    @contextmanager
    def real_inference_spy(self):
        # autospec preserves the bound self argument, then calls the real trees.
        with patch.object(PortableForest, "predict_proba", autospec=True,
                          side_effect=ACTUAL_PREDICT) as predict:
            yield predict

    def test_both_versions_run_real_forest_on_exact_row_without_outcomes_or_score_lookup(self):
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
            self.assertEqual(self.widget(app.selectbox, "Prediction model").value, "strict")
            self.assertEqual(predict.call_count, 0, "Opening the page cannot run inference")
            for count, version in enumerate(("strict", "v1"), 1):
                with self.subTest(version=version):
                    if version != "strict":
                        self.widget(app.selectbox, "Prediction model").set_value(version).run(timeout=60)
                        self.assert_cleared(app)
                        self.assertEqual(predict.call_count, count - 1)
                    selected = self.widget(app.selectbox, "Historical classification order").value
                    spec = MANIFEST["versions"][version]
                    frozen_selection = json.loads((PHASE / spec["selection_file"]).read_text())
                    self.assertEqual(spec["capacity_threshold"],
                                     frozen_selection["thresholds"]["capacity_threshold"])
                    self.assertEqual(spec["selection_sha256"], frozen_selection["selection_sha256"])
                    inputs = original_read(ASSETS / spec["inputs"])
                    ordinal = inputs.order_id.tolist().index(selected)
                    matrix = np.load(ASSETS / spec["matrix"], allow_pickle=False)
                    forest = load_forest(ASSETS / spec["forest"])
                    expected = float(ACTUAL_PREDICT(forest, matrix[[ordinal]])[0, 1])
                    result = self.run_prediction(app)
                    self.assertEqual(predict.call_count, count)
                    np.testing.assert_array_equal(predict.call_args.args[1], matrix[[ordinal]])
                    self.assertEqual(predict.call_args.args[1].shape, (1, spec["n_features"]))
                    self.assertAlmostEqual(result["risk_score"], expected, places=14)
                    self.assertEqual(result["validation_threshold"], spec["capacity_threshold"])
                    self.assertEqual(result["above_threshold"], expected >= spec["capacity_threshold"])
                    self.assertEqual(result["version"], version)
                    self.assertEqual(self.metric(app, "Low-review risk score"), f"{expected:.3f}")
                    self.assertEqual(self.metric(app, "Frozen validation threshold"), f"{spec['capacity_threshold']:.4f}")
                    self.assertFalse(self.widget(app.checkbox, "Reveal the historical review").value)
                    self.assertNotIn("Historical review", [metric.label for metric in app.metric])
                    self.assertEqual(len(app.get("download_button")), 1)
            outcomes.assert_not_called()
        self.assertTrue(any(Path(path).name == "strict_inputs.parquet" for path in read_paths))
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
            self.assertFalse(set(downloaded.columns) & (page.OUTCOME_COLUMNS | {"y"}))
            # A second explicit Run also closes a previously revealed review.
            self.run_prediction(app)
            self.assertFalse(self.widget(app.checkbox, "Reveal the historical review").value)
            self.assertNotIn("Historical review", [metric.label for metric in app.metric])
            self.assertEqual(load_outcomes.call_count, 1)
            self.assertEqual(predict.call_count, 2)

    def test_model_and_matching_filter_changes_clear_prediction_and_download(self):
        inputs = pd.read_parquet(ASSETS / MANIFEST["versions"]["strict"]["inputs"])
        with self.real_inference_spy() as predict:
            app = self.launch()
            chosen = self.widget(app.selectbox, "Historical classification order").value
            row = inputs.set_index("order_id").loc[chosen]
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.selectbox, "Trigger month").set_value(row.t_month).run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(self.widget(app.selectbox, "Historical classification order").value, chosen)
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.selectbox, "Customer destination").set_value(row.customer_state).run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(self.widget(app.selectbox, "Historical classification order").value, chosen)
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.selectbox, "Prediction model").set_value("v1").run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 3)

    def test_order_navigation_manual_selection_and_page_leave_never_restore_results(self):
        with self.real_inference_spy() as predict:
            app = self.launch()
            original = self.widget(app.selectbox, "Historical classification order").value
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.button, "Next →").click().run(timeout=60)
            self.assert_cleared(app)
            self.assertNotEqual(self.widget(app.selectbox, "Historical classification order").value, original)
            self.widget(app.button, "← Previous").click().run(timeout=60)
            self.assert_cleared(app)
            self.assertEqual(self.widget(app.selectbox, "Historical classification order").value, original)
            self.run_prediction(app)
            self.reveal(app)
            other = next(order for order in pd.read_parquet(ASSETS / "strict_inputs.parquet").order_id if order != original)
            self.widget(app.selectbox, "Historical classification order").set_value(other).run(timeout=60)
            self.assert_cleared(app)
            self.run_prediction(app)
            self.reveal(app)
            self.navigate(app, "Delivery prediction")
            self.assertNotIn("clf_forecast", app.session_state)
            self.navigate(app, "Low-review prediction")
            self.assert_cleared(app)
            self.assertEqual(predict.call_count, 3)

    def test_random_stays_in_filter_and_stale_state_resets_for_new_month(self):
        inputs = pd.read_parquet(ASSETS / "strict_inputs.parquet")
        with self.real_inference_spy() as predict:
            app = self.launch()
            self.widget(app.selectbox, "Trigger month").set_value("2018-06").run(timeout=60)
            self.widget(app.selectbox, "Customer destination").set_value("SP").run(timeout=60)
            before = self.widget(app.selectbox, "Historical classification order").value
            self.run_prediction(app)
            self.reveal(app)
            self.widget(app.button, "Random").click().run(timeout=60)
            self.assert_cleared(app)
            after = self.widget(app.selectbox, "Historical classification order").value
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
            selected = self.widget(app.selectbox, "Historical classification order").value
            self.assertEqual(inputs.set_index("order_id").loc[selected, "t_month"], "2018-07")
            self.assertEqual(predict.call_count, 2)

    def test_missing_manifest_and_corrupt_input_assets_fail_gracefully(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.object(page, "ASSETS", base):
                app = self.launch()
                self.assert_cleared(app)
                self.assertTrue(any("package is unavailable" in warning.value for warning in app.warning))
                shutil.copytree(ASSETS, base, dirs_exist_ok=True)
                (base / "strict_matrix.npy").write_bytes(b"corrupt matrix")
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
                    forest = base / MANIFEST["versions"]["strict"]["forest"]
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
        details = next(text.value for text in app.markdown if 'class="dataframe olist-input-table"' in text.value)
        for column in columns:
            self.assertIn(f"<td>{column}</td>", details)
            self.assertIn(CLASSIFICATION_INPUT_MEANINGS[column].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), details)
        self.assertFalse(app.text_input)
        self.assertFalse(app.number_input)
        self.assertFalse(app.get("file_uploader"))


if __name__ == "__main__":
    unittest.main()
