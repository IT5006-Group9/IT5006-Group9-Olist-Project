"""Shared exploration cohort and cross-page widget lifecycle checks.

Only committed marketplace data is read by the UI checks. No model is fitted or
scored, and no evaluation outputs are changed.
"""
from pathlib import Path
import sys
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest


PHASE = Path(__file__).resolve().parents[1]
REPOSITORY = PHASE.parent
sys.path.insert(0, str(PHASE))

from deployment import explorer_filters as filters  # noqa: E402


def fixture():
    return pd.DataFrame([
        ("early", "2016-10", "SP", "books", "credit_card", True, "delivered", "2016-10-10", 5),
        ("sp_same", "2018-06", "SP", "books", "credit_card", True, "delivered", "2018-06-10", 5),
        ("sp_cross", "2018-06", "SP", "books", "credit_card", False, "delivered", "2018-06-12", 2),
        ("rj_cross", "2018-06", "RJ", "books", "boleto", False, "shipped", None, 1),
        ("rj_same", "2018-07", "RJ", "toys", "credit_card", True, "delivered", None, None),
        ("unknown_route", "2018-06", "SP", "books", "credit_card", None, "delivered", "2018-06-15", 3),
        ("late", "2018-09", "SP", "toys", "boleto", False, "delivered", "2018-09-20", 4),
    ], columns=["order_id", "month", "customer_state", "category", "payment_type", "same_state",
                "order_status", "delivered_ts", "review_score"], index=[2, 4, 6, 8, 10, 12, 14])


class SharedFilterTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.state = {}
        filters.init_filters(self.state, self.frame)

    def test_default_window_and_complete_category_options(self):
        self.assertEqual(filters.read_filters(self.state)["month_range"], ("2016-10", "2018-09"))
        # A full extract has both default endpoints; sparse edge months remain selectable.
        full = pd.concat([self.frame, self.frame.iloc[[0]].assign(month="2017-01"),
                          self.frame.iloc[[0]].assign(month="2018-08")])
        self.assertEqual(filters.init_filters({}, full)["month_range"], ("2017-01", "2018-08"))
        many = pd.concat([self.frame.iloc[[0]].assign(category=f"category_{i:02}") for i in range(35)])
        self.assertEqual(len(filters.filter_options(many)["categories"]), 35)

    def test_intersection_supports_multiple_states_and_strict_route_membership(self):
        chosen = filters.write_filters(
            self.state, month_range=("2018-06", "2018-07"), states=["SP", "RJ"],
            categories=["books"], payments=["credit_card"], route="Cross state", delivered_only=True,
        )
        self.assertEqual(filters.apply_filters(self.frame, chosen).order_id.tolist(), ["sp_cross"])
        # Missing route information does not mean a known cross-state route.
        self.assertNotIn("unknown_route", filters.apply_filters(self.frame, chosen).order_id.tolist())
        chosen = filters.write_filters(self.state, payments=[], delivered_only=False)
        self.assertEqual(filters.apply_filters(self.frame, chosen).order_id.tolist(), ["sp_cross", "rj_cross"])

    def test_empty_intersection_is_preserved_and_can_be_reset(self):
        filters.write_filters(self.state, states=["SP"], categories=["toys"], route="Same state")
        self.assertTrue(filters.apply_filters(self.frame, filters.read_filters(self.state)).empty)
        # Re-entering a page must not silently broaden a valid but empty choice.
        filters.init_filters(self.state, self.frame)
        self.assertEqual(filters.read_filters(self.state)["categories"], ("toys",))
        cleared = filters.reset_filters(self.state)
        self.assertGreater(len(filters.apply_filters(self.frame, cleared)), 0)
        self.assertEqual(cleared["states"], ())
        self.assertEqual(cleared["route"], "All")

    def test_extra_reset_retains_visible_months_states_then_full_reset_clears_everything(self):
        filters.write_filters(
            self.state, month_range=("2018-06", "2018-07"), states=["SP", "RJ"], categories=["books"],
            payments=["boleto"], route="Cross state", delivered_only=True,
        )
        kept = filters.reset_filters(self.state, extra_only=True)
        self.assertEqual(kept["month_range"], ("2018-06", "2018-07"))
        self.assertEqual(kept["states"], ("RJ", "SP"))
        self.assertEqual((kept["categories"], kept["payments"], kept["route"], kept["delivered_only"]),
                         ((), (), "All", False))
        defaults = filters.init_filters({}, self.frame)
        self.assertEqual(filters.reset_filters(self.state), defaults)

    def test_widget_cleanup_does_not_own_durable_state_and_read_is_detached(self):
        self.state["overview_filter_states"] = ["SP", "RJ"]
        filters.write_filters(self.state, states=self.state["overview_filter_states"])
        del self.state["overview_filter_states"]
        restored = filters.init_filters(self.state, self.frame)
        self.assertEqual(restored["states"], ("RJ", "SP"))
        restored["states"] = ()
        self.assertEqual(filters.read_filters(self.state)["states"], ("RJ", "SP"))

    def test_summary_exposes_every_hidden_condition(self):
        chosen = filters.write_filters(self.state, states=["SP"], categories=["books"],
                                       payments=["boleto"], route="Cross state", delivered_only=True)
        text = filters.summary(chosen)
        for visible in ("SP", "books", "boleto", "Cross state", "Delivered orders only"):
            self.assertIn(visible, text)

    def test_invalid_write_is_atomic_and_does_not_erase_previous_choices(self):
        previous = filters.write_filters(self.state, states=["SP"])
        for invalid in ({"month_range": ("2018-09", "2016-10")}, {"states": ["XX"]},
                        {"payments": "boleto"}, {"route": "unknown"}, {"delivered_only": "yes"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                filters.write_filters(self.state, **invalid)
            self.assertEqual(filters.read_filters(self.state), previous)

    def test_status_filter_preserves_original_delivery_and_rating_denominators(self):
        original = self.frame.copy(deep=True)
        chosen = filters.write_filters(self.state, month_range=("2018-06", "2018-07"))
        selected = filters.apply_filters(self.frame, chosen)
        self.assertIn("rj_cross", selected.order_id.tolist(), "Observed reviews need not have a delivery date")
        self.assertEqual(selected.delivered_ts.notna().sum(), 3)
        self.assertEqual(selected.review_score.notna().sum(), 4)
        chosen = filters.write_filters(self.state, delivered_only=True)
        delivered_status = filters.apply_filters(self.frame, chosen)
        self.assertIn("rj_same", delivered_status.order_id.tolist(), "Volume uses order status, not date availability")
        self.assertNotIn("rj_cross", delivered_status.order_id.tolist())
        pd.testing.assert_frame_equal(self.frame, original)


class SharedFilterAppTests(unittest.TestCase):
    def assert_healthy(self, app):
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual([error.value for error in app.error], [])

    def widget(self, widgets, label):
        found = [widget for widget in widgets if widget.label == label]
        self.assertEqual(len(found), 1, f"Expected one {label!r} widget")
        return found[0]

    def navigate(self, app, page):
        app.sidebar.radio(key="workspace").set_value(page).run(timeout=60)
        self.assert_healthy(app)

    def orders(self, app):
        found = [metric for metric in app.metric if metric.label == "Orders"]
        self.assertTrue(found)
        # Explorer geography also has route-specific "Orders" tiles later.
        return int(found[0].value.replace(",", ""))

    def test_real_cross_page_persistence_hidden_extras_and_both_reset_modes(self):
        app = AppTest.from_file(str(REPOSITORY / "dashboard/app.py")).run(timeout=60)
        self.assert_healthy(app)
        default_count = self.orders(app)
        self.widget(app.multiselect, "Customer state").set_value(["SP", "RJ"]).run(timeout=60)
        self.widget(app.select_slider, "Purchase window").set_value(("2018-06", "2018-08")).run(timeout=60)
        visible_count = self.orders(app)
        self.navigate(app, "Marketplace explorer")
        self.assertEqual(set(self.widget(app.sidebar.multiselect, "Customer state").value), {"SP", "RJ"})
        self.assertEqual(self.widget(app.sidebar.select_slider, "Purchase month").value, ("2018-06", "2018-08"))
        self.widget(app.sidebar.multiselect, "Product category").set_value(["bed_bath_table"]).run(timeout=60)
        self.widget(app.sidebar.multiselect, "Payment type").set_value(["credit_card"]).run(timeout=60)
        self.widget(app.sidebar.radio, "Route").set_value("Cross state").run(timeout=60)
        self.widget(app.sidebar.checkbox, "Delivered orders only").check().run(timeout=60)
        self.assert_healthy(app)
        narrowed_count = self.orders(app)
        self.assertGreater(narrowed_count, 0)
        self.assertLess(narrowed_count, visible_count)
        # A third page removes both pages' temporary widget keys.
        self.navigate(app, "Guide & methodology")
        self.navigate(app, "Overview")
        self.assertEqual(self.orders(app), narrowed_count)
        self.assertEqual(set(self.widget(app.multiselect, "Customer state").value), {"SP", "RJ"})
        self.assertTrue(any("bed_bath_table" in caption.value and "credit_card" in caption.value
                            for caption in app.caption))
        self.widget(app.button, "Clear extra filters").click().run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.orders(app), visible_count)
        self.navigate(app, "Marketplace explorer")
        self.assertEqual(self.widget(app.sidebar.multiselect, "Product category").value, [])
        self.assertEqual(self.widget(app.sidebar.multiselect, "Payment type").value, [])
        self.assertEqual(self.widget(app.sidebar.radio, "Route").value, "All")
        self.assertFalse(self.widget(app.sidebar.checkbox, "Delivered orders only").value)
        self.widget(app.sidebar.button, "Reset all filters").click().run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.orders(app), default_count)
        self.assertEqual(self.widget(app.sidebar.multiselect, "Customer state").value, [])
        self.assertEqual(self.widget(app.sidebar.select_slider, "Purchase month").value, ("2017-01", "2018-08"))

    def test_standalone_explorer_empty_selection_keeps_working_reset(self):
        app = AppTest.from_file(str(REPOSITORY / "Phase 1/dashboard/app.py")).run(timeout=60)
        self.assert_healthy(app)
        source = pd.read_parquet(REPOSITORY / "Phase 1/dashboard/orders.parquet")
        default = source[source.month.astype(str).between("2017-01", "2018-08")]
        self.widget(app.sidebar.select_slider, "Purchase month").set_value(("2018-06", "2018-08")).run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.widget(app.sidebar.select_slider, "Purchase month").value, ("2018-06", "2018-08"))
        recent = default[default.month.astype(str).between("2018-06", "2018-08")]
        # Choose a genuine incompatible pair from the data, rather than relying
        # on an unknown option or a hardcoded historical count.
        pair = next((state, category) for state in sorted(recent.customer_state.dropna().unique())
                    for category in sorted(default.category.dropna().unique())
                    if not ((recent.customer_state == state) & (recent.category == category)).any())
        self.widget(app.sidebar.multiselect, "Customer state").set_value([pair[0]]).run(timeout=60)
        self.widget(app.sidebar.multiselect, "Product category").set_value([pair[1]]).run(timeout=60)
        self.assert_healthy(app)
        self.assertTrue(any("No orders match" in warning.value for warning in app.warning))
        self.assertEqual(self.widget(app.sidebar.multiselect, "Product category").value, [pair[1]])
        self.widget(app.sidebar.button, "Reset all filters").click().run(timeout=60)
        self.assert_healthy(app)
        self.assertEqual(self.orders(app), len(default))
        self.assertEqual(self.widget(app.sidebar.select_slider, "Purchase month").value, ("2017-01", "2018-08"))


if __name__ == "__main__":
    unittest.main()
