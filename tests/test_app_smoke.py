"""Headless UI smoke tests via streamlit.testing.AppTest.

These execute the real page scripts (widgets, badges, data_editor, tabs) and
fail on any uncaught exception — the cheapest guard against UI regressions.
"""

from streamlit.testing.v1 import AppTest

from resumesort.sample_data import load_sample_reports


def test_landing_renders_without_exception():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception


def test_sample_data_flow_renders_results():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    buttons = [b for b in at.button if "sample" in str(b.label).lower()]
    assert buttons, "landing page must offer the sample-data CTA"
    buttons[0].click().run()
    assert not at.exception
    assert len(at.session_state["reports"]) == 3


def test_compare_page_renders_with_reports():
    at = AppTest.from_file("pages/1_Compare.py", default_timeout=30)
    at.session_state["reports"] = load_sample_reports()
    at.run()
    assert not at.exception


def test_methodology_page_renders():
    at = AppTest.from_file("pages/2_Methodology.py", default_timeout=30)
    at.run()
    assert not at.exception
