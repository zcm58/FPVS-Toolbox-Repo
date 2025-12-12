import logging

import numpy as np
import pandas as pd

from Tools.Stats.Legacy.cross_phase_lmm_core import _build_contrast, _build_fixed_effects_table
from Tools.Stats.PySide6.stats_cross_phase import _run_roi_condition_lmm


def _dummy_result_for_contrast(exog_columns):
    class DummyResult:
        def __init__(self, columns):
            self.fe_params = pd.Series(np.zeros(len(columns)), index=columns)
            self._cov = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)

        def cov_params(self):
            return self._cov

    return DummyResult(exog_columns)


def test_fixed_effects_all_terms_present():
    df = pd.DataFrame(
        [
            {"bca": 1.0, "subject": "s1", "group": "A", "phase": "X", "condition": "c1", "roi": "r1"},
            {"bca": 1.5, "subject": "s1", "group": "A", "phase": "Y", "condition": "c1", "roi": "r1"},
            {"bca": 0.9, "subject": "s2", "group": "B", "phase": "X", "condition": "c1", "roi": "r1"},
            {"bca": 1.4, "subject": "s2", "group": "B", "phase": "Y", "condition": "c1", "roi": "r1"},
        ]
    )

    results = _run_roi_condition_lmm(
        df,
        roi="r1",
        condition="c1",
        phase_labels=["X", "Y"],
        message_cb=lambda msg: None,
    )

    missing = [row for row in results["fixed_effects"] if row.get("term_missing")]
    assert not missing
    assert results["meta"].get("contrast_meta", {}).get("missing_reason") is None


def test_fixed_effects_single_group_flags_missing_terms():
    df = pd.DataFrame(
        [
            {"bca": 1.0, "subject": "s1", "group": "A", "phase": "X", "condition": "c1", "roi": "r1"},
            {"bca": 1.2, "subject": "s1", "group": "A", "phase": "Y", "condition": "c1", "roi": "r1"},
            {"bca": 0.8, "subject": "s2", "group": "A", "phase": "X", "condition": "c1", "roi": "r1"},
            {"bca": 1.1, "subject": "s2", "group": "A", "phase": "Y", "condition": "c1", "roi": "r1"},
        ]
    )

    results = _run_roi_condition_lmm(
        df,
        roi="r1",
        condition="c1",
        phase_labels=["X", "Y"],
        message_cb=lambda msg: None,
    )

    missing = {row["effect"]: row for row in results["fixed_effects"] if row.get("term_missing")}
    assert "group[T.A]" in missing
    assert missing["group[T.A]"]["missing_reason"] == "single_group_level"
    assert np.isnan(missing["group[T.A]"]["p"])
    assert results["meta"].get("contrast_meta", {}).get("missing_reason") == "single_group_level"


def test_build_fixed_effects_missing_term_set_to_nan():
    class DummyResult:
        def __init__(self):
            self.fe_params = pd.Series([1.0], index=["Intercept"])
            self.bse_fe = pd.Series([0.1], index=["Intercept"])
            self.model = type("M", (), {"exog_names": ["Intercept"]})

    dummy = DummyResult()
    table = _build_fixed_effects_table(
        dummy,
        expected_terms=["Intercept", "group[T.B]"],
        missing_reasons=None,
    )

    missing_rows = [row for row in table if row["effect"] == "group[T.B]"]
    assert missing_rows and missing_rows[0]["term_missing"] is True
    assert missing_rows[0]["missing_reason"] == "term_not_in_fe_params"
    assert np.isnan(missing_rows[0]["estimate"])


def test_build_contrast_reports_missing_columns():
    exog_columns = ["Intercept", "group[T.B]", "phase[T.Y]", "group[T.B]:phase[T.Y]", "missing_col"]
    dummy_result = _dummy_result_for_contrast(exog_columns)
    contrasts, meta = _build_contrast(
        "group * phase",
        exog_columns,
        ["A", "B"],
        ("X", "Y"),
        "c1",
        "r1",
        dummy_result,
        logger=logging.getLogger(__name__),
    )

    assert meta.get("missing_reason") == "missing_exog_columns"
    assert all(row["term_missing"] for row in contrasts)
    assert contrasts[0].get("missing_cols")
