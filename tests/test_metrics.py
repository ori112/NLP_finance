"""Tests for evaluation metrics and error analysis."""

import pandas as pd
import pytest

from src.evaluation.error_analysis import export_failure_cases, extract_failure_cases
from src.evaluation.metrics import compute_classification_metrics, compute_pearson_correlation


# ---------------------------------------------------------------------------
# compute_classification_metrics
# ---------------------------------------------------------------------------

def test_perfect_prediction_gives_1_0() -> None:
    y = ["positive", "negative", "neutral", "positive", "negative"]
    metrics = compute_classification_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0


def test_all_wrong_gives_0_accuracy() -> None:
    y_true = ["positive", "positive", "positive"]
    y_pred = ["negative", "negative", "negative"]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.0


def test_returns_all_expected_keys() -> None:
    metrics = compute_classification_metrics(["positive"], ["positive"])
    expected = {
        "accuracy", "macro_f1", "weighted_f1",
        "positive_f1", "neutral_f1", "negative_f1",
    }
    assert expected.issubset(set(metrics.keys()))


def test_mixed_prediction_accuracy() -> None:
    y_true = ["positive", "positive", "negative", "negative"]
    y_pred = ["positive", "negative", "negative", "negative"]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75


# ---------------------------------------------------------------------------
# compute_pearson_correlation
# ---------------------------------------------------------------------------

def test_perfect_positive_correlation() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    r, p = compute_pearson_correlation(xs, xs)
    assert abs(r - 1.0) < 1e-4


def test_perfect_negative_correlation() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [-1.0, -2.0, -3.0, -4.0, -5.0]
    r, p = compute_pearson_correlation(xs, ys)
    assert abs(r - (-1.0)) < 1e-4


def test_too_few_points_returns_zero() -> None:
    r, p = compute_pearson_correlation([1.0, 2.0], [1.0, 2.0])
    assert r == 0.0
    assert p == 1.0


# ---------------------------------------------------------------------------
# extract_failure_cases
# ---------------------------------------------------------------------------

def _make_df(n_mismatches: int = 12, n_heuristic: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_mismatches):
        rows.append({
            "ticker": f"T{i}", "date": "2023-01-01", "company": "Test Co",
            "model_label": "positive",
            "label_1d": "negative",
            "market_adj_1d": -0.03,
            "split_successful": True,
            "split_method": "regex_primary",
            "prepared_text": "positive text " * 20,
            "qa_text": "more positive text " * 20,
        })
    for i in range(n_heuristic):
        rows.append({
            "ticker": f"H{i}", "date": "2023-02-01", "company": "Test Co",
            "model_label": None,
            "label_1d": None,
            "market_adj_1d": None,
            "split_successful": False,
            "split_method": "heuristic",
            "prepared_text": "some text " * 20,
            "qa_text": "more text " * 20,
        })
    return pd.DataFrame(rows)


def test_extract_includes_mismatches() -> None:
    df = _make_df(n_mismatches=12, n_heuristic=0)
    result = extract_failure_cases(df, model_label_col="model_label")
    mismatch_rows = result[result["failure_reason"] != "split_heuristic"]
    assert len(mismatch_rows) == 12


def test_extract_includes_heuristic_splits() -> None:
    df = _make_df(n_mismatches=8, n_heuristic=5)
    result = extract_failure_cases(df, model_label_col="model_label")
    heuristic_rows = result[result["failure_reason"] == "split_heuristic"]
    assert len(heuristic_rows) == 5


def test_extract_raises_if_insufficient() -> None:
    df = _make_df(n_mismatches=3, n_heuristic=2)
    with pytest.raises(ValueError, match="at least 10"):
        extract_failure_cases(df, model_label_col="model_label", min_cases=10)


def test_extract_failure_reason_labels() -> None:
    df = _make_df(n_mismatches=12, n_heuristic=0)
    result = extract_failure_cases(df, model_label_col="model_label")
    valid_reasons = {"false_positive", "false_negative", "neutral_missed", "split_heuristic"}
    assert set(result["failure_reason"].unique()).issubset(valid_reasons)


def test_export_raises_if_too_few(tmp_path) -> None:
    small_df = pd.DataFrame({"a": range(5)})
    with pytest.raises(ValueError, match="at least 10"):
        export_failure_cases(small_df, output_path=tmp_path / "out.csv", min_cases=10)


def test_export_writes_csv(tmp_path) -> None:
    df = _make_df(n_mismatches=12, n_heuristic=0)
    failures = extract_failure_cases(df, model_label_col="model_label")
    out = tmp_path / "failures.csv"
    export_failure_cases(failures, output_path=out)
    assert out.exists()
    loaded = pd.read_csv(out)
    assert len(loaded) >= 10
