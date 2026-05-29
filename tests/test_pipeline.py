import logging
from unittest.mock import patch

import pytest

from pipeline import load_data, predict, preprocess, run_pipeline


def test_pipeline():
    data = load_data()
    processed = preprocess(data)
    result = predict(processed)

    assert result == 12


def test_integration_pipeline():
    output = run_pipeline()

    assert output["result"] == 12
    assert output["records"] == 3
    assert output["elapsed_seconds"] < 1.0


def test_logging_validation(caplog):
    caplog.set_level(logging.INFO)

    run_pipeline()

    messages = [record.message for record in caplog.records]
    assert any("Loading data" in msg for msg in messages)
    assert any("Preprocessing" in msg for msg in messages)
    assert any("Running prediction" in msg for msg in messages)
    assert any("Pipeline finished" in msg for msg in messages)


def test_timing_check():
    output = run_pipeline()
    assert output["elapsed_seconds"] >= 0
    assert output["elapsed_seconds"] < 0.5


@patch("pipeline.load_data", return_value=[5, 5, 5])
def test_mock_database_input(mock_load):
    output = run_pipeline(source="database")

    mock_load.assert_called_once_with(source="database")
    assert output["result"] == 30
