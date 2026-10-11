"""Unit tests for Agent 2 — Trend Analysis Agent (Port 8002)."""

import os
import pytest

os.environ["MOCK_LLM"] = "1"

from app.agents.trends.agent import TrendAnalysisAgent
from app.agents.schemas import TrendAnalysisRequest, TrendDirection


@pytest.fixture
def trend_agent():
    return TrendAnalysisAgent()


def test_historical_trend_calculation(trend_agent):
    req = TrendAnalysisRequest(business_id="PP3BBaVxZLcJU54uP_wL6Q")
    res = trend_agent.analyze(req)
    assert res.status == "success"
    payload = res.payload
    assert payload["total_reviews_analyzed"] > 1000
    assert 1.0 <= payload["average_rating"] <= 5.0
    assert len(payload["aspect_trends"]) > 0

    directions = [t["direction"] for t in payload["aspect_trends"]]
    for d in directions:
        assert d in ("increasing", "decreasing", "stable")


def test_forecast_evaluation_on_held_out_split(trend_agent):
    req = TrendAnalysisRequest(business_id="PP3BBaVxZLcJU54uP_wL6Q", forecast_horizon=3)
    res = trend_agent.analyze(req)
    assert res.status == "success"
    forecast = res.payload["forecast"]
    assert forecast["status"] == "computed"
    assert forecast["horizon_months"] == 3
    assert forecast["mae"] is not None and forecast["mae"] >= 0.0
    assert forecast["rmse"] is not None and forecast["rmse"] >= 0.0
    assert len(forecast["forecast_values"]) == 3


def test_customer_satisfaction_prediction(trend_agent):
    req = TrendAnalysisRequest(business_id="PP3BBaVxZLcJU54uP_wL6Q")
    res = trend_agent.analyze(req)
    assert res.status == "success"
    sat = res.payload["satisfaction_prediction"]
    assert sat["status"] == "computed"
    assert 1.0 <= sat["predicted_score"] <= 5.0
    assert sat["metric_value"] is not None and sat["metric_value"] >= 0.0
    assert sat["train_samples"] > 0
    assert sat["test_samples"] > 0


def test_insufficient_data_handling(trend_agent):
    req = TrendAnalysisRequest(business_id="NON_EXISTENT_BUSINESS_XYZ_12345")
    res = trend_agent.analyze(req)
    assert res.status == "insufficient_data"
    assert len(res.warnings) > 0
    assert res.payload == {}
