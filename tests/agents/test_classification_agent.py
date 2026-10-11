"""Unit tests for Agent 1 — Classification Agent (Port 8001)."""

import os
import pytest

os.environ["MOCK_LLM"] = "1"

from app.agents.classification.agent import ClassificationAgent
from app.agents.schemas import (
    ClassificationRequest,
    ComplaintCategory,
    AspectTag,
    SentimentType,
    SeverityLevel,
)


@pytest.fixture
def classifier():
    return ClassificationAgent()


def test_positive_review_classification(classifier):
    req = ClassificationRequest(
        review_id="rev_pos",
        text="The food was amazing, extremely delicious and fresh! We loved the cozy vibe."
    )
    res = classifier.analyze(req)
    assert res.status == "success"
    assert res.payload["sentiment"] == SentimentType.POSITIVE.value
    assert res.payload["severity"] == SeverityLevel.LOW.value
    assert AspectTag.TASTE_FRESHNESS.value in res.payload["aspects"]
    assert len(res.evidence) > 0
    assert res.evidence[0].review_id == "rev_pos"


def test_service_speed_negative_complaint(classifier):
    req = ClassificationRequest(
        review_id="rev_wait",
        text="We waited 45 minutes just to get our water, and the service was so slow!"
    )
    res = classifier.analyze(req)
    assert res.status == "success"
    assert res.payload["sentiment"] == SentimentType.NEGATIVE.value
    assert ComplaintCategory.SERVICE_SPEED.value in res.payload["categories"]
    assert AspectTag.WAITING_TIME.value in res.payload["aspects"]
    assert res.payload["severity"] in (SeverityLevel.MEDIUM.value, SeverityLevel.HIGH.value)
    assert len(res.evidence) > 0


def test_critical_health_hazard_severity(classifier):
    req = ClassificationRequest(
        review_id="rev_crit",
        text="Found a live roach in my salad! Later that evening I got severe food poisoning and ended up in the hospital."
    )
    res = classifier.analyze(req)
    assert res.status == "success"
    assert res.payload["severity"] == SeverityLevel.CRITICAL.value
    assert ComplaintCategory.CLEANLINESS.value in res.payload["categories"]
    assert AspectTag.HYGIENE.value in res.payload["aspects"]
    assert "critical" in res.payload["severity_reason"].lower()


def test_mixed_multi_aspect_review(classifier):
    req = ClassificationRequest(
        review_id="rev_mixed",
        text="The steak tasted delicious and fresh, but the server was very rude and our drinks were delayed 30 minutes."
    )
    res = classifier.analyze(req)
    assert res.status == "success"
    assert res.payload["sentiment"] == SentimentType.MIXED.value
    assert res.payload["is_multi_aspect"] is True
    assert AspectTag.TASTE_FRESHNESS.value in res.payload["aspects"]
    assert AspectTag.STAFF_RESPONSIVENESS.value in res.payload["aspects"]
    assert AspectTag.WAITING_TIME.value in res.payload["aspects"]


def test_empty_input_handling(classifier):
    req = ClassificationRequest(review_id="rev_empty", text="")
    res = classifier.analyze(req)
    assert res.status == "insufficient_data"
    assert len(res.warnings) > 0
    assert res.payload["severity"] == SeverityLevel.LOW.value


def test_unsupported_or_ambiguous_category_fallback(classifier):
    req = ClassificationRequest(
        review_id="rev_other",
        text="The parking meter outside was broken."
    )
    res = classifier.analyze(req)
    assert res.status == "success"
    assert ComplaintCategory.OTHER.value in res.payload["categories"]
    assert AspectTag.OTHER.value in res.payload["aspects"]
