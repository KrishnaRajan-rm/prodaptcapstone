"""Phase 4 Statistical and Historical Freshness Validation.
Discloses historical dataset cutoff (Jan 2022 vs Oct 2026), validates trend calculations,
and ensures forecasts are explicitly labeled as predictions rather than current observed facts.
"""

from typing import Any, Dict, List, Optional
from app.quality.schemas import QualityStatus, StatisticalCheckResult

DATASET_CUTOFF_YEAR_MONTH = "2022-01"
CURRENT_REFERENCE_YEAR_MONTH = "2026-10"


def validate_statistics_and_freshness(
    query: str,
    trend_payload: Optional[Dict[str, Any]],
) -> StatisticalCheckResult:
    """Validates trend analysis metrics, forecast labeling, and historical freshness."""
    result = StatisticalCheckResult(
        dataset_cutoff_date=DATASET_CUTOFF_YEAR_MONTH,
        current_reference_date=CURRENT_REFERENCE_YEAR_MONTH,
        stale_historical_data_disclosed=True,
    )

    # Standard staleness disclosure notice for historical Yelp dataset
    staleness_notice = (
        f"Historical Data Notice: Dataset cutoff is {DATASET_CUTOFF_YEAR_MONTH}. "
        f"Any forecasts or trends reflect historical projection horizons from early 2022, "
        f"not current operational conditions in {CURRENT_REFERENCE_YEAR_MONTH}."
    )
    result.staleness_disclosure = staleness_notice

    if not trend_payload or not isinstance(trend_payload, dict):
        result.warnings.append("No trend analysis payload provided for statistical verification.")
        return result

    # 1. Check review count and rating boundaries
    total_reviews = trend_payload.get("total_reviews_analyzed", 0)
    avg_rating = trend_payload.get("average_rating")

    if total_reviews is not None and total_reviews < 0:
        result.status = QualityStatus.FLAGGED
        result.warnings.append(f"Invalid total_reviews_analyzed: {total_reviews}")

    if avg_rating is not None and not (1.0 <= avg_rating <= 5.0):
        result.status = QualityStatus.FLAGGED
        result.warnings.append(f"Invalid average_rating out of [1.0, 5.0] range: {avg_rating}")

    # 2. Check forecast labeling
    forecast = trend_payload.get("forecast")
    if forecast and isinstance(forecast, dict):
        forecast_status = forecast.get("status")
        forecast_values = forecast.get("forecast_values", {})

        # Ensure forecast is labeled as computed / prediction, not observed fact
        if forecast_status not in ("computed", "prediction", "forecast", "insufficient_data"):
            result.forecast_properly_labeled = False
            result.status = QualityStatus.FLAGGED
            result.warnings.append(
                f"Forecast status '{forecast_status}' is ambiguous; must be explicitly labeled as prediction."
            )

        # Check if historical forecast dates (e.g. 2022-02) are being presented without disclosure
        for period_key in forecast_values.keys():
            if period_key.startswith("2022-") or period_key.startswith("2021-"):
                result.verified_trends.append({
                    "period": period_key,
                    "value": forecast_values[period_key],
                    "type": "historical_projection",
                    "cutoff_notice": staleness_notice
                })

    # 3. Check query freshness expectation
    query_lower = (query or "").lower()
    current_intent_cues = ["today", "right now", "currently", "latest", "recent", "2026", "this month"]
    if any(cue in query_lower for cue in current_intent_cues):
        result.warnings.append(
            f"User query refers to current/recent operations ('{query}'), but underlying corpus ends in {DATASET_CUTOFF_YEAR_MONTH}. "
            "Staleness disclosure is required."
        )

    return result
