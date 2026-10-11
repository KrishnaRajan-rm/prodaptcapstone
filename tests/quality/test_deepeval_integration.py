"""Unit test for DeepEval recommendation evaluation integration."""

import pytest
from evaluation.deepeval_eval import RestaurantOpsDeepEvalLLM, run_deepeval_benchmark


def test_deepeval_benchmark_execution():
    """Verifies DeepEval benchmark executes Faithfulness and Relevancy metrics without errors."""
    results = run_deepeval_benchmark()
    assert results["status"] == "success"
    assert results["faithfulness_score"] >= 0.7
    assert results["relevancy_score"] >= 0.7
    assert results["passed"] is True
