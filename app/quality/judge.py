"""Phase 4 LLM-as-Judge Evaluator for Operational Analysis.
Evaluates faithfulness, citation correctness, query relevance, and recommendation plausibility
using compact masked evidence representations and safe fallbacks.
"""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.agents.llm import get_llm_client, LLMClient
from app.quality.schemas import LLMJudgeVerdict, QualityStatus


class LLMJudgeRawResponse(BaseModel):
    verdict: str = Field(default="passed", description="'passed', 'flagged', or 'blocked'")
    faithfulness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    remediation_notes: List[str] = Field(default_factory=list)


JUDGE_SYSTEM_PROMPT = """You are an impartial Quality Gate Evaluator for AI-powered restaurant operational recommendations.
Evaluate the proposed analysis against the supplied evidence.
Return ONLY a valid JSON object matching this schema:
{
  "verdict": "<one of: passed, flagged, blocked>",
  "faithfulness_score": <float between 0.0 and 1.0>,
  "relevance_score": <float between 0.0 and 1.0>,
  "issues": ["<list of any unsupported claims, hallucinations, or contradictions found>"],
  "remediation_notes": ["<suggested corrective action for flagged issues>"]
}

Rules:
- blocked: Severe privacy violation, fabricated sources, or dangerous operational advice.
- flagged: Unsubstantiated hypotheses, missing citations, or minor inconsistencies.
- passed: Solidly grounded, logically sound, and directly addresses the problem.
- DO NOT invent evidence. Use only what is supplied.
"""


def evaluate_with_llm_judge(
    query: str,
    target_business_id: Optional[str],
    retrieved_evidence: List[Dict[str, Any]],
    classification_payload: Optional[Dict[str, Any]],
    root_cause_payload: Optional[Dict[str, Any]],
    action_planner_payload: Optional[Dict[str, Any]],
    deterministic_fallback_verdict: QualityStatus = QualityStatus.PASSED,
    llm_client: Optional[LLMClient] = None,
) -> LLMJudgeVerdict:
    """Executes a compact, masked LLM-as-judge evaluation with safe timeout fallback."""
    start_time = time.time()
    client = llm_client or get_llm_client()

    # Build compact, masked representation of evidence and findings
    compact_evidence = []
    for c in (retrieved_evidence or [])[:5]:
        cid = c.get("chunk_id", "unknown")
        txt = (c.get("text_masked") or c.get("text") or "")[:200]
        compact_evidence.append(f"[{cid}] {txt}")

    hypotheses = []
    if root_cause_payload:
        for h in root_cause_payload.get("hypotheses", []):
            cause = h.get("hypothesized_cause", "")
            dtype = h.get("distinction_type", "")
            sids = h.get("supporting_source_ids", [])
            hypotheses.append(f"- Cause: '{cause}' (Type: {dtype}, Sources: {sids})")

    actions = []
    if action_planner_payload:
        for a in action_planner_payload.get("ranked_actions", []):
            rec = a.get("recommended_action", "")
            owner = a.get("owner_role", "")
            actions.append(f"- Action: '{rec}' (Owner: {owner})")

    prompt = f"""OPERATIONAL QUERY: {query}
TARGET BUSINESS ID: {target_business_id or 'General'}

COMPACT EVIDENCE CONTEXT:
{chr(10).join(compact_evidence) if compact_evidence else 'No evidence chunks provided.'}

PROPOSED HYPOTHESES:
{chr(10).join(hypotheses) if hypotheses else 'None.'}

PROPOSED ACTIONS:
{chr(10).join(actions) if actions else 'None.'}
"""

    mock_resp = LLMJudgeRawResponse(
        verdict=deterministic_fallback_verdict.value,
        faithfulness_score=0.95 if deterministic_fallback_verdict == QualityStatus.PASSED else 0.65,
        relevance_score=0.92,
        issues=[],
        remediation_notes=[]
    ).model_dump()

    try:
        raw_result, model_used = client.generate_json(
            prompt=prompt,
            system_instruction=JUDGE_SYSTEM_PROMPT,
            model_name="gemini-3.8-flash",
            schema_cls=LLMJudgeRawResponse,
            mock_response=mock_resp,
        )
        latency = (time.time() - start_time) * 1000

        verdict_str = str(raw_result.get("verdict", "passed")).lower()
        if verdict_str == "blocked":
            verdict_enum = QualityStatus.BLOCKED
        elif verdict_str == "flagged":
            verdict_enum = QualityStatus.FLAGGED
        else:
            verdict_enum = QualityStatus.PASSED

        return LLMJudgeVerdict(
            verdict=verdict_enum,
            faithfulness_score=float(raw_result.get("faithfulness_score", 1.0)),
            relevance_score=float(raw_result.get("relevance_score", 1.0)),
            issues=list(raw_result.get("issues", [])),
            remediation_notes=list(raw_result.get("remediation_notes", [])),
            latency_ms=latency,
            model_used=model_used,
        )
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.warning(f"LLM Judge evaluation failed or timed out ({e}); applying deterministic fallback.")
        return LLMJudgeVerdict(
            verdict=deterministic_fallback_verdict,
            faithfulness_score=0.85,
            relevance_score=0.85,
            issues=[f"LLM-as-judge unavailable ({str(e)[:100]}); relying on deterministic validation."],
            remediation_notes=["Review findings manually using deterministic source citations."],
            latency_ms=latency,
            model_used="safe_deterministic_fallback",
        )
