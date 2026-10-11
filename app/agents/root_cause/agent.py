"""Root-Cause Analysis Agent (Service Port 8003).
Investigates why recurring operational failures happen using HybridRetriever evidence
and deterministic operational records. Distinguishes observed facts, statistical associations,
and inferred explanations, and checks for contradictory evidence.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import time
from typing import Any, Dict, List, Optional
import duckdb
from google.adk.agents import LlmAgent

from app.core.config import settings
from app.core.logging import logger
from app.retrieval.hybrid import HybridRetriever, get_retriever, SearchResult
from app.agents.schemas import (
    RootCauseRequest,
    RootCausePayload,
    RootCauseHypothesis,
    DistinctionType,
    AgentResponseEnvelope,
    EvidenceRef,
)
from app.agents.llm import get_llm_client


ROOT_CAUSE_INSTRUCTION = """You are a Root-Cause Analysis Specialist for restaurant operations.
Investigate the problem statement and evidence chunks provided.
Return ONLY a valid JSON object matching this schema:
{
  "problem_statement": "<problem_statement>",
  "business_id": "<business_id_or_null>",
  "retrieved_chunk_count": <number_of_chunks>,
  "factual_observations": ["<strictly observed facts from the evidence>"],
  "hypotheses": [
    {
      "hypothesis_id": "hyp_1",
      "hypothesized_cause": "<plausible operational root cause>",
      "distinction_type": "<one of: observed_fact, statistical_association, inferred_explanation>",
      "supporting_evidence": ["<verbatim or exact paraphrased quotes>"],
      "supporting_source_ids": ["<matching chunk_ids from the evidence>"],
      "contradictory_evidence": ["<quotes indicating opposite or unaffected service, or empty if none>"],
      "contradictory_source_ids": ["<matching chunk_ids, or empty>"],
      "alternative_explanations": ["<other plausible explanations for the observed symptom>"],
      "confidence": "<one of: low, medium, high>",
      "additional_info_needed": "<specific operational data needed to definitively verify without doubt (e.g. POS ticket logs, KDS timestamps, shift rosters)>"
    }
  ]
}

Epistemic rules:
- observed_fact: Directly reported in reviews (e.g. 'table waited 45 minutes for drinks').
- statistical_association: Pattern correlated with time/aspect (e.g. 'complaints concentrated on Friday evenings').
- inferred_explanation: Theoretical root cause (e.g. 'kitchen understaffing or prep delay'). You must label this as inferred, NOT an absolute proven fact.
- NEVER invent chunk IDs or quotes not in the provided evidence.
"""


class RootCauseAgent:
    """Specialist Agent for evidence-backed root cause analysis."""

    def __init__(self, retriever: Optional[HybridRetriever] = None, model_name: str = "gemini-3.8-flash"):
        self.model_name = model_name
        self.retriever = retriever or get_retriever()
        self.llm_client = get_llm_client()

        self.adk_agent = LlmAgent(
            name="root_cause_agent",
            model=model_name,
            instruction=ROOT_CAUSE_INSTRUCTION,
            description="Investigates restaurant operational failures using Hybrid RAG evidence and grounded root-cause hypotheses."
        )

    def analyze(self, request: RootCauseRequest) -> AgentResponseEnvelope:
        """Executes evidence retrieval via HybridRetriever and generates grounded hypotheses."""
        start_time = time.time()
        problem = (request.problem_statement or "").strip()

        if not problem:
            return AgentResponseEnvelope(
                agent_id="root_cause_agent",
                status="insufficient_data",
                warnings=["Problem statement is empty."],
                limitations=["Cannot investigate root causes without a specified problem statement."],
                payload={},
                execution_metadata={"latency_ms": (time.time() - start_time) * 1000}
            )

        # 1. Retrieve evidence chunks using existing HybridRetriever
        filters = {}
        if request.business_id:
            filters["business_id"] = request.business_id

        retrieved_chunks = self.retriever.search(
            query=problem,
            top_k=request.top_k_evidence,
            filters=filters if filters else None,
        )

        # 2. Check operational facts from DuckDB if business_id provided
        operational_facts = self._lookup_operational_context(request.business_id)

        # 3. Construct deterministic fallback hypotheses
        fallback_payload = self._build_grounded_hypotheses(
            problem=problem,
            business_id=request.business_id,
            chunks=retrieved_chunks,
            operational_facts=operational_facts,
            upstream_classification=request.classification_findings,
            upstream_trends=request.trend_findings
        )

        # 4. LLM synthesis with grounded prompt
        try:
            evidence_text = "\n".join([
                f"[Chunk {c.chunk_id}] (Stars: {c.stars}★ | Aspects: {c.aspects})\nReview excerpt: \"{c.text_masked}\""
                for c in retrieved_chunks
            ])

            prompt = f"""Problem Statement: {problem}
Target Business ID: {request.business_id or 'General / Multi-unit'}
Operational Context: {operational_facts}

RETRIEVED REVIEW EVIDENCE ({len(retrieved_chunks)} Chunks):
{evidence_text if evidence_text else 'No review chunks found matching filters.'}
"""
            llm_result, model_used = self.llm_client.generate_json(
                prompt=prompt,
                system_instruction=ROOT_CAUSE_INSTRUCTION,
                model_name=self.model_name,
                schema_cls=RootCausePayload,
                mock_response=fallback_payload.model_dump()
            )
            payload = RootCausePayload.model_validate(llm_result)
        except Exception as e:
            logger.warning(f"Root Cause LLM synthesis failed, using rule-based synthesis: {e}")
            payload = fallback_payload
            model_used = "deterministic_evidence_synthesizer"

        # Construct traceable evidence references
        evidence_refs: List[EvidenceRef] = []
        for c in retrieved_chunks:
            evidence_refs.append(
                EvidenceRef(
                    source_id=c.chunk_id,
                    chunk_id=c.chunk_id,
                    review_id=c.review_id,
                    business_id=c.business_id,
                    snippet=c.text_masked[:200],
                    relevance_score=c.rerank_score or c.rrf_score,
                    is_supporting=True,
                    metadata={"stars": c.stars, "aspects": c.aspects}
                )
            )

        envelope = AgentResponseEnvelope(
            agent_id="root_cause_agent",
            status="success",
            payload=payload.model_dump(),
            evidence=evidence_refs,
            warnings=[] if retrieved_chunks else ["No matching review evidence chunks retrieved for query."],
            limitations=[
                "Root-cause claims are hypotheses grounded in customer reports and statistical aggregations.",
                "Definitive operational proof requires internal telemetry (POS, KDS, inventory logs)."
            ],
            execution_metadata={
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "model": model_used,
                "retrieved_chunks_count": len(retrieved_chunks),
            }
        )
        return envelope

    def _lookup_operational_context(self, business_id: Optional[str]) -> List[str]:
        """Queries factual business context from DuckDB."""
        if not business_id:
            return []

        facts = []
        p_path = settings.PARQUET_DIR / "clean" / "restaurant_profiles.parquet"
        if p_path.exists():
            try:
                ddb = duckdb.connect()
                row = ddb.execute(f"SELECT name, city, state, stars, review_count, categories FROM '{p_path}' WHERE business_id = '{business_id}'").fetchone()
                if row:
                    facts.append(f"Restaurant '{row[0]}' in {row[1]}, {row[2]} ({row[3]}★ across {row[4]} reviews). Categories: {row[5]}")
            except Exception as e:
                logger.warning(f"Failed to lookup business profile: {e}")
        return facts

    def _build_grounded_hypotheses(
        self,
        problem: str,
        business_id: Optional[str],
        chunks: List[SearchResult],
        operational_facts: List[str],
        upstream_classification: Optional[Dict[str, Any]],
        upstream_trends: Optional[Dict[str, Any]]
    ) -> RootCausePayload:
        """Deterministic baseline synthesizing retrieved chunks into evidence-backed hypotheses."""
        factual_obs = list(operational_facts)
        for c in chunks[:3]:
            factual_obs.append(f"Review chunk {c.chunk_id} ({c.stars}★): \"{c.text_masked[:120]}...\"")

        # Analyze chunk content for support and contradictions
        supporting_ids = [c.chunk_id for c in chunks if c.stars <= 3.0]
        contradictory_ids = [c.chunk_id for c in chunks if c.stars >= 4.0]

        supporting_snippets = [c.text_masked[:150] for c in chunks if c.stars <= 3.0][:3]
        contradictory_snippets = [c.text_masked[:150] for c in chunks if c.stars >= 4.0][:2]

        hypotheses: List[RootCauseHypothesis] = []

        lower_prob = problem.lower()
        if any(w in lower_prob for w in ["wait", "slow", "delay", "service"]):
            hypotheses.append(
                RootCauseHypothesis(
                    hypothesis_id="hyp_1",
                    hypothesized_cause="Front-of-House to Kitchen coordination bottleneck during peak volume periods",
                    distinction_type=DistinctionType.INFERRED_EXPLANATION,
                    supporting_evidence=supporting_snippets if supporting_snippets else [f"Repeated customer complaints regarding {problem}"],
                    supporting_source_ids=supporting_ids,
                    contradictory_evidence=contradictory_snippets,
                    contradictory_source_ids=contradictory_ids,
                    alternative_explanations=[
                        "Understaffed kitchen line during shift turnover",
                        "POS terminal delays or order ticket dispatch lag",
                        "High dine-in demand colliding with takeout delivery queue"
                    ],
                    confidence="high" if len(supporting_ids) >= 3 else "medium",
                    additional_info_needed="POS ticket firing timestamps, KDS ticket completion durations, and floor staffing schedules."
                )
            )
        elif any(w in lower_prob for w in ["cold", "food", "taste", "quality", "burnt", "stale"]):
            hypotheses.append(
                RootCauseHypothesis(
                    hypothesis_id="hyp_1",
                    hypothesized_cause="Food staging/holding delay between kitchen completion and server pickup",
                    distinction_type=DistinctionType.INFERRED_EXPLANATION,
                    supporting_evidence=supporting_snippets,
                    supporting_source_ids=supporting_ids,
                    contradictory_evidence=contradictory_snippets,
                    contradictory_source_ids=contradictory_ids,
                    alternative_explanations=[
                        "Improper heat lamp holding equipment temperature",
                        "Prep recipe batch consistency variation across kitchen shifts"
                    ],
                    confidence="medium",
                    additional_info_needed="Food line pass-through temperature audits and expediter ticket logs."
                )
            )
        else:
            hypotheses.append(
                RootCauseHypothesis(
                    hypothesis_id="hyp_1",
                    hypothesized_cause=f"Operational inconsistency relating to {problem}",
                    distinction_type=DistinctionType.INFERRED_EXPLANATION,
                    supporting_evidence=supporting_snippets,
                    supporting_source_ids=supporting_ids,
                    contradictory_evidence=contradictory_snippets,
                    contradictory_source_ids=contradictory_ids,
                    alternative_explanations=["Individual shift staffing anomaly", "Vendor supplier variation"],
                    confidence="medium" if supporting_ids else "low",
                    additional_info_needed="Internal shift managerial logs and guest feedback records."
                )
            )

        return RootCausePayload(
            problem_statement=problem,
            business_id=business_id,
            retrieved_chunk_count=len(chunks),
            factual_observations=factual_obs,
            hypotheses=hypotheses
        )


def main():
    """Interactive command-line runner for Root-Cause Analysis Agent."""
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    agent = RootCauseAgent()

    # Direct CLI argument support
    if len(sys.argv) > 1:
        cli_prompt = " ".join(sys.argv[1:])
        print(f"\n[ROOT-CAUSE ANALYSIS AGENT] Investigating: \"{cli_prompt}\"...")
        req = RootCauseRequest(problem_statement=cli_prompt, top_k_evidence=4)
        envelope = agent.analyze(req)
        p = envelope.payload
        print("\n" + "=" * 76)
        print("ROOT-CAUSE ANALYSIS FINDINGS")
        print("=" * 76)
        print(f"Retrieved Evidence Chunks: {p.get('retrieved_chunk_count', 0)}")
        for idx, hyp in enumerate(p.get("hypotheses", []), 1):
            print(f"\n[Hypothesis {idx}] {hyp.get('hypothesized_cause')}")
            print(f" • Type:       {hyp.get('distinction_type')} | Confidence: {hyp.get('confidence')}")
            print(f" • Source IDs: {hyp.get('supporting_source_ids')}")
            if hyp.get('supporting_evidence'):
                print(f" • Supporting Review Excerpts:")
                for s in hyp.get('supporting_evidence', []):
                    print(f"    - \"{s}\"")
            if hyp.get('contradictory_evidence'):
                print(f" • Contradictory Evidence ({hyp.get('contradictory_source_ids')}):")
                for c in hyp.get('contradictory_evidence', []):
                    print(f"    - \"{c}\"")
            if hyp.get('alternative_explanations'):
                print(f" • Alternative Causes: {', '.join(hyp.get('alternative_explanations', []))}")
            print(f" • Verification Needed: {hyp.get('additional_info_needed')}")
        print(f"\nModel: {envelope.execution_metadata.get('model')} | Latency: {envelope.execution_metadata.get('latency_ms', 0):.1f}ms")
        print("=" * 76 + "\n")
        return

    print("\n" + "=" * 76)
    print("ROOT-CAUSE ANALYSIS AGENT (Port 8003)")
    print("Investigates restaurant operational failures using Hybrid RAG evidence.")
    print("=" * 76)
    print("Type your operational problem statement (e.g. 'long wait times during dinner')")
    print("Or type 'exit' or 'q' to quit.\n")

    while True:
        try:
            user_input = input("Enter problem statement > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        print(f"\nAnalyzing root causes for: '{user_input}'...")
        req = RootCauseRequest(problem_statement=user_input, top_k_evidence=4)
        envelope = agent.analyze(req)

        print("\n" + "-" * 76)
        print(f"STATUS: {envelope.status.upper()} | Model: {envelope.execution_metadata.get('model', 'unknown')} | Latency: {envelope.execution_metadata.get('latency_ms', 0):.1f}ms")
        print("-" * 76)

        payload = envelope.payload
        hypotheses = payload.get("hypotheses", [])
        print(f"Retrieved Evidence Chunks: {payload.get('retrieved_chunk_count', 0)}")

        for idx, hyp in enumerate(hypotheses, 1):
            print(f"\n[Hypothesis {idx}] {hyp.get('hypothesized_cause')}")
            print(f" • Type: {hyp.get('distinction_type')} | Confidence: {hyp.get('confidence')}")
            print(f" • Supporting Source Chunks: {hyp.get('supporting_source_ids')}")
            print(f" • Supporting Excerpts:")
            for s in hyp.get('supporting_evidence', []):
                print(f"    - \"{s}\"")
            if hyp.get('contradictory_evidence'):
                print(f" • Contradictory Evidence ({hyp.get('contradictory_source_ids')}):")
                for c in hyp.get('contradictory_evidence', []):
                    print(f"    - \"{c}\"")
            if hyp.get('alternative_explanations'):
                print(f" • Alternative Explanations:")
                for alt in hyp.get('alternative_explanations', []):
                    print(f"    - {alt}")
            print(f" • Data Needed to Verify: {hyp.get('additional_info_needed')}")

        print("\n" + "=" * 76 + "\n")


if __name__ == "__main__":
    main()

