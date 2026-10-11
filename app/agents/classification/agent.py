"""Classification Agent (Service Port 8001).
Analyzes restaurant reviews and complaints to determine categories, aspects, sentiment, and severity.
Built using Google ADK with robust deterministic fallback for input validation and offline testing.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import time
import re
from typing import Any, Dict, List, Optional
from google.adk.agents import LlmAgent

from app.core.logging import logger
from app.agents.schemas import (
    ClassificationRequest,
    ClassificationPayload,
    AgentResponseEnvelope,
    EvidenceRef,
    ComplaintCategory,
    AspectTag,
    SentimentType,
    SeverityLevel,
)
from app.agents.llm import get_llm_client


CLASSIFICATION_INSTRUCTION = """You are a specialist Restaurant Complaint Classification Agent.
Analyze the provided customer review text and return ONLY a valid JSON object matching this schema:
{
  "request_id": "<request_id>",
  "categories": ["<one or more from: Service speed, Food quality, Cleanliness, Pricing, Staff behaviour, Other / uncategorized>"],
  "aspects": ["<one or more from: Waiting time, Food temperature, Taste and freshness, Hygiene, Staff responsiveness, Portion size, Value for money, Other>"],
  "sentiment": "<one of: positive, neutral, negative, mixed>",
  "severity": "<one of: low, medium, high, critical>",
  "severity_reason": "<documented reason why this severity was assigned, distinguishing sentiment from operational severity>",
  "short_explanation": "<concise analytical summary>",
  "evidence_snippets": ["<exact quotes from the review supporting the classification>"],
  "is_multi_aspect": <true if multiple distinct issues/aspects exist, else false>
}

Severity Guidelines:
- critical: Severe health/safety hazards (food poisoning, hospital, roaches/rodents, raw chicken, violence).
- high: Major operational breakdowns (hostile staff, extreme wait >60min, major cleanliness failure, refused refund).
- medium: Common service delays (20-40 min wait), lukewarm food, minor order errors, poor value for price.
- low: Minor personal taste preference, ambient noise, or positive/neutral reviews with no operational danger.
IMPORTANT: Do not confuse negative sentiment with critical severity. A customer hating a decor is negative sentiment but LOW severity.
"""


class ClassificationAgent:
    """Specialist Agent for complaint classification and aspect-based severity analysis."""

    def __init__(self, model_name: str = "gemini-3.8-flash"):
        self.model_name = model_name
        self.llm_client = get_llm_client()

        # Google ADK LlmAgent wrapper
        self.adk_agent = LlmAgent(
            name="classification_agent",
            model=model_name,
            instruction=CLASSIFICATION_INSTRUCTION,
            description="Classifies restaurant customer complaints, aspects, sentiment, and severity with grounded evidence."
        )

    def analyze(self, request: ClassificationRequest) -> AgentResponseEnvelope:
        """Executes full classification analysis with validation and evidence extraction."""
        start_time = time.time()
        text = (request.text or "").strip()

        # Handle empty or malformed input gracefully
        if not text:
            return AgentResponseEnvelope(
                agent_id="classification_agent",
                status="insufficient_data",
                warnings=["Input text is empty or missing."],
                limitations=["Cannot classify empty review text."],
                payload=ClassificationPayload(
                    request_id=request.review_id or "empty",
                    categories=[ComplaintCategory.OTHER.value],
                    aspects=[AspectTag.OTHER.value],
                    sentiment=SentimentType.NEUTRAL,
                    severity=SeverityLevel.LOW,
                    severity_reason="No complaint content provided to evaluate.",
                    short_explanation="Empty review input.",
                    evidence_snippets=[],
                    is_multi_aspect=False
                ).model_dump(),
                execution_metadata={"latency_ms": (time.time() - start_time) * 1000, "model": "rule_based_fallback"}
            )

        # Deterministic heuristic baseline as fallback / mock
        heuristic_payload = self._deterministic_classify(request.review_id or "req", text)

        # Attempt LLM generation
        try:
            prompt = f"Review ID: {request.review_id or 'req'}\nReview Text:\n\"\"\"{text}\"\"\""
            llm_result, model_used = self.llm_client.generate_json(
                prompt=prompt,
                system_instruction=CLASSIFICATION_INSTRUCTION,
                model_name=self.model_name,
                schema_cls=ClassificationPayload,
                mock_response=heuristic_payload.model_dump()
            )
            payload = ClassificationPayload.model_validate(llm_result)
        except Exception as e:
            logger.exception("Classification LLM failed; no heuristic response will be returned.")
            raise RuntimeError(f"Classification analysis failed: {e}") from e

        # Construct traceable evidence references
        evidence_refs: List[EvidenceRef] = []
        for idx, snippet in enumerate(payload.evidence_snippets):
            evidence_refs.append(
                EvidenceRef(
                    source_id=f"{request.review_id or 'rev'}_ev_{idx}",
                    chunk_id=request.chunk_id,
                    review_id=request.review_id,
                    business_id=request.business_id,
                    snippet=snippet,
                    is_supporting=True,
                    metadata={"aspects": payload.aspects, "severity": payload.severity.value}
                )
            )

        # Build standard envelope
        envelope = AgentResponseEnvelope(
            agent_id="classification_agent",
            status="success",
            payload=payload.model_dump(),
            evidence=evidence_refs,
            warnings=[],
            limitations=["Classification relies on reviewer perceptions and text content."],
            execution_metadata={
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "model": model_used,
                "request_id": request.review_id,
            }
        )
        return envelope

    def _deterministic_classify(self, request_id: str, text: str) -> ClassificationPayload:
        """Rule-based classifier ensuring 100% deterministic fallback and offline testability."""
        lower = text.lower()
        categories: List[str] = []
        aspects: List[str] = []
        evidence_snippets: List[str] = []

        # Service speed & Waiting time
        if any(w in lower for w in ["wait", "slow", "delayed", "forever", "minutes", "hour", "line", "took long"]):
            categories.append(ComplaintCategory.SERVICE_SPEED.value)
            aspects.append(AspectTag.WAITING_TIME.value)
            m = re.search(r"([^.?!]*(?:wait|slow|delayed|minute|hour)[^.?!]*)", text, re.IGNORECASE)
            if m:
                evidence_snippets.append(m.group(1).strip())

        # Food quality & Taste / Temperature
        if any(w in lower for w in ["cold", "lukewarm"]):
            categories.append(ComplaintCategory.FOOD_QUALITY.value)
            aspects.append(AspectTag.FOOD_TEMPERATURE.value)
            m = re.search(r"([^.?!]*(?:cold|lukewarm)[^.?!]*)", text, re.IGNORECASE)
            if m:
                evidence_snippets.append(m.group(1).strip())

        if any(w in lower for w in ["bland", "taste", "stale", "flavor", "salty", "undercooked", "burnt", "fresh"]):
            if ComplaintCategory.FOOD_QUALITY.value not in categories:
                categories.append(ComplaintCategory.FOOD_QUALITY.value)
            aspects.append(AspectTag.TASTE_FRESHNESS.value)
            m = re.search(r"([^.?!]*(?:taste|stale|bland|burnt|salty|undercooked)[^.?!]*)", text, re.IGNORECASE)
            if m:
                evidence_snippets.append(m.group(1).strip())

        # Cleanliness & Hygiene
        if any(w in lower for w in ["dirty", "filthy", "smell", "roach", "bug", "rat", "hair", "gross", "cleanliness", "poison"]):
            categories.append(ComplaintCategory.CLEANLINESS.value)
            aspects.append(AspectTag.HYGIENE.value)
            m = re.search(r"([^.?!]*(?:dirty|filthy|roach|bug|hair|gross|clean|poison)[^.?!]*)", text, re.IGNORECASE)
            if m:
                evidence_snippets.append(m.group(1).strip())

        # Pricing & Value for money
        if any(w in lower for w in ["expensive", "overpriced", "ripoff", "price", "cost", "bill", "money", "worth"]):
            categories.append(ComplaintCategory.PRICING.value)
            aspects.append(AspectTag.VALUE_FOR_MONEY.value)
            m = re.search(r"([^.?!]*(?:expensive|overpriced|price|cost|bill|worth)[^.?!]*)", text, re.IGNORECASE)
            if m:
                evidence_snippets.append(m.group(1).strip())

        # Staff behaviour & Responsiveness
        if any(w in lower for w in ["rude", "attitude", "ignored", "hostile", "impolite", "unfriendly", "manager", "server"]):
            categories.append(ComplaintCategory.STAFF_BEHAVIOUR.value)
            aspects.append(AspectTag.STAFF_RESPONSIVENESS.value)
            m = re.search(r"([^.?!]*(?:rude|attitude|ignored|hostile|unfriendly|server)[^.?!]*)", text, re.IGNORECASE)
            if m:
                evidence_snippets.append(m.group(1).strip())

        # Portion size
        if any(w in lower for w in ["portion", "tiny", "small amount", "bite size"]):
            aspects.append(AspectTag.PORTION_SIZE.value)

        # Fallback if uncategorized
        if not categories:
            categories.append(ComplaintCategory.OTHER.value)
        if not aspects:
            aspects.append(AspectTag.OTHER.value)
        if not evidence_snippets and text:
            evidence_snippets.append(text[:120])

        # Sentiment determination
        neg_words = ["bad", "horrible", "terrible", "awful", "worst", "hate", "disgusting", "poor", "never", "slow", "cold", "rude", "dirty"]
        pos_words = ["good", "great", "excellent", "amazing", "delicious", "love", "friendly", "fast", "clean", "wonderful", "best"]

        n_neg = sum(1 for w in neg_words if w in lower)
        n_pos = sum(1 for w in pos_words if w in lower)

        if n_pos > 0 and n_neg > 0:
            sentiment = SentimentType.MIXED
        elif n_neg > 0:
            sentiment = SentimentType.NEGATIVE
        elif n_pos > 0:
            sentiment = SentimentType.POSITIVE
        else:
            sentiment = SentimentType.NEUTRAL

        # Severity assessment
        if any(w in lower for w in ["poison", "hospital", "roach", "cockroach", "rat", "police", "threaten", "raw chicken"]):
            severity = SeverityLevel.CRITICAL
            reason = "Critical health, safety, or legal hazard reported."
        elif any(w in lower for w in ["filthy", "disgusting", "screamed", "unacceptable", "refused refund", "extremely rude", "2 hours"]):
            severity = SeverityLevel.HIGH
            reason = "Significant operational failure with acute customer dissatisfaction."
        elif n_neg > 0:
            severity = SeverityLevel.MEDIUM
            reason = "Operational service defect affecting guest experience."
        else:
            severity = SeverityLevel.LOW
            reason = "No high-risk operational defect detected."

        return ClassificationPayload(
            request_id=request_id,
            categories=list(dict.fromkeys(categories)),
            aspects=list(dict.fromkeys(aspects)),
            sentiment=sentiment,
            severity=severity,
            severity_reason=reason,
            short_explanation=f"Identified {len(categories)} categories and {len(aspects)} operational aspects.",
            evidence_snippets=evidence_snippets,
            is_multi_aspect=len(aspects) > 1
        )


def main():
    """Interactive command-line runner for Classification Agent."""
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    agent = ClassificationAgent()

    # Direct CLI argument support
    if len(sys.argv) > 1:
        cli_prompt = " ".join(sys.argv[1:])
        print(f"\n[CLASSIFICATION AGENT] Analyzing: \"{cli_prompt}\"...")
        req = ClassificationRequest(review_id="cli_rev", text=cli_prompt)
        envelope = agent.analyze(req)
        p = envelope.payload
        print("\n" + "=" * 76)
        print("CLASSIFICATION RESULTS")
        print("=" * 76)
        print(f"Categories: {p.get('categories')}")
        print(f"Aspects:    {p.get('aspects')}")
        print(f"Sentiment:  {p.get('sentiment')}")
        print(f"Severity:   {p.get('severity')} ({p.get('severity_reason')})")
        print(f"Summary:    {p.get('short_explanation')}")
        if p.get('evidence_snippets'):
            print("\nEvidence Excerpts:")
            for s in p.get('evidence_snippets', []):
                print(f"  • \"{s}\"")
        print(f"\nModel: {envelope.execution_metadata.get('model')} | Latency: {envelope.execution_metadata.get('latency_ms', 0):.1f}ms")
        print("=" * 76 + "\n")
        return

    print("\n" + "=" * 76)
    print("CLASSIFICATION AGENT (Port 8001)")
    print("Analyzes restaurant reviews for categories, aspects, sentiment, and severity.")
    print("=" * 76)
    print("Type a customer review to classify (e.g. 'Food was cold and waiter was rude')")
    print("Or type 'exit' or 'q' to quit.\n")

    while True:
        try:
            user_input = input("Enter review text > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        print("\nClassifying review...")
        req = ClassificationRequest(review_id="cli_rev", text=user_input)
        envelope = agent.analyze(req)

        print("\n" + "-" * 76)
        print(f"STATUS: {envelope.status.upper()} | Model: {envelope.execution_metadata.get('model', 'unknown')} | Latency: {envelope.execution_metadata.get('latency_ms', 0):.1f}ms")
        print("-" * 76)

        p = envelope.payload
        print(f"Categories: {p.get('categories')}")
        print(f"Aspects:    {p.get('aspects')}")
        print(f"Sentiment:  {p.get('sentiment')}")
        print(f"Severity:   {p.get('severity')} ({p.get('severity_reason')})")
        print(f"Summary:    {p.get('short_explanation')}")
        if p.get('evidence_snippets'):
            print("Evidence Excerpts:")
            for snip in p.get('evidence_snippets', []):
                print(f"  • \"{snip}\"")
        print("\n" + "=" * 76 + "\n")


if __name__ == "__main__":
    main()

