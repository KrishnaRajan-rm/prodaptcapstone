"""Phase 4 Grounding Validation: Source ID verification, quote matching, and business isolation."""

import re
from typing import Any, Dict, List, Optional, Set
from rapidfuzz import fuzz

from app.quality.schemas import (
    GroundingCheckResult,
    GroundingFinding,
    QualityStatus,
)


def normalize_text(text: Optional[str]) -> str:
    """Normalizes whitespace and casing for reliable excerpt comparison."""
    if not text:
        return ""
    # Strip markdown quotes and brackets
    cleaned = re.sub(r'[\'"`\[\]]', ' ', text)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
    return cleaned


def excerpt_in_evidence(excerpt: str, evidence_texts: List[str]) -> float:
    """Computes the best normalized match ratio for an excerpt across evidence texts."""
    norm_excerpt = normalize_text(excerpt)
    if not norm_excerpt:
        return 1.0

    best_ratio = 0.0
    for ev in evidence_texts:
        norm_ev = normalize_text(ev)
        if norm_excerpt in norm_ev:
            return 1.0
        # Partial token match or rapidfuzz partial ratio
        ratio = fuzz.partial_ratio(norm_excerpt, norm_ev) / 100.0
        if ratio > best_ratio:
            best_ratio = ratio

    return best_ratio


def validate_grounding(
    target_business_id: Optional[str],
    retrieved_evidence: List[Dict[str, Any]],
    classification_payload: Optional[Dict[str, Any]],
    root_cause_payload: Optional[Dict[str, Any]],
    action_planner_payload: Optional[Dict[str, Any]],
) -> GroundingCheckResult:
    """Deterministically validates all cited sources, excerpts, and business isolation."""
    result = GroundingCheckResult()

    # Index retrieved evidence by chunk_id, review_id, and positional indices
    evidence_by_id: Dict[str, Dict[str, Any]] = {}
    evidence_texts: List[str] = []

    for idx, chunk in enumerate(retrieved_evidence or []):
        cid = str(chunk.get("chunk_id", "")).strip()
        rid = str(chunk.get("review_id", "")).strip()
        if cid:
            evidence_by_id[cid] = chunk
            evidence_by_id[cid.lower()] = chunk
            clean_c = cid.replace("chk_", "").replace("chk-", "").strip()
            evidence_by_id[clean_c] = chunk
            evidence_by_id[f"chk_{clean_c}"] = chunk
            evidence_by_id[f"chunk_{clean_c}"] = chunk
            evidence_by_id[f"chunk {clean_c}".lower()] = chunk
            evidence_by_id[f"chunk {cid}".lower()] = chunk
        if rid:
            evidence_by_id[rid] = chunk
            evidence_by_id[rid.lower()] = chunk
        # Positional index representations
        idx_str = str(idx + 1)
        evidence_by_id[idx_str] = chunk
        evidence_by_id[f"[{idx_str}]"] = chunk
        evidence_by_id[f"chunk {idx_str}".lower()] = chunk
        evidence_by_id[f"src_{idx_str}".lower()] = chunk
        evidence_by_id[f"source_{idx_str}".lower()] = chunk
        # 0-based index
        idx0_str = str(idx)
        evidence_by_id[idx0_str] = chunk
        evidence_by_id[f"[{idx0_str}]"] = chunk
        evidence_by_id[f"chunk {idx0_str}".lower()] = chunk
        evidence_by_id[f"src_{idx0_str}".lower()] = chunk

        txt = chunk.get("text_masked") or chunk.get("text") or ""
        if txt:
            evidence_texts.append(txt)

    # 1. Collect cited sources and excerpts from Root Cause
    citations_to_check: List[Dict[str, Any]] = []

    if root_cause_payload and isinstance(root_cause_payload, dict):
        hypotheses = root_cause_payload.get("hypotheses", [])
        for hyp in hypotheses:
            hid = hyp.get("hypothesis_id", "hyp")
            source_ids = hyp.get("supporting_source_ids", []) or []
            excerpts = hyp.get("supporting_evidence", []) or []

            for sid in source_ids:
                citations_to_check.append({
                    "source_id": str(sid).strip(),
                    "claim_id": hid,
                    "excerpts": excerpts,
                    "context": "root_cause_hypothesis"
                })

    # Also check Action Planner citations if present
    if action_planner_payload and isinstance(action_planner_payload, dict):
        ranked = action_planner_payload.get("ranked_actions", [])
        for act in ranked:
            sids = act.get("source_ids", []) or []
            rationale = act.get("evidence_rationale", "")
            for sid in sids:
                citations_to_check.append({
                    "source_id": str(sid).strip(),
                    "claim_id": str(act.get("rank", "act")),
                    "excerpts": [rationale] if rationale else [],
                    "context": "action_planner_recommendation"
                })

    # Also check Classification evidence snippets if source refs present
    if classification_payload and isinstance(classification_payload, dict):
        snippets = classification_payload.get("evidence_snippets", [])
        if snippets and evidence_texts:
            for snip in snippets:
                overlap = excerpt_in_evidence(snip, evidence_texts)
                if overlap < 0.65:
                    result.unrelated_citations.append(f"Classification snippet '{snip[:40]}' not found in evidence")

    result.total_citations = len(citations_to_check)

    # Evaluate each citation
    for item in citations_to_check:
        sid = item["source_id"]
        excerpts = item["excerpts"]

        finding = GroundingFinding(source_id=sid)

        # Flexible matching
        matched_chunk = evidence_by_id.get(sid)
        if not matched_chunk:
            norm_sid = sid.strip().lower()
            clean_s = re.sub(r'^[\[\(\"\'\s]+|[\]\)\"\'\s]+$', '', norm_sid)
            matched_chunk = evidence_by_id.get(norm_sid) or evidence_by_id.get(clean_s)
            if not matched_chunk:
                stripped_s = clean_s.replace("chunk", "").replace("chk_", "").replace("src_", "").strip(" _-:")
                matched_chunk = evidence_by_id.get(stripped_s)

        # Check A1: Does source ID exist in retrieved evidence?
        if not matched_chunk:
            finding.exists = False
            finding.supported = False
            finding.note = f"Source ID '{sid}' does not exist in retrieved evidence context."
            result.missing_citations.append(sid)
            result.findings.append(finding)
            continue

        chunk = matched_chunk

        # Check A3: Business ID isolation
        chunk_biz = str(chunk.get("business_id", "")).strip()
        req_biz = str(target_business_id).strip() if target_business_id else None

        if req_biz and chunk_biz and req_biz != chunk_biz:
            finding.business_id_match = False
            finding.supported = False
            finding.note = f"Business ID mismatch: evidence belongs to '{chunk_biz}', requested '{req_biz}'."
            result.business_id_mismatches.append(f"{sid} ({chunk_biz} != {req_biz})")
            result.findings.append(finding)
            continue

        # Check A2: Excerpt matching
        chunk_text = chunk.get("text_masked") or chunk.get("text") or ""
        best_overlap = 1.0
        if excerpts:
            best_overlap = max((excerpt_in_evidence(exc, [chunk_text]) for exc in excerpts), default=0.0)

        finding.excerpt_overlap = best_overlap
        if best_overlap < 0.65 and excerpts:
            finding.supported = False
            finding.note = f"Quoted excerpt does not match source chunk text (overlap: {best_overlap:.2f})."
            result.unrelated_citations.append(sid)
        else:
            finding.supported = True
            finding.note = "Verified against source chunk."
            result.verified_citations += 1

        result.findings.append(finding)

    # Determine status
    if result.business_id_mismatches:
        result.status = QualityStatus.BLOCKED
    elif result.missing_citations or result.unrelated_citations:
        result.status = QualityStatus.FLAGGED
    else:
        result.status = QualityStatus.PASSED

    return result
