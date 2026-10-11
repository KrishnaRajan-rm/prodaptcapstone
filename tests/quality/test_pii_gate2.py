"""Unit Tests for Phase 4 PII Gate 2 and Reversible Token Vault.
Covers scenarios 1, 2, 3, 4, 17, 18 from Quality specifications:
1. PII is detected and masked before LLM invocation.
2. Residual PII blocks the affected model call.
3. Request-scoped tokens cannot be decoded across requests.
4. Unauthorized roles cannot recover masked information.
17. Only authorized final responses are decoded.
18. Logs and error messages do not expose raw PII.
"""

import pytest
from app.core.pii.gate2 import TokenVault, AUTHORIZED_ROLES


def test_01_pii_detected_and_masked_before_llm():
    """1. PII is detected and masked with reversible tokens before LLM invocation."""
    vault = TokenVault(request_id="req_001")
    raw_query = "Contact customer Alice Smith at alice.smith@example.com or 215-555-0199 regarding order."

    masked = vault.mask_text(raw_query)

    assert "[EMAIL_1]" in masked
    assert "[PHONE_1]" in masked
    assert "alice.smith@example.com" not in masked
    assert "215-555-0199" not in masked
    assert vault.token_count >= 2


def test_02_residual_pii_blocks_affected_model_call():
    """2. Residual PII blocks the affected model call (fails closed)."""
    vault = TokenVault(request_id="req_002")

    # Clean text without residual leaks
    clean_text = "The pasta was delicious and service was quick."
    leaks_found, _ = vault.has_residual_pii(clean_text)
    assert not leaks_found

    # Text containing unmasked sensitive email or credit card
    leaking_text = "Here is my unmasked credit card 4111-2222-3333-4444 and email test@leak.com"
    leaks_found, leak_types = vault.has_residual_pii(leaking_text)
    assert leaks_found
    assert any("CARD" in lt for lt in leak_types)
    assert any("EMAIL" in lt for lt in leak_types)


def test_03_request_scoped_tokens_cannot_decode_across_requests():
    """3. Request-scoped tokens cannot be decoded across different requests."""
    vault_a = TokenVault(request_id="req_A")
    vault_b = TokenVault(request_id="req_B")

    raw_text = "Call John Doe at 555-123-4567."
    masked_a = vault_a.mask_text(raw_text)

    # Vault A can decode with manager role
    decoded_a = vault_a.unmask_text(masked_a, role="manager")
    assert "555-123-4567" in decoded_a

    # Vault B has no mapping for Vault A's tokens; decoding MUST NOT recover raw PII
    decoded_b = vault_b.unmask_text(masked_a, role="manager")
    assert decoded_b == masked_a
    assert "555-123-4567" not in decoded_b


def test_04_unauthorized_roles_cannot_recover_masked_information():
    """4. Unauthorized roles (analyst, restricted, guest, empty) receive masked content."""
    vault = TokenVault(request_id="req_004")
    raw = "Refund Bob at bob@restaurant.com."
    masked = vault.mask_text(raw)

    for unauthorized_role in ["analyst", "restricted", "viewer", "guest", "", None]:
        unmasked = vault.unmask_text(masked, role=unauthorized_role)
        assert unmasked == masked
        assert "bob@restaurant.com" not in unmasked


def test_17_only_authorized_final_responses_decoded():
    """17. Only authorized roles (manager, owner, admin) decode reversible tokens."""
    vault = TokenVault(request_id="req_017")
    raw = "Manager Alice at 555-987-6543 handles escalations."
    masked = vault.mask_text(raw)

    for authorized_role in ["manager", "owner", "admin", "MANAGER", "Admin"]:
        unmasked = vault.unmask_text(masked, role=authorized_role)
        assert "555-987-6543" in unmasked

    # Nested data structures (dictionaries and lists)
    data = {
        "problem": masked,
        "actions": [{"note": f"Contact: {masked}"}],
    }
    unmasked_dict = vault.unmask_data(data, role="manager")
    assert "555-987-6543" in unmasked_dict["problem"]
    assert "555-987-6543" in unmasked_dict["actions"][0]["note"]


def test_18_logs_and_summaries_do_not_expose_raw_pii():
    """18. Vault summaries, logs, and representations do not expose unmasked raw PII."""
    vault = TokenVault(request_id="req_018")
    raw_email = "sensitive_owner@private-domain.org"
    vault.mask_text(f"Send info to {raw_email}")

    summary = vault.get_summary()
    summary_str = str(summary)

    assert raw_email not in summary_str
    assert "private-domain.org" not in summary_str
    assert summary["request_id"] == "req_018"
    assert summary["total_tokens_masked"] >= 1
