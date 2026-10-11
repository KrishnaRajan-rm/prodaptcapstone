"""Phase 4: PII Gate 2 and Request-Scoped Reversible Token Vault.
Provides pre-LLM reversible redaction, strict allow-listing, residual PII leak detection,
and role-based decoding (authorized: manager/owner/admin; restricted: analyst/restricted).
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import spacy
from rapidfuzz import fuzz

from app.core.config import settings
from app.core.logging import logger

# Regex patterns matching PII Gate 1 conventions plus SSN
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b')
PHONE_REGEX = re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
CARD_REGEX = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
URL_REGEX = re.compile(r'https?://[^\s]+|www\.[^\s]+')

# Authorized roles permitted to decode reversible tokens at response boundary
AUTHORIZED_ROLES: Set[str] = {"manager", "owner", "admin"}

# Singleton cache for vocabulary and spacy model
_nlp_model = None
_food_vocab_cache: Optional[Set[str]] = None
_city_allowlist_cache: Optional[Set[str]] = None


def get_nlp():
    """Lazily loads spaCy en_core_web_sm with parser and lemmatizer disabled."""
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception as e:
            logger.warning(f"Could not load spaCy en_core_web_sm: {e}")
            _nlp_model = None
    return _nlp_model


def get_food_vocab() -> Set[str]:
    """Loads food vocabulary allowlist from data_preprocessing or fallback."""
    global _food_vocab_cache
    if _food_vocab_cache is not None:
        return _food_vocab_cache

    vocab_paths = [
        settings.BASE_DIR / "data_preprocessing" / "preprocessing" / "food_vocab.txt",
        Path(__file__).resolve().parents[2] / "data_preprocessing" / "preprocessing" / "food_vocab.txt",
    ]
    for p in vocab_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                _food_vocab_cache = {line.strip().lower() for line in f if line.strip()}
                return _food_vocab_cache

    _food_vocab_cache = {"pizza", "burger", "pasta", "salad", "steak", "taco", "soup", "sushi"}
    return _food_vocab_cache


def get_city_allowlist() -> Set[str]:
    """Loads known city names to prevent masking locations as people."""
    global _city_allowlist_cache
    if _city_allowlist_cache is not None:
        return _city_allowlist_cache

    cities = {"philadelphia", "tampa", "tucson", "nashville", "indianapolis", "reno", "new orleans", "boise"}
    try:
        import duckdb
        bus_path = settings.PARQUET_DIR / "clean" / "restaurant_profiles.parquet"
        if bus_path.exists():
            con = duckdb.connect()
            c_rows = con.execute(f"SELECT DISTINCT lower(city) FROM read_parquet('{str(bus_path).replace(chr(92), '/')}')").fetchall()
            for r in c_rows:
                if r[0]:
                    cities.add(r[0].strip().lower())
    except Exception as e:
        logger.debug(f"City allowlist lookup from parquet skipped: {e}")

    _city_allowlist_cache = cities
    return _city_allowlist_cache


class TokenVault:
    """Request-scoped reversible token vault.
    Stores bidirectional mappings between raw PII and generated tokens for a single request.
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self._token_to_raw: Dict[str, str] = {}
        self._raw_to_token: Dict[str, str] = {}
        self._counts: Dict[str, int] = {
            "PERSON": 0,
            "EMAIL": 0,
            "PHONE": 0,
            "CARD": 0,
            "SSN": 0,
            "URL": 0,
        }

    def _get_or_create_token(self, pii_type: str, raw_value: str) -> str:
        """Returns existing token for raw_value or creates a new indexed token."""
        raw_key = raw_value.strip()
        if raw_key in self._raw_to_token:
            return self._raw_to_token[raw_key]

        self._counts[pii_type] = self._counts.get(pii_type, 0) + 1
        idx = self._counts[pii_type]
        token = f"[{pii_type}_{idx}]"

        self._token_to_raw[token] = raw_key
        self._raw_to_token[raw_key] = token
        return token

    def mask_text(
        self,
        text: str,
        restaurant_name: Optional[str] = None,
        food_vocab: Optional[Set[str]] = None,
        city_allowlist: Optional[Set[str]] = None,
    ) -> str:
        """Masks PII in a string into reversible request-scoped tokens."""
        if not text or not isinstance(text, str):
            return text

        vocab = food_vocab if food_vocab is not None else get_food_vocab()
        cities = city_allowlist if city_allowlist is not None else get_city_allowlist()
        rest_lower = (restaurant_name or "").strip().lower()
        rest_tokens = set(re.findall(r'\b\w+\b', rest_lower)) if rest_lower else set()

        # Step 1: Regex masking for deterministic structured PII
        def replace_email(match):
            return self._get_or_create_token("EMAIL", match.group(0))

        def replace_phone(match):
            return self._get_or_create_token("PHONE", match.group(0))

        def replace_card(match):
            return self._get_or_create_token("CARD", match.group(0))

        def replace_ssn(match):
            return self._get_or_create_token("SSN", match.group(0))

        def replace_url(match):
            return self._get_or_create_token("URL", match.group(0))

        masked = EMAIL_REGEX.sub(replace_email, text)
        masked = PHONE_REGEX.sub(replace_phone, masked)
        masked = CARD_REGEX.sub(replace_card, masked)
        masked = SSN_REGEX.sub(replace_ssn, masked)
        masked = URL_REGEX.sub(replace_url, masked)

        # Step 2: spaCy NER for PERSON
        nlp = get_nlp()
        if nlp is not None:
            try:
                doc = nlp(masked)
                spans_to_mask = []
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        ent_raw = ent.text
                        ent_lower = ent_raw.lower().strip()

                        # Skip already masked tokens like [PERSON_1] or [EMAIL_1]
                        if ent_raw.startswith("[") and ent_raw.endswith("]"):
                            continue

                        # Check allow-lists
                        if ent_lower in vocab:
                            continue
                        if ent_lower in cities:
                            continue
                        if rest_tokens and (ent_lower in rest_tokens or ent_lower == rest_lower):
                            continue
                        if rest_lower and fuzz.ratio(ent_lower, rest_lower) >= 90:
                            continue
                        ent_words = ent_lower.split()
                        if any(w in vocab for w in ent_words):
                            continue

                        spans_to_mask.append((ent.start_char, ent.end_char, ent_raw))

                # Replace right-to-left to preserve character offsets
                for start, end, raw_name in sorted(spans_to_mask, key=lambda x: x[0], reverse=True):
                    token = self._get_or_create_token("PERSON", raw_name)
                    masked = masked[:start] + token + masked[end:]
            except Exception as e:
                logger.warning(f"spaCy NER masking encountered warning: {e}")

        return masked

    def has_residual_pii(self, text: str) -> Tuple[bool, List[str]]:
        """Scans for unmasked high-risk structured PII (emails, phones, cards, SSNs).
        Returns: (True, [leaking_types]) if residual unmasked PII found.
        """
        if not text or not isinstance(text, str):
            return False, []

        leaks: List[str] = []

        # Check for unmasked emails (excluding vault tokens)
        for m in EMAIL_REGEX.finditer(text):
            val = m.group(0)
            if not (val.startswith("[") and val.endswith("]")):
                leaks.append(f"EMAIL:{val[:3]}***")

        # Check for unmasked phones
        for m in PHONE_REGEX.finditer(text):
            val = m.group(0)
            if not (val.startswith("[") and val.endswith("]")):
                leaks.append(f"PHONE:***{val[-4:]}")

        # Check for unmasked credit cards
        for m in CARD_REGEX.finditer(text):
            val = m.group(0)
            if not (val.startswith("[") and val.endswith("]")):
                leaks.append("CARD:****")

        # Check for unmasked SSNs
        for m in SSN_REGEX.finditer(text):
            val = m.group(0)
            if not (val.startswith("[") and val.endswith("]")):
                leaks.append("SSN:***")

        return (len(leaks) > 0), leaks

    def unmask_text(self, text: str, role: Optional[str] = None) -> str:
        """Decodes request-scoped tokens back to raw text ONLY if role is authorized.
        Restricted roles receive masked content. Tokens from other requests are untouched.
        """
        if not text or not isinstance(text, str):
            return text

        normalized_role = (role or "").strip().lower()
        if normalized_role not in AUTHORIZED_ROLES:
            return text

        # Replace all known tokens belonging to this vault
        result = text
        for token, raw in self._token_to_raw.items():
            if token in result:
                result = result.replace(token, raw)

        return result

    def unmask_data(self, data: Any, role: Optional[str] = None) -> Any:
        """Recursively traverses dictionaries, lists, and strings to unmask for authorized role."""
        normalized_role = (role or "").strip().lower()
        if normalized_role not in AUTHORIZED_ROLES:
            return data

        if isinstance(data, str):
            return self.unmask_text(data, role)
        elif isinstance(data, dict):
            return {k: self.unmask_data(v, role) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.unmask_data(item, role) for item in data]
        return data

    @property
    def token_count(self) -> int:
        return len(self._token_to_raw)

    def get_summary(self) -> Dict[str, Any]:
        """Safe summary of token counts without revealing sensitive raw values."""
        return {
            "request_id": self.request_id,
            "total_tokens_masked": len(self._token_to_raw),
            "counts_by_type": dict(self._counts),
        }
