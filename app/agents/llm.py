"""LLM integration for Specialist Agents with Google GenAI and OpenRouter failover.
Primary: Gemini 3.8 Flash (via GEMINI_API_KEY / GOOGLE_API_KEY)
Failover: OpenRouter Free Models (liquid/lfm, nvidia/nemotron, apodex)
Mock Mode: Supported for deterministic offline unit testing.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Type, TypeVar
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from app.core.logging import logger

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Default free models for failover
FREE_OPENROUTER_FALLBACKS = [
    os.getenv("OPENROUTER_MODEL", "liquid/lfm-2.5-2.6b:free"),
    "nvidia/nemotron-3-super-120b-a12b:free",
    "apodex/apodex-1.1-mini:free",
    "poolside/laguna-s-2.1:free",
]


class LLMClient:
    """Multi-tiered LLM client supporting Gemini primary, OpenRouter failover, and offline mock."""

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.mock_mode = os.getenv("MOCK_LLM", "0").lower() in ("1", "true", "yes")

        self._genai_client = None
        if self.gemini_key and not self.mock_mode:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.gemini_key)
            except Exception as e:
                logger.warning(f"Could not initialize google.genai client: {e}")

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "",
        model_name: str = "gemini-3.8-flash",
        schema_cls: Optional[Type[T]] = None,
        mock_response: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], str]:
        """Generates structured JSON with schema validation, failing over gracefully.
        Returns: (parsed_json_dict, model_name_used)
        """
        if self.mock_mode and mock_response is not None:
            return mock_response, "mock_llm"

        failure_reasons: List[str] = []

        # Tier 1: Try Gemini API
        if self._genai_client:
            try:
                contents = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
                response = self._genai_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                text = response.text or ""
                parsed = self._extract_json(text)
                if parsed:
                    if schema_cls:
                        validated = schema_cls.model_validate(parsed)
                        return validated.model_dump(), model_name
                    return parsed, model_name
                failure_reasons.append("Gemini returned no valid JSON response.")
            except Exception as e:
                failure_reasons.append(f"Gemini failed: {type(e).__name__}: {e}")
                logger.warning(f"Gemini generation failed ({e}), attempting OpenRouter failover...")
        else:
            failure_reasons.append("Gemini is not configured with an API key.")

        # Tier 2: OpenRouter Free Models Failover
        if self.openrouter_key:
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            }
            system_msg = system_instruction or "You are an expert restaurant operations analyst. Always output valid JSON only."
            for candidate in FREE_OPENROUTER_FALLBACKS:
                if not candidate:
                    continue
                try:
                    payload = {
                        "model": candidate,
                        "messages": [
                            {"role": "system", "content": system_msg + "\nReply ONLY with a raw valid JSON object. No markdown ticks, no preamble."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 2048,
                    }
                    resp = requests.post(
                        f"{self.openrouter_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=40
                    )
                    resp.raise_for_status()
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            raw_text = choices[0].get("message", {}).get("content")
                            if raw_text:
                                parsed = self._extract_json(raw_text)
                                if parsed:
                                    if schema_cls:
                                        validated = schema_cls.model_validate(parsed)
                                        return validated.model_dump(), f"openrouter/{candidate}"
                                    return parsed, f"openrouter/{candidate}"
                        failure_reasons.append(f"OpenRouter model {candidate} returned no valid JSON response.")
                except Exception as ex:
                    failure_reasons.append(f"OpenRouter model {candidate} failed: {type(ex).__name__}: {ex}")
                    logger.warning(f"OpenRouter candidate {candidate} failed: {ex}")
                    continue
        else:
            failure_reasons.append("OpenRouter is not configured with an API key.")

        details = "; ".join(failure_reasons[-4:])
        raise RuntimeError(f"All live LLM providers failed. {details}")

    def _extract_json(self, text: Optional[str]) -> Optional[Dict[str, Any]]:
        """Extracts JSON dict from code fences or raw text."""
        if not text or not isinstance(text, str):
            return None
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            val = json.loads(cleaned)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

        # Try regex search for first { and matching }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                candidate = cleaned[start:end+1]
                return json.loads(candidate)
            except Exception:
                pass
        return None


# Global singleton instance
_llm_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
