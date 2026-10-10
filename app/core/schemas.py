"""Common Pydantic v2 schemas used across data pipelines, retrieval, and API."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class BusinessSchema(BaseModel):
    business_id: str
    name: str
    address: Optional[str] = None
    city: str
    state: str
    postal_code: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    stars: float
    review_count: int
    is_open: int
    categories: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    hours: Optional[Dict[str, str]] = None


class ReviewSchema(BaseModel):
    review_id: str
    user_id: str
    business_id: str
    stars: float
    useful: int = 0
    funny: int = 0
    cool: int = 0
    text: str
    date: str


class ChunkSchema(BaseModel):
    chunk_id: str
    review_id: str
    business_id: str
    restaurant_name: str
    city: str
    state: str
    categories: str
    price_range: Optional[int] = None
    is_open: int = 1
    stars: float
    business_stars: float
    date: str
    year_month: str
    credibility: float
    text_masked: str
    aspects: List[str] = Field(default_factory=list)
    aspect_sentiments: Dict[str, float] = Field(default_factory=dict)
    n_tokens: int


class GuardrailDecision(BaseModel):
    allowed: bool
    reasons: List[str] = Field(default_factory=list)
    sanitized_query: str
