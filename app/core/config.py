"""Central application configuration using pydantic-settings."""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR: Path = BASE_DIR / "dataset"
    DATA_STORE_DIR: Path = BASE_DIR / "data_store"
    PROGRESS_DIR: Path = DATA_STORE_DIR / "progress"
    PARQUET_DIR: Path = DATA_STORE_DIR / "parquet"
    SQLITE_DIR: Path = DATA_STORE_DIR / "sqlite"
    VECTORS_DIR: Path = DATA_STORE_DIR / "vectors"
    MODELS_DIR: Path = DATA_STORE_DIR / "models"

    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None

    # Pipeline Mode: "dev" or "full"
    PIPELINE_MODE: str = Field(default="dev", description="'dev' or 'full'")
    DEV_CITY: str = Field(default="Philadelphia", description="City for dev mode")
    DEV_MIN_REVIEWS: int = Field(default=100, description="Min review count for dev restaurant inclusion")
    DEV_MAX_RESTAURANTS: int = Field(default=300, description="Max restaurants in dev mode")
    EMBED_MAX_CHUNKS: int = Field(default=400000, description="Cap on embedded chunks in full mode")

    # LLM Settings
    LLM_MODE: str = Field(default="mock", description="'live' or 'mock'")
    GENERATOR_CANDIDATES: str = "gemini/gemini-2.0-flash,openrouter/meta-llama/llama-3.3-70b-instruct:free"
    JUDGE_CANDIDATES: str = "openrouter/google/gemini-2.0-flash-exp:free,gemini/gemini-2.0-flash"
    LLM_RPM_LIMIT: int = 8
    VIEWER_ROLE_DEFAULT: str = "manager"

    # Model parameters
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_DIM: int = 384
    MAX_SEQ_LENGTH: int = 256
    EMBED_BATCH_SIZE: int = 64

    # Restaurant filter categories
    POSITIVE_CATEGORIES: List[str] = [
        "Restaurants", "Cafes", "Coffee & Tea", "Bakeries", "Bars",
        "Breweries", "Brewpubs", "Pubs", "Food Trucks", "Diners",
        "Fast Food", "Desserts", "Ice Cream & Frozen Yogurt"
    ]
    NEGATIVE_CATEGORIES: List[str] = [
        "Grocery", "Convenience Stores"
    ]


settings = Settings()
