"""Custom exceptions for the Restaurant Operations Assistant application."""


class RestaurantOpsError(Exception):
    """Base exception for all domain-specific errors."""
    pass


class ConfigurationError(RestaurantOpsError):
    """Raised when application configuration is missing or invalid."""
    pass


class DataIngestionError(RestaurantOpsError):
    """Raised during the data ingestion pipeline."""
    pass


class DataCleaningError(RestaurantOpsError):
    """Raised during data cleaning or normalization."""
    pass


class PreprocessingError(RestaurantOpsError):
    """Raised during feature extraction, aspect tagging, or chunking."""
    pass


class PIIError(RestaurantOpsError):
    """Raised during PII detection, redaction, or unmasking."""
    pass


class StorageError(RestaurantOpsError):
    """Raised during SQLite, FTS5, or Vector storage operations."""
    pass


class RetrievalError(RestaurantOpsError):
    """Raised during dense, BM25, or hybrid search execution."""
    pass


class LLMGatewayError(RestaurantOpsError):
    """Raised when all LLM providers or fallbacks fail."""
    pass


class GuardrailValidationError(RestaurantOpsError):
    """Raised when an input query or prompt violates safety guardrails."""
    pass
