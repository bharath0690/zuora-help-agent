"""
Configuration management for Zuora Help Agent
Loads settings from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application Settings
    APP_NAME: str = "Zuora Help Agent"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API Keys
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "chroma"  # Options: chroma, pinecone, weaviate
    CHROMA_PERSIST_DIRECTORY: str = "../data/chroma_db"
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = "zuora-help"

    # Embedding Model Settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI embedding model
    EMBEDDING_DIMENSION: int = 1536

    # LLM Settings
    LLM_PROVIDER: str = "openai"  # Options: openai, anthropic
    LLM_MODEL: str = "gpt-4-turbo-preview"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000

    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".md", ".docx"]

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Helper functions
def get_data_dir() -> Path:
    """Get the data directory path"""
    return Path(__file__).parent.parent / "data"


def get_chroma_db_path() -> Path:
    """Get the Chroma DB persist directory path"""
    return Path(__file__).parent.parent / settings.CHROMA_PERSIST_DIRECTORY


def validate_api_keys() -> bool:
    """Validate that required API keys are present"""
    if settings.LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        return False
    if settings.LLM_PROVIDER == "anthropic" and not settings.ANTHROPIC_API_KEY:
        return False
    return True
