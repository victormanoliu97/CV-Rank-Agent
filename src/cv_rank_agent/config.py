"""Pydantic-settings configuration loaded from .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Path to the project root (where .env lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings — values are read from .env automatically."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore env vars not declared here
    )

    # Default values for fallback if not set in .env (can be overridden by env vars)
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.0

    # Scoring strategy
    llm_only_threshold: int = 5   # <= this many jobs → LLM scores all (Option A)
    llm_top_n: int = 10           # > threshold → LLM deep-scores top N (Option B)

    # Limits
    max_jobs: int = 50


# Singleton instance — import this wherever you need settings
settings = Settings()
