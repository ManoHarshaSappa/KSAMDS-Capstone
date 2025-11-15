"""
Configuration management for KSAMDS API
Handles database connection settings and application configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    Create a .env file in the api directory with these variables
    """

    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "ksamds"
    DB_USER: str  # Required - must be set via environment variable
    DB_PASSWORD: str  # Required - must be set via environment variable
    DB_SCHEMA: str = "ksamds"

    # Application Configuration
    APP_NAME: str = "KSAMDS API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # CORS Configuration
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",  # Allow file:// protocol for development
    ]

    # API Configuration
    API_PREFIX: str = "/api"

    # Pagination
    DEFAULT_LIMIT: int = 50
    MAX_LIMIT: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

    @property
    def database_url(self) -> str:
        """
        Construct PostgreSQL connection URL
        Format: postgresql://user:password@host:port/database
        """
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def async_database_url(self) -> str:
        """
        Construct async PostgreSQL connection URL for asyncpg
        Format: postgresql+asyncpg://user:password@host:port/database
        """
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Create cached settings instance
    Using lru_cache ensures settings are loaded only once
    """
    return Settings()


# Convenience instance for importing
settings = get_settings()
