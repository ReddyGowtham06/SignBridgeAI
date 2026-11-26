"""
SignBridge AI - Application Configuration
Supports environment-based configuration via pydantic-settings.
"""
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "SIGNBRIDGE AI Backend"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api"
    ENVIRONMENT: str = "development"

    # Database Settings
    # Default is PostgreSQL. In environments where PostgreSQL is not yet configured,
    # fallback to SQLite is explicitly controlled and logged.
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/signbridge_db"
    DATABASE_CONNECT_TIMEOUT: int = 2
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # JWT Authentication
    JWT_SECRET_KEY: str = "signbridge_insecure_default_secret_key_change_in_production_2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "https://sign-bridge-ai-ecru.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # AI Model Path (M1 Champion)
    M1_MODEL_PATH: str = "frontend/public/models/sign_classifier.json"

    # Rate Limiting
    RATE_LIMIT_AUTH_PER_MINUTE: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
