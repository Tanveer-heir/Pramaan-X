"""Configuration settings with fallback for environments without pydantic-settings."""

import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    HAS_PYDANTIC_SETTINGS = False

if HAS_PYDANTIC_SETTINGS:
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )
        ENVIRONMENT: str = "development"
        LOG_LEVEL: str = "INFO"
        API_PORT: int = 8000
        API_HOST: str = "0.0.0.0"

        # Part 1: Source Attribution
        GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
        YANDEX_API_KEY: Optional[str] = None
        OPENAI_API_KEY: Optional[str] = None
        GEMINI_API_KEY: Optional[str] = None
        REDDIT_CLIENT_ID: Optional[str] = None
        REDDIT_CLIENT_SECRET: Optional[str] = None
        REDDIT_USER_AGENT: str = "cph-attribution-agent:v1.0"
        YOUTUBE_API_KEY: Optional[str] = None
        TWITTER_BEARER_TOKEN: Optional[str] = None
        APIFY_API_TOKEN: Optional[str] = None

        # Part 2: Fingerprinting
        PDQ_HAMMING_THRESHOLD: int = 30
        PDQ_DATASET_PATH: str = "data/mock_instances.json"
        PRNU_REFERENCE_DIR: str = "data/prnu_reference/"

        # Part 3: Metadata & Provenance
        C2PATINY_TOOL_PATH: str = "c2patool"
        EXIFTOOL_PATH: str = "exiftool"
        CHAIN_OF_CUSTODY_SECRET: str = "cph-default-ledger-secret"

        # Security
        JWT_SECRET_KEY: str = "cph-super-secure-jwt-secret-key-2026"
        JWT_ALGORITHM: str = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    settings = Settings()
else:
    class FallbackSettings:
        def __init__(self):
            self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
            self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
            self.API_PORT = int(os.getenv("API_PORT", "8000"))
            self.API_HOST = os.getenv("API_HOST", "0.0.0.0")

            self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            self.YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            self.REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
            self.REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
            self.REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "cph-attribution-agent:v1.0")
            self.YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
            self.TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
            self.APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

            self.PDQ_HAMMING_THRESHOLD = int(os.getenv("PDQ_HAMMING_THRESHOLD", "30"))
            self.PDQ_DATASET_PATH = os.getenv("PDQ_DATASET_PATH", "data/mock_instances.json")
            self.PRNU_REFERENCE_DIR = os.getenv("PRNU_REFERENCE_DIR", "data/prnu_reference/")

            self.C2PATINY_TOOL_PATH = os.getenv("C2PATINY_TOOL_PATH", "c2patool")
            self.EXIFTOOL_PATH = os.getenv("EXIFTOOL_PATH", "exiftool")
            self.CHAIN_OF_CUSTODY_SECRET = os.getenv("CHAIN_OF_CUSTODY_SECRET", "cph-default-ledger-secret")

            self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "cph-super-secure-jwt-secret-key-2026")
            self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
            self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    settings = FallbackSettings()
