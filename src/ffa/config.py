"""Settings loaded from environment variables (see .env.example)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://ffa:change-me@localhost:5432/ffa"
    warehouse_dir: str = "./data/warehouse"
    current_season: int = 2026

    sleeper_username: str = ""
    sleeper_league_id: str = ""

    yahoo_client_id: str = ""
    yahoo_client_secret: str = ""
    yahoo_league_key: str = ""


settings = Settings()
