from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="openai_")

    api_key: str = "<your_api_key>"
    serper_api_key: str = "<your_serper_api_key>"


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="postgres_")

    host: str = "localhost"
    port: int = 5432
    username: str = "postgres"
    password: str = "postgres"
    db: str = "vectordb"

    @property
    def default_postgres_driver(self) -> str:
        return "psycopg2"

    @property
    def postgres_psycopg2_url(self) -> str:
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.db}"
