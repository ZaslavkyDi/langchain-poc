from functools import lru_cache

from langchain_poc.config.settings import DatabaseSettings, OpenAISettings


@lru_cache
def get_openai_settings() -> OpenAISettings:
    return OpenAISettings()


@lru_cache
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()
