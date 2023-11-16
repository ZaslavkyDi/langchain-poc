from functools import lru_cache

from langchain_poc.config.settings import OpenAISettings


@lru_cache
def get_openai_settings() -> OpenAISettings:
    return OpenAISettings()
