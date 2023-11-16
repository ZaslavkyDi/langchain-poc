from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='openai_')

    api_key: str = "<your_api_key>"
