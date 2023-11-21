from langchain.chat_models import ChatOpenAI

from langchain_poc.config import get_openai_settings
from langchain_poc.examples.intro_chat_prompts import IntroChatPrompts
from langchain_poc.examples.lang_parser import LangParser

llm_model: str = "gpt-3.5-turbo"
chat_model = ChatOpenAI(temperature=0, api_key=get_openai_settings().api_key)


def _get_intro_chat_prompts() -> IntroChatPrompts:
    return IntroChatPrompts(llm_model=llm_model, chat_model=chat_model)


def _get_lang_parser() -> LangParser:
    return LangParser(llm_model=llm_model, chat_model=chat_model)


def main() -> None:
    example = _get_lang_parser()
    example.run_pydantic_parser_example()


if __name__ == "__main__":
    main()
