from langchain.chat_models import ChatOpenAI

from langchain_poc.config import get_openai_settings
from langchain_poc.examples.chains.chains import ChainsExample
from langchain_poc.examples.chains.chains_story import ChainsStoryExample
from langchain_poc.examples.intro_chat_prompts import IntroChatPrompts
from langchain_poc.examples.lang_parser import LangParser
from langchain_poc.examples.memory import BufferMemoryExample

llm_model: str = "gpt-3.5-turbo"
chat_model = ChatOpenAI(temperature=0, api_key=get_openai_settings().api_key, model_name=llm_model)


def _get_intro_chat_prompts() -> IntroChatPrompts:
    return IntroChatPrompts(chat_model=chat_model)


def _get_lang_parser() -> LangParser:
    return LangParser(chat_model=chat_model)


def _get_memory() -> BufferMemoryExample:
    return BufferMemoryExample(chat_model=chat_model)


def _get_chains() -> ChainsExample:
    return ChainsExample(chat_model=chat_model)


def _get_chains_story() -> ChainsStoryExample:
    return ChainsStoryExample(chat_model=chat_model)


def main() -> None:
    example = _get_chains_story()
    example.run_example()


if __name__ == "__main__":
    main()
