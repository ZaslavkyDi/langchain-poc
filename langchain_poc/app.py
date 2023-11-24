from langchain.chat_models import ChatOpenAI

from langchain_poc.config import get_openai_settings
from langchain_poc.examples.langchain.chains.chain import ChainsExample
from langchain_poc.examples.langchain.chains.router_chain import RouterChainExample
from langchain_poc.examples.langchain.chains.sequential_chain import SequentialChainExample
from langchain_poc.examples.langchain.intro_chat_prompts import IntroChatPrompts
from langchain_poc.examples.langchain.lang_parser import LangParser
from langchain_poc.examples.langchain.memory import BufferMemoryExample
from langchain_poc.examples.langchain.pdf_loader import SimplePdfLoaderExample

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


def _get_sequential_chain() -> SequentialChainExample:
    return SequentialChainExample(chat_model=chat_model)


def _get_router_chain() -> RouterChainExample:
    return RouterChainExample(chat_model=chat_model)


def _get_simple_pdf_loader_example() -> SimplePdfLoaderExample:
    return SimplePdfLoaderExample(chat_model=chat_model)


def main() -> None:
    example = _get_simple_pdf_loader_example()
    example.run_example()


if __name__ == "__main__":
    main()
