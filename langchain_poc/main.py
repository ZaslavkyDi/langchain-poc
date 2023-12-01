from langchain.chat_models import ChatOpenAI

from langchain_poc.config import get_openai_settings
from langchain_poc.examples.langchain.agents.agent_with_two_tools import AgentWithTwoToolsExample
from langchain_poc.examples.langchain.agents.conversational_agent import ConversationalAgentExample
from langchain_poc.examples.langchain.agents.docstore_agent import DocstoreAgentExample
from langchain_poc.examples.langchain.agents.self_ask_agent import SelfAskWithSearchAgentExample
from langchain_poc.examples.langchain.agents.simple_agent import SimpleAgentExample
from langchain_poc.examples.langchain.chains.chain import ChainsExample
from langchain_poc.examples.langchain.chains.router_chain import RouterChainExample
from langchain_poc.examples.langchain.chains.sequential_chain import SequentialChainExample
from langchain_poc.examples.langchain.intro_chat_prompts import IntroChatPrompts
from langchain_poc.examples.langchain.lang_parser import LangParser
from langchain_poc.examples.langchain.memory import BufferMemoryExample
from langchain_poc.examples.langchain.pdf_loader import SimplePdfLoaderExample
from langchain_poc.examples.langchain.splitters.text_splitter import TextSplitterExample
from langchain_poc.examples.langchain.splitters.text_splitter_recursive import (
    RecursiveTextSplitterExample,
)
from langchain_poc.examples.langchain.vector_embeds.embeds_creating import (
    EmbedsCreateExample,
)
from langchain_poc.examples.langchain.vector_embeds.embeds_retrieval import EmbedsRetrievalExample
from langchain_poc.examples.langchain.vector_embeds.embeds_storing import (
    EmbedsStoringExample,
)

LLM_MODEL_GENERATION: str = "gpt-3.5-turbo"
chat_model = ChatOpenAI(
    temperature=0, api_key=get_openai_settings().api_key, model_name=LLM_MODEL_GENERATION
)


# Prompts & Parsers & Memory
def _get_intro_chat_prompts() -> IntroChatPrompts:
    return IntroChatPrompts(chat_model=chat_model)


def _get_lang_parser() -> LangParser:
    return LangParser(chat_model=chat_model)


def _get_memory() -> BufferMemoryExample:
    return BufferMemoryExample(chat_model=chat_model)


# Chains
def _get_chains() -> ChainsExample:
    return ChainsExample(chat_model=chat_model)


def _get_sequential_chain() -> SequentialChainExample:
    return SequentialChainExample(chat_model=chat_model)


def _get_router_chain() -> RouterChainExample:
    return RouterChainExample(chat_model=chat_model)


# PDF loader & Text Splitter
def _get_simple_pdf_loader_example() -> SimplePdfLoaderExample:
    return SimplePdfLoaderExample(chat_model=chat_model)


def _get_text_splitter_example() -> TextSplitterExample:
    return TextSplitterExample(chat_model=chat_model)


def _get_text_splitter_recursive_example() -> RecursiveTextSplitterExample:
    # the best text splitter to use
    return RecursiveTextSplitterExample(chat_model=chat_model)


# VectorStore
def _get_vector_embeds_creating_example() -> EmbedsCreateExample:
    return EmbedsCreateExample(chat_model=chat_model)


def _get_vector_embeds_storing_example() -> EmbedsStoringExample:
    return EmbedsStoringExample(chat_model=chat_model)


def _get_vector_embeds_retrieval_example() -> EmbedsRetrievalExample:
    return EmbedsRetrievalExample(chat_model=chat_model)


# Agents & Tools
def _get_simple_agent_example() -> SimpleAgentExample:
    return SimpleAgentExample(chat_model=chat_model)


def _get_agent_with_2_tools_example() -> AgentWithTwoToolsExample:
    return AgentWithTwoToolsExample(chat_model=chat_model)


def _get_conversational_agent_example() -> ConversationalAgentExample:
    # TODO: return an error message from the first query - need to investigate
    return ConversationalAgentExample(chat_model=chat_model)


def _get_docstore_agent_example() -> DocstoreAgentExample:
    return DocstoreAgentExample(chat_model=chat_model)


def _get_self_ask_agent_example() -> SelfAskWithSearchAgentExample:
    return SelfAskWithSearchAgentExample(chat_model=chat_model)


def main() -> None:
    example = _get_self_ask_agent_example()
    example.run_example()


if __name__ == "__main__":
    main()
