from langchain.chat_models import ChatOpenAI


class BaseExample:
    def __init__(self, llm_model: str, chat_model: ChatOpenAI) -> None:
        self.llm_model = llm_model
        self.chat_model = chat_model
