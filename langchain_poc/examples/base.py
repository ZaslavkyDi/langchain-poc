from langchain.chat_models import ChatOpenAI


class BaseExample:
    def __init__(self, chat_model: ChatOpenAI) -> None:
        self.chat_model = chat_model
