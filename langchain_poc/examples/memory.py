
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain_poc.examples.base import BaseExample


class BufferMemoryExample(BaseExample):
    def run_example(self) -> None:
        def show_that_llm_is_stateless() -> None:
            # check if ChatOpenAI remember the name 'Bob' - nope
            print(self.chat_model.predict("My name is Bob. What's yours?"))
            print(self.chat_model.predict("Greate! What is my name?"))  # we have memory issue

        # show_that_llm_is_stateless()

        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=self.chat_model,
            memory=memory,
            verbose=True,
        )
        conversation.predict(input="Hello there, I am Tom.")
        conversation.predict(input="Why is the sky blue?")
        conversation.predict(input="What is my name?")

        print(memory.load_memory_variables(inputs={}))
