from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from langchain_poc.examples.langchain.base import BaseExample


class ConversationalAgentExample(BaseExample):
    _1st_query = """
        How old is a man who born in 1998 until now (now is 2023 year)?
    """
    _2nd_query = """
        How old the man from the previous question will be in 2034 year?
    """

    def run_example(self) -> None:
        general_tool = self._make_general_question_tool()
        tools = load_tools(
            tool_names=["llm-math"],
            llm=self.chat_model,
        )
        tools.append(general_tool)

        conversational_agent = initialize_agent(
            tools=tools,
            llm=self.chat_model,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3,
            memory=ConversationBufferMemory(memory_key="chat_history"),
        )
        conversational_agent.run(self._1st_query)
        conversational_agent.run(self._2nd_query)

    def _make_general_question_tool(self) -> Tool:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="{query}",
        )
        general_purpose_chain = LLMChain(llm=self.chat_model, prompt=prompt)
        return Tool(
            name="General Purpose",
            func=general_purpose_chain.run,
            description="Useful when need to answer general purpose questions and logic.",
        )
