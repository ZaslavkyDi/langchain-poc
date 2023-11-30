from langchain.agents import Agent, initialize_agent, AgentType, AgentExecutor, load_tools
from langchain.chains import LLMMathChain
from langchain.tools import Tool

from langchain_poc.examples.langchain.base import BaseExample


class SimpleAgentExample(BaseExample):
    _math_query = "What is 3.2^2.1?"

    def run_example(self) -> None:
        # the LLM model returns incorrect result of 3.1^2.1 = 9.113. The correct one is 10.76
        incorrect_response = self.chat_model.predict(text=self._math_query)
        print(incorrect_response)

        custom_math_tool = self._create_custom_math_tool()
        # tools = [custom_math_tool] custom tools
        tools = load_tools(
            tool_names=["llm-math"],  # the name you can find in source code
            llm=self.chat_model,
        )
        print(tools)

        agent: AgentExecutor = initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=self.chat_model,
            verbose=True,
            max_iterations=3,  # reduce
        )
        response = agent(self._math_query)  # gives a correct answer 11.50
        print(response)

    def _create_custom_math_tool(self) -> Tool:
        # Using Agent with math tool
        math_chain = LLMMathChain.from_llm(llm=self.chat_model)
        return Tool(
            name="Calculator",
            func=math_chain.run,
            description="Useful, when you need to answer math related questions",
        )
