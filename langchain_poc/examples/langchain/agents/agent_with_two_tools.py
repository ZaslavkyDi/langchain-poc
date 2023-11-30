from langchain.agents import Agent, initialize_agent, AgentType, AgentExecutor, load_tools
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from langchain_poc.examples.langchain.base import BaseExample


class AgentWithTwoToolsExample(BaseExample):
    _math_query = "What is 3.2^2.1?"
    _generic_query = "What is the capital of France?"

    def run_example(self) -> None:
        # the LLM model returns incorrect result of 3.1^2.1 = 9.113. The correct one is 10.76
        incorrect_response = self.chat_model.predict(text=self._math_query)
        print(incorrect_response)

        default_tool = self._create_default_tool()

        tools = load_tools(
            tool_names=["llm-math"],  # the name you can find in source code
            llm=self.chat_model,
        )
        tools.append(default_tool)  # adding new tool to tools list

        agent: AgentExecutor = initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=tools,
            llm=self.chat_model,
            verbose=True,
            max_iterations=3,  # reduce number of periodical call
        )
        math_response = agent(self._math_query)  # gives a correct answer 11.50
        print(f"{math_response = }")

        generic_response = agent(self._generic_query)
        print(f"{generic_response = }")

        # show agent template
        print(f"{agent.agent.llm_chain.prompt.template = }")

    def _create_default_tool(self) -> Tool:
        prompt = PromptTemplate(
            input_variables=["query"],
            template="{query}",
        )
        chain = LLMChain(
            llm=self.chat_model,
            prompt=prompt,
        )
        return Tool(
            name="General Purpose",
            func=chain.run,
            description="Useful for answering general purpose questions and logic."
        )
