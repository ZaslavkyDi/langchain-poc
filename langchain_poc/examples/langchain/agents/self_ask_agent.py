from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.utilities.serpapi import SerpAPIWrapper

from langchain_poc.config import get_openai_settings
from langchain_poc.examples.langchain.base import BaseExample


class SelfAskWithSearchAgentExample(BaseExample):
    """Uses 'google-search-results' library for searching."""

    def run_example(self) -> None:
        search_api = SerpAPIWrapper(serpapi_api_key=get_openai_settings().serper_api_key)
        llm = ChatOpenAI(
            temperature=0, api_key=get_openai_settings().api_key, model_name="gpt-3.5-turbo-0301"
        )
        tools = [
            Tool(
                name="Intermediate Answer",
                func=search_api.run,
                description="useful for when you need to ask with search",
            )
        ]

        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.SELF_ASK_WITH_SEARCH,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=4,
        )

        r = agent_executor.invoke({"input": "How to make pizza?"})
        print(r)
