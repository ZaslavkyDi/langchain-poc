from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.chains import LLMChain
from langchain.docstore import Wikipedia
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

from langchain_poc.examples.langchain.base import BaseExample


class DocstoreAgentExample(BaseExample):
    """Search and Lookup data from Wikipedia website"""

    _query = "What was Bach's lat piece he wrote?"

    def run_example(self) -> None:
        docstore_explorer = DocstoreExplorer(docstore=Wikipedia())
        tools = [
            Tool(
                name="Search",
                func=docstore_explorer.search,
                description="Search Wikipedia",
            ),
            Tool(
                name="Lookup",
                func=docstore_explorer.lookup,
                description="Lookup a term in Wikipedia",
            ),
        ]
        llm = OpenAI(temperature=0, model_name="text-davinci-002")  # that LLM works over ChatOpenAI
        docstore_agent = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True, max_iterations=3)
        result = docstore_agent(self._query)
        print(result)
