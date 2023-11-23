from operator import itemgetter
from typing import Literal

from langchain.chains import LLMChain, LLMRouterChain, MultiRouteChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from pydantic.v1 import BaseModel, Field

from langchain_poc.examples.langchain.base import BaseExample


class RouterChainExample(BaseExample):
    _biology_template = """You are a very smart biology professor.
        You are great at answering questions about biology in a concise and easy to understand.
        When you don't know the answer to a question you admit that you don't know.
        
        Here is a question: {input}
    """

    _math_template = """You are a very god mathematician. 
        You are great at answering math questions.
        You are so good because you are able to break down hard problems into their component parts,
        answer the component parts, and then put them together to answer the broader questions.
        
        Here is a question: {input}
    """

    _travel_agent_template = """"You are a very good travel agent with a large amount
        of knowledge when it comes to getting people the best deals and recommendations for travel, vacations, 
        flights and world's best destinations for vacation.
        You are great at answering travel, vacation, flights, transportation, tourist guidelines.
        You are so good because you are able to break down hard problems into their component parts,
        answer the component parts, and then put them together to answer the broader questions.
        
        Here is a question: {input}
    """

    _default_template = """Answer this questions: {input}"""

    def run_legacy_example(self) -> None:
        prompt_topic_infos = [
            {
                "topic": "biology",
                "description": "Good for answering biology related questions",
                "prompt_template": self._biology_template,
            },
            {
                "topic": "math",
                "description": "Good for answering math related questions",
                "prompt_template": self._math_template,
            },
            {
                "topic": "travel_agent",
                "description": "Good for answering travel, tourism and vacation questions",
                "prompt_template": self._travel_agent_template,
            },
        ]

        destination_topic_chains: dict[str, LLMChain] = {}
        for topic_info in prompt_topic_infos:
            topic = topic_info["topic"]
            prompt = ChatPromptTemplate.from_template(template=topic_info["prompt_template"])
            chain = LLMChain(llm=self.chat_model, prompt=prompt)
            destination_topic_chains[topic] = chain

        default_topic_chain = self._make_default_topic_chain()

        # setup Router Chain
        destinations: list[str] = [f"{p['topic']}: {p['description']}" for p in prompt_topic_infos]
        destinations_str = "\n".join(destinations)

        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
        router_prompt_template = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )

        router_chain = LLMRouterChain.from_llm(
            llm=self.chat_model,
            prompt=router_prompt_template,
        )
        main_chain = MultiRouteChain(
            router_chain=router_chain,
            destination_chains=destination_topic_chains,
            default_chain=default_topic_chain,
            verbose=True,
        )

        response = main_chain("Give me a list of places to visit in Ukraine.")
        print(response)

    def run_example(self) -> None:
        class TopicClassifier(BaseModel):
            """ "Classify the topic of the user question"""

            topic: Literal["math", "physics", "general"] = Field(
                description="The topic of the user question. One of 'math', 'physics' or 'general'."
            )

        biology_prompt = PromptTemplate.from_template(self._biology_template)
        math_prompt = PromptTemplate.from_template(self._math_template)
        travel_agent_prompt = PromptTemplate.from_template(self._travel_agent_template)
        default_prompt = PromptTemplate.from_template(self._default_template)

        prompt_branch = RunnableBranch(
            (lambda x: x["topic"] == "biology", biology_prompt),
            (lambda x: x["topic"] == "math", math_prompt),
            (lambda x: x["topic"] == "travel_agent", travel_agent_prompt),
            default_prompt,
        )
        classifier_function = convert_pydantic_to_openai_function(TopicClassifier)

        llm = self.chat_model.bind(
            functions=[classifier_function], function_call={"name": "TopicClassifier"}
        )
        parser = PydanticAttrOutputFunctionsParser(
            pydantic_schema=TopicClassifier, attr_name="topic"
        )
        classifier_chain = llm | parser

        final_chain = (
            RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain)
            | prompt_branch
            | self.chat_model
            | StrOutputParser()
        )
        response = final_chain.invoke(
            {
                "input": "What I can see in Ukraine? Also, in the last output line give me the topic which you selected."
            }
        )
        print(response)

    def _make_default_topic_chain(self) -> LLMChain:
        prompt_template = ChatPromptTemplate.from_template(self._default_template)
        return LLMChain(llm=self.chat_model, prompt=prompt_template)
