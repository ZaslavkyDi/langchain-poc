from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_poc.examples.langchain.base import BaseExample


class ChainsExample(BaseExample):
    def run_example(self) -> None:
        prompt_template = PromptTemplate(
            input_variables=["language"],
            template="How do you say 'Good morning' in {language}",
        )
        llm_chain = LLMChain(prompt=prompt_template, llm=self.chat_model)

        response = llm_chain.run(language="Ukrainian")
        print(response)
