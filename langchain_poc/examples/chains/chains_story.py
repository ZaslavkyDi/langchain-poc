from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_poc.examples.base import BaseExample


class ChainsStoryExample(BaseExample):
    story_template = """
        As a children's book writer, please come up with a simple and short (90 words)
        lullaby based on the location {location}
        and the main character {name}

        STORY:
    """

    def run_example(self) -> None:
        self.chat_model.temperature = 0.5

        prompt_template = PromptTemplate(
            input_variables=["location", "name"],
            template=self.story_template,
        )
        chain_story = LLMChain(llm=self.chat_model, prompt=prompt_template, verbose=True)

        story: dict[str, str] = chain_story(
                inputs={
                    "location": "Ukraine",
                    "name": "Ira",
                }
            )

        print(story["text"])
