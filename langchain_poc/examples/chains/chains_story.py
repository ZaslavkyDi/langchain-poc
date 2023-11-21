from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

from langchain_poc.examples.base import BaseExample


class SequentialChainExample(BaseExample):
    story_template = """
        As a children's book writer, please come up with a simple and short (90 words)
        lullaby based on the location {location}
        and the main character {name}

        STORY:
    """

    translation_template = """
        Translate the {story} into {language}. Make sure the language is simple and fun.
        
        TRANSLATION:
    """

    def run_example(self) -> None:
        self.chat_model.temperature = 0.5

        story_chain = self._make_story_chain()
        translated_story = self._make_translate_story_chain()

        overall_chain = SequentialChain(
            chains=[story_chain, translated_story],
            input_variables=["location", "name", "language"],
            output_variables=["story", "translated"],
        )
        response = overall_chain(
            inputs={
                "location": "Ukraine",
                "name": "Ira",
                "language": "Ukrainian",
            }
        )
        print(f"Overall Chain response: {response}")

        print(f"Story Chain Output: {response['story']}")
        print(f"Translation Chain Output: {response['translated']}")

    def _make_story_chain(self) -> LLMChain:
        story_prompt_template = PromptTemplate(
            input_variables=["location", "name"],
            template=self.story_template,
        )
        return LLMChain(
            llm=self.chat_model,
            prompt=story_prompt_template,
            output_key="story",  # uses in SequentialChain(..., output_variables=["story"])
            verbose=True,
        )

    def _make_translate_story_chain(self) -> LLMChain:
        translation_prompt_template = PromptTemplate(
            input_variables=["story", "language"], template=self.translation_template
        )
        return LLMChain(
            llm=self.chat_model,
            prompt=translation_prompt_template,
            output_key="translated",
            verbose=True,
        )
