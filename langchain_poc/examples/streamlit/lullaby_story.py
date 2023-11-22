import streamlit as st
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain_poc.config import get_openai_settings
from langchain_poc.examples.langchain.base import BaseExample


class LullabyStoryCreatorExample(BaseExample):
    _story_template = """
        As a children's book writer, please come up with a simple and short (90 words)
        lullaby based on the location {location}
        and the main character {person_name}
    """

    _translation_template = """
        Translate the {story} into {language}. Make sure the language is simple and fun.
    """

    def maka_up_lullaby_story(
        self, location: str, person_name: str, language: str
    ) -> tuple[str, str]:
        story_chain = self._make_story_chain()
        translator_story_chain = self._make_story_translator_chain()

        main_chain = SequentialChain(
            chains=[story_chain, translator_story_chain],
            input_variables=["location", "person_name", "language"],
            output_variables=["story", "translated"],
            verbose=True,
        )

        response = main_chain(
            inputs={
                "location": location,
                "person_name": person_name,
                "language": language,
            }
        )
        origin_story: str = response["story"]
        translated_to_language_story: str = response["translated"]

        return origin_story, translated_to_language_story

    def _make_story_chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=self._story_template,
            input_variables=["location", "person_name"],
        )
        return LLMChain(
            llm=self.chat_model,
            prompt=prompt,
            output_key="story",
            verbose=True,
        )

    def _make_story_translator_chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=self._translation_template,
            input_variables=["story", "language"],
        )
        return LLMChain(
            llm=self.chat_model,
            prompt=prompt,
            output_key="translated",
            verbose=True,
        )


class StreamlitLullabyStoryPage:
    def __init__(self, lullaby_maker: LullabyStoryCreatorExample) -> None:
        self._lullaby_maker = lullaby_maker
        self._location_input: str | None = None
        self._character_input: str | None = None
        self._language_input: str | None = None
        self._submit_button: bool = False

    def setup_page(self) -> None:
        st.set_page_config(
            page_title="Generate Children's Lullaby",
            layout="centered",
        )
        st.title("Let AI Write and Translate a Lullaby for You 🙌📕")
        st.header("Get Started...")

        self._location_input = st.text_input(label="Where is story set?")
        self._character_input = st.text_input(label="What's the main character's name?")
        self._language_input = st.text_input(label="Translate the story into...")

        self._submit_button = st.button("Submit")

    def make_lullaby_story(self) -> tuple[str, str]:
        has_all_inputs: bool = all(
            [self._language_input, self._character_input, self._location_input]
        )

        if self._submit_button and not has_all_inputs:
            st.warning("All input fields must be populated!")
            return

        if self._submit_button:
            with st.spinner("Generating lullaby..."):
                yield self._lullaby_maker.maka_up_lullaby_story(
                    location=self._location_input,
                    person_name=self._character_input,
                    language=self._language_input,
                )

            st.success("Lullaby successfully generated!")

    def show_story(self, origin_story: str, translated_story: str) -> None:
        with st.expander("English Version"):
            st.write(origin_story)

        with st.expander(f"Translated into {self._language_input}"):
            st.write(translated_story)


def setup_and_run_streamlit_app() -> None:
    llm_model: str = "gpt-3.5-turbo"
    chat_model = ChatOpenAI(
        temperature=0, api_key=get_openai_settings().api_key, model_name=llm_model
    )

    app = StreamlitLullabyStoryPage(lullaby_maker=LullabyStoryCreatorExample(chat_model=chat_model))
    app.setup_page()
    for origin_story, translated_story in app.make_lullaby_story():
        app.show_story(origin_story, translated_story)


if __name__ == "__main__":
    setup_and_run_streamlit_app()
