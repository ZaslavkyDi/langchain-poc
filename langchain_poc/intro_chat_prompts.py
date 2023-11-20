from typing import Any

import openai
from langchain.chat_models import ChatOpenAI


class IntroChatPrompts:
    def __init__(self, llm_model: str, chat_model: ChatOpenAI) -> None:
        self._llm_model = llm_model
        self._chat_model = chat_model

    def get_completion(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        response = openai.ChatCompletion.create(
            model=self._llm_model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]

    def run_example(self) -> None:
        # Translate text, review
        customer_review = """
        Your product is terrable! I don't know how you were able to get this to the market.
        I don't want this! Actually, no one should want this.
        Seriously! Give me my money!
        """
        tone = "Proper English in a nice warm, respectful tone"
        language = "France"
        prompt = f"""
            Rewrite the following {customer_review} in {tone}, and then please translate the new review 
            message into {language}.
        """
        rewrite_content: str = self.get_completion(prompt)
        print(rewrite_content)
