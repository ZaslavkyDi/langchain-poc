from typing import Any

import openai
from langchain.prompts import ChatPromptTemplate

from langchain_poc.examples.base import BaseExample


class IntroChatPrompts(BaseExample):
    customer_review = """
    Your product is terrable! I don't know how you were able to get this to the market.
    I don't want this! Actually, no one should want this.
    Seriously! Give me my money!
    """

    def get_completion(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]

    def run_example(self) -> None:
        # Translate text, review
        tone = "Proper English in a nice warm, respectful tone"
        language = "France"
        prompt = f"""
            Rewrite the following {self.customer_review} in {tone}, and then please translate the new review 
            message into {language}.
        """
        rewrite_content: str = self.get_completion(prompt)
        print(rewrite_content)

    def run_example_with_prompt_template(self) -> None:
        template_string = """Translate the following text {customer_review}
         into italiano in a polite tone.
         And company name is {company_name}
         """

        prompt_template = ChatPromptTemplate.from_template(template=template_string)
        translation_message = prompt_template.format_messages(
            customer_review=self.customer_review, company_name="Google"
        )

        response = self.chat_model(translation_message)
        print(response.content)
