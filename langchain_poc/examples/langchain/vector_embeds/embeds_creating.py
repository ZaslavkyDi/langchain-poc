from typing import ClassVar

from langchain.embeddings import OpenAIEmbeddings

from langchain_poc.config import get_openai_settings
from langchain_poc.examples.langchain.base import BaseExample


class EmbedsCreateExample(BaseExample):
    _texts: ClassVar[dict[str, str]] = {
        "1": "Math is a greate subject to study.",
        "2": "Dogs are friendly when they are happy and well fed.",
        "3": "Physics is not my favorite subject.",
    }

    def run_example(self) -> None:
        embedding = OpenAIEmbeddings(openai_api_key=get_openai_settings().api_key)

        embed1: list[float] = embedding.embed_query(text=self._texts["1"])
        embed2: list[float] = embedding.embed_query(text=self._texts["2"])
        embed3: list[float] = embedding.embed_query(text=self._texts["3"])

        print(f"{embed1 = }")
        print(f"{len(embed1) = }")

        import numpy as np

        print(f"Similarity 1 to 2 {np.dot(embed1, embed2) * 100}")
        print(f"Similarity 1 to 3 {np.dot(embed1, embed3) * 100}")
