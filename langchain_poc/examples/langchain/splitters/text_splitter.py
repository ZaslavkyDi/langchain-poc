from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from langchain_poc.examples.langchain.base import BaseExample


class TextSplitterExample(BaseExample):
    _resource_path_template = "../resources/{file_name}"

    def run_example(self) -> None:
        file_path = self._resource_path_template.format(file_name="text_splitter_example.txt")

        with open(file_path) as file:
            list_content: list[str] = file.readlines()

        with open(file_path) as file:
            str_content: str = file.read()

        text_splitter = CharacterTextSplitter(
            chunk_size=100,  # number of characters in a chunk
            chunk_overlap=50,  # overlapping with 50 characters for each chunk
            length_function=len,
        )
        # almost the same result
        list_content_chunks: list[Document] = text_splitter.create_documents(
            texts=list_content
        )  # len == 14
        str_content_chunks: list[Document] = text_splitter.create_documents(
            texts=[str_content]
        )  # len == 13 - getting a warning about chunk size

        print(
            f"The same first chunk: {str_content_chunks[0].page_content == list_content_chunks[0].page_content}"
        )
        print(list_content_chunks[0])
