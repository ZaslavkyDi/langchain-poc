from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_poc.examples.langchain.base import BaseExample


class RecursiveTextSplitterExample(BaseExample):
    _resource_path_template = "../resources/{file_name}"

    def run_example(self) -> None:
        file_path = self._resource_path_template.format(file_name="text_splitter_example.txt")

        with open(file_path) as file:
            list_content: list[str] = file.readlines()

        with open(file_path) as file:
            str_content: str = file.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # number of characters in a chunk
            chunk_overlap=100,  # overlapping with 100 characters for each chunk
            length_function=len,
            add_start_index=True,
        )
        # almost the same result
        list_content_chunks: list[Document] = text_splitter.create_documents(
            texts=list_content
        )  # len == 14
        str_content_chunks: list[Document] = text_splitter.create_documents(
            texts=[str_content]
        )  # len == 13 - getting a warning about chunk size

        print(
            f"The same first chunk: {str_content_chunks[1].page_content == list_content_chunks[1].page_content}"
        )
        print(f"Chunk content len: {len(list_content_chunks[1].page_content)}")
        print(list_content_chunks[1])
