from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

from langchain_poc.examples.langchain.base import BaseExample


class SimplePdfLoaderExample(BaseExample):
    _resource_path_template = "../resources/{file_name}"

    def run_example(self) -> None:
        file_path = self._resource_path_template.format(file_name="react-paper.pdf")

        pdf_loader = PyPDFLoader(file_path=file_path)
        pages: list[Document] = pdf_loader.load()

        print(pages[0])
        print()
        print(pages[0].page_content[0:700])  # get text from 0 to 699 character
