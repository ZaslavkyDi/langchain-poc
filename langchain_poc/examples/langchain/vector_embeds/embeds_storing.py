from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from langchain_poc.config import get_openai_settings
from langchain_poc.examples.langchain.base import BaseExample


class EmbedsStoringExample(BaseExample):
    _pdf_file_path = "../resources/react-paper.pdf"
    _persist_directory = "../data/db/chromadb"

    def run_example(self) -> None:
        documents = self._load_file_spits(file_path=self._pdf_file_path)
        openai_embeddings = OpenAIEmbeddings(openai_api_key=get_openai_settings().api_key)

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=openai_embeddings,
            persist_directory=self._persist_directory,
        )
        answers = self._execute_query(
            query="What do they said about ReAct prompting method?",
            vector_db=vectorstore,
        )
        vectorstore.persist()  # save answers
        print(f"{len(answers) = }")
        print(f"{answers[0].page_content = }")

    @staticmethod
    def _load_file_spits(file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path=file_path)
        docs: list[Document] = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
        )
        splits: list[Document] = text_splitter.split_documents(documents=docs)
        return splits

    @staticmethod
    def _execute_query(query: str, vector_db: Chroma) -> list[Document]:
        number_of_docs_to_retrieve = 3
        return vector_db.similarity_search(
            query=query,
            k=number_of_docs_to_retrieve,
        )
