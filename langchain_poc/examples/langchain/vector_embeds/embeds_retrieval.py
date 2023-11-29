import shutil
from typing import Any

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from langchain_poc.config import get_openai_settings
from langchain_poc.examples.langchain.base import BaseExample


class EmbedsRetrievalExample(BaseExample):
    _pdf_file_path = "../resources/react-paper.pdf"
    _persist_directory = "../data/db/chromadb"

    def __init__(self, chat_model: ChatOpenAI):
        super().__init__(chat_model)
        self.openai_embeddings = OpenAIEmbeddings(openai_api_key=get_openai_settings().api_key)

    def run_example(self) -> None:
        self._init_database_data()

        # load stored embeddings from Chroma directory
        vectorstore = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=self.openai_embeddings,
        )
        # make a retriever
        vector_retriever: VectorStoreRetriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 2,  # number of output docs
            },
        )
        docs = vector_retriever.get_relevant_documents(query="Tell me more about ReAct prompting.")
        print(f"{vector_retriever.search_type = }")
        print(f"{docs[0].page_content = }")

        # make a retrieval chain
        retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            retriever=vector_retriever,
            return_source_documents=True,  # return pages which has been used for generating answer
        )

        llm_response = retrieval_qa_chain(inputs={"query": "Tell me about ReAct prompting"})
        self._process_llm_response(llm_response)

    def _init_database_data(self) -> None:
        # clean ChromaDB directory with all files
        shutil.rmtree(self._persist_directory)

        file_chunks = self._load_pdf_file_chunks()
        vectorstore = self._get_vector_store_from_documents(
            documents=file_chunks,
        )
        vectorstore.persist()

    def _load_pdf_file_chunks(self) -> list[Document]:
        loader = PyPDFLoader(file_path=self._pdf_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
        )
        return text_splitter.split_documents(documents=docs)

    def _get_vector_store_from_documents(
        self,
        documents: list[Document],
    ) -> Chroma:
        vectorstore = Chroma.from_documents(
            embedding=self.openai_embeddings,
            documents=documents,
            persist_directory=self._persist_directory,
        )
        vectorstore.persist()
        return vectorstore

    @staticmethod
    def _process_llm_response(response: dict[str, Any]) -> None:
        print(f"{response['result'] = }")
        print("\n\nSources: ")
        for source in response["source_documents"]:
            print(source.metadata)
