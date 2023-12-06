from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

from langchain_poc.config import get_database_settings, get_openai_settings
from langchain_poc.examples.langchain.base import BaseExample


class EmbedsRetrievalPGVectorExample(BaseExample):
    _pdf_file_path = "../resources/the-three-sillies.txt"
    _collection_name = "fairy_tails"
    _template = """Use data from given embeddings to answer on user query: {query}.
    
    User these embeddings to find and answer {history}. 
    Return result as JSON array.
    """

    def __init__(self, chat_model: ChatOpenAI) -> None:
        super().__init__(chat_model)
        self._openai_embeddings = OpenAIEmbeddings(openai_api_key=get_openai_settings().api_key)
        self._pgvector_connection_url = PGVector.connection_string_from_db_params(
            driver=get_database_settings().default_postgres_driver,
            host=get_database_settings().host,
            port=get_database_settings().port,
            database=get_database_settings().db,
            user=get_database_settings().username,
            password=get_database_settings().password,
        )

    def run_example(self) -> None:
        docs = self._extract_docs()

        # init and populate table {_collection_name} with docs
        store = PGVector(
            collection_name=self._collection_name,
            connection_string=self._pgvector_connection_url,
            embedding_function=self._openai_embeddings,
            pre_delete_collection=True,  # override existing collection
        )
        store.add_documents(documents=docs)

        vector_retrieval: VectorStoreRetriever = store.as_retriever(
            search_kwargs={
                "k": 2,  # number of output docs
            },
        )

        prompt_template = ChatPromptTemplate.from_template(
            template=self._template,
        )
        memory = VectorStoreRetrieverMemory(retriever=vector_retrieval)
        chain = LLMChain(llm=self.chat_model, prompt=prompt_template, verbose=True, memory=memory)
        print(f'{chain("What is this story about? Summarize it.")["text"] = }')
        print(f'{chain("Summarize first paragraph.")["text"] = }')

    def _extract_docs(self) -> list[Document]:
        loader = TextLoader(self._pdf_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)
