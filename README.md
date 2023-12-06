# LangChain-POC

Proof of concept of using LangChain

## Project Requirements

- Python 3.11
- langchain
- openai
- streamlit
- pypdf
- tiktoken
- chromadb  <!-- (Vector Embeds Store Database) -->
- wikipedia <!-- (Docsstore client) -->
- google-search-results <!-- (Self Asked Agent) -->
- pgvector
- psycopg2-binary

## Project Setup

1. Create and activate **venv**

> - python -m venv /path/to/new/virtual/environment
> - source venv/bin/activate

2. Upgrade pip to the latest version

> pip install --upgrade pip

3. Install and init **poetry**

> - pip install poetry
> - poetry install

4. Create **.env** file from ****[example.local.env](example.local.env)**** and place it on the same level. Populate *
   *OPENAI_API_KEY** variable.

## Run Apps

### Streamlit Example Web App

To run Streamlit example app create run next command.
> streamlit run langchain_poc/examples/streamlit/<module_name>.py

### LangChain Example Code

To run LangChain code examples chose desire example in langchain_poc/app.py module and run next command.
> python3 langchain_poc/app.py