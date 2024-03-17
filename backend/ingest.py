"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.astradb import AstraDB
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"] 
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]


def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embeddings = get_embeddings_model()
    vstore = AstraDB(
        embedding=embeddings,
        collection_name="genai",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
    )

    loader = PyPDFDirectoryLoader("../data/", recursive=True)
    docs = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
        chunk_overlap=50))

    vstore.add_documents(docs)
    num_vecs = len(docs)
    logger.info(
        f"LangChain now has this many vectors: {num_vecs}",
    )

if __name__ == "__main__":
    ingest_docs()
