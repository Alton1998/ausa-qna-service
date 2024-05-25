import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

load_dotenv()

client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))
db = client[os.getenv("MONGODB_NAME")]
MONGODB_COLLECTION = db[os.getenv("MONGODB_COLLECTION_NAME")]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "medical_info_index"


data = []
DIR = "./medical_training_docs"

for current_path, folders, files in os.walk(DIR):
    files = filter(lambda x:x.endswith(".pdf"),files)
    for file in files:
        data.extend(PyPDFLoader(os.path.join(current_path,file)).load())



text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")

vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=docs, embedding=embeddings, collection=MONGODB_COLLECTION, index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)