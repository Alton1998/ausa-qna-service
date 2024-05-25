import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.llamafile import Llamafile
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnablePick

load_dotenv()

MONGO_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = os.getenv("MONGODB_NAME")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = "medical_info_index"

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    DB_NAME + "." + COLLECTION_NAME,
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

llm = Llamafile()
llm.invoke()


# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_prompt = PromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following "
                                          "pieces of retrieved context to answer the question. If you don't know the "
                                          "answer, just say that you don't know. Use three sentences maximum and keep "
                                          "the answer concise.Question: {question} Context: {context} Answer:")

chain = (
        RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
        | rag_prompt
        | llm
        | StrOutputParser()
)

question = "Tell me about gpt 4 compute"
chain.invoke({"context": docs, "question": question})
