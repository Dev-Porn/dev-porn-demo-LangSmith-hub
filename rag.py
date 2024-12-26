import os
from dotenv import load_dotenv

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langsmith import traceable

from utils.utils import format_documents

load_dotenv()

pdf_loader = PyPDFLoader(f"{os.getenv('DATA_PATH')}")

pdf_documents = pdf_loader.load_and_split()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = Chroma.from_documents(pdf_documents, embeddings)


chat_model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.25)


prompt_template = hub.pull("daethyra/rag-prompt")


@traceable
def retriever(query: str):
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)
    return relevant_docs


@traceable
def rag_chain(question: str):
    docs = retriever(question)
    format_context = format_documents(docs)
    rag_workflow = (
        RunnableParallel(
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        )
        | prompt_template
        | chat_model
    )

    result = rag_workflow.invoke({"context": format_context, "question": question})
    return result


rag_chain("比特幣存在的目的?")

# rag_chain("以太幣存在的目的??")
