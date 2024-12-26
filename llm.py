import os
from dotenv import load_dotenv

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langsmith import traceable

from utils.utils import format_documents


class Llm:
    def __init__(self, data_path: str, rag_prompt_id: str, evaluate_prompt_id: str):
        pdf_loader = PyPDFLoader(data_path)
        self.pdf_documents = pdf_loader.load_and_split()

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vector_store = Chroma.from_documents(self.pdf_documents, embeddings)

        self.chat_model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.25)

        self.rag_prompt_template = hub.pull(rag_prompt_id)
        self.evaluate_prompt_template = hub.pull(evaluate_prompt_id)

    @traceable
    def retriever(self, query: str):
        retriever = self.vector_store.as_retriever()
        relevant_docs = retriever.get_relevant_documents(query)
        return relevant_docs

    @traceable
    def rag_chain(self, question: str):
        docs = self.retriever(question)
        format_context = format_documents(docs)
        rag_workflow = (
            RunnableParallel(
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            )
            | self.rag_prompt_template
            | self.chat_model
        )

        result = rag_workflow.invoke({"context": format_context, "question": question})
        return result

    @traceable
    def evaluate(
        self, question: str, answer: str, topic: str, criteria: str, examples: str
    ):
        input_data = {
            "modelInput": question,
            "modelOutput": answer,
            "criteria": criteria,
            "topic": topic,
            "examples": examples,
        }
        evaluate_workflow = self.evaluate_prompt_template | self.chat_model
        evaluation_result = evaluate_workflow.invoke(input_data)
        return evaluation_result
