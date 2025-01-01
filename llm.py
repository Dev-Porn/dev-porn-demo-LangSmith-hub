import os
from dotenv import load_dotenv

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langsmith import traceable

from utils.utils import format_documents, format_web_search


class Llm:
    def __init__(
        self,
        data_path: str,
        rag_prompt_id: str,
        evaluate_prompt_id: str,
        evaluate_docs_prompt_id: str,
        rewrite_question_prompt_id: str,
    ):
        pdf_loader = PyPDFLoader(data_path)
        self.pdf_documents = pdf_loader.load_and_split()

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vector_store = Chroma.from_documents(self.pdf_documents, embeddings)

        self.chat_model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
        self.summarize_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an expert summarizer. Based on the following documents, provide a concise summary that answers the question.

            Documents:
            {context}

            Question:
            {question}

            Summary:
            """,
        )

        self.rag_prompt_template = hub.pull(rag_prompt_id)
        self.evaluate_documents_prompt_template = hub.pull(evaluate_docs_prompt_id)
        self.evaluate_answer_prompt_template = hub.pull(evaluate_prompt_id)
        self.rewrite_question_prompt_template = hub.pull(rewrite_question_prompt_id)

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
    def summarize_web_result(self, documents: list, question: str):
        format_context = format_web_search(documents)
        summarize_prompt = self.summarize_prompt_template.format(
            context=format_context, question=question
        )
        result = self.chat_model.invoke(summarize_prompt)
        print(result)
        return result

    @traceable
    def evaluate_answer(
        self, question: str, answer: str, topic: str, criteria: str, examples: str
    ):
        input_data = {
            "modelInput": question,
            "modelOutput": answer,
            "criteria": criteria,
            "topic": topic,
            "examples": examples,
        }
        evaluate_workflow = self.evaluate_answer_prompt_template | self.chat_model
        evaluation_result = evaluate_workflow.invoke(input_data)
        return evaluation_result

    @traceable
    def evaluate_documents(self, question: str, documents: list):
        filtered_docs = []
        web_search = "No"
        for doc in documents:
            input_data = {"question": question, "context": doc.page_content}
            evaluate_docs_workflow = (
                self.evaluate_documents_prompt_template | self.chat_model
            )
            evaluate_docs_result = evaluate_docs_workflow.invoke(input_data)
            print("Evaluation result raw output:", evaluate_docs_result)
            if "yes" in evaluate_docs_result.content.lower():
                filtered_docs.append(doc)
        if len(filtered_docs) == 0:
            web_search = "Yes"
        else:
            web_search = "No"
        print("相關文件數:", len(filtered_docs))
        return filtered_docs, web_search

    @traceable
    def rewrite_question(self, question: str):
        input_data = {"question": question}
        rewrite_workflow = self.rewrite_question_prompt_template | self.chat_model
        rewrite_result = rewrite_workflow.invoke(input_data)
        return rewrite_result.content

    @traceable
    def web_search(self, question: str):
        web_search_tool = TavilySearchResults(
            max_results=2,
            include_answer=True,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )
        return web_search_tool.invoke({"query": question})
