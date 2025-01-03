import os

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
    """
    提供與大型語言模型 (LLM) 互動的功能，用於文件檢索、摘要、評估和問題重寫任務。

    屬性:
        data_path (str): PDF 資料的路徑。
        rag_prompt_id (str): 用於 RAG 提示模板 ID。
        evaluate_prompt_id (str): 用於答案評估提示模板 ID。
        evaluate_docs_prompt_id (str): 用於文件評估提示模板 ID。
        rewrite_question_prompt_id (str): 用於問題重寫提示模板 ID。
    """

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
        """
        根據查詢字串檢索相關文件。

        參數:
            query (str): 查詢字串。

        回傳:
            list: 與查詢相關的文件列表。
        """
        retriever = self.vector_store.as_retriever()
        relevant_docs = retriever.get_relevant_documents(query)
        return relevant_docs

    @traceable
    def rag_chain(self, question: str):
        """
        執行 RAG (檢索增強生成) 流程，回答指定問題。

        參數:
            question (str): 使用者的提問。

        回傳:
            str: RAG 生成的答案。
        """
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
        """
        根據網路搜尋的結果生成摘要。

        參數:
            documents (list): 文件列表。
            question (str): 使用者的提問。

        回傳:
            str: 根據文件內容和提問生成的摘要。
        """
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
        """
        評估答案的質量。

        參數:
            question (str): 問題。
            answer (str): 答案。
            topic (str): 主題。
            criteria (str): 評估標準。
            examples (str): 參考範例。

        回傳:
            dict: 評估結果。
        """
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
        """
        評估文件與問題的相關性，並決定是否需要進行網路搜尋。

        參數:
            question (str): 問題。
            documents (list): 文件列表。

        回傳:
            tuple: (相關文件列表, 是否需要網路搜尋)。
        """
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
        """
        重寫問題以提高檢索準確性。

        參數:
            question (str): 原始問題。

        回傳:
            str: 重寫後的問題。
        """
        input_data = {"question": question}
        rewrite_workflow = self.rewrite_question_prompt_template | self.chat_model
        rewrite_result = rewrite_workflow.invoke(input_data)
        return rewrite_result.content

    @traceable
    def web_search(self, question: str):
        """
        使用網路搜尋工具查詢答案。

        參數:
            question (str): 查詢問題。

        回傳:
            dict: 網路搜尋結果。
        """
        web_search_tool = TavilySearchResults(
            max_results=2,
            include_answer=True,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )
        return web_search_tool.invoke({"query": question})
