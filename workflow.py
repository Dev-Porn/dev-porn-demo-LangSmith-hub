from langgraph.graph import START, END, StateGraph
from llm import Llm
from typing import List, TypedDict

from utils.utils import save_grpah_as_image


class GraphState(TypedDict):
    """
    定義狀態圖中的節點狀態。

    屬性:
        question (str): 問題字串。
        documents (List[str]): 文件內容的列表。
        generation (str): LLM 生成的結果。
        web_search (str): 是否需要進行網路搜尋 ("Yes" 或 "No")。
        llm (Llm): 用於處理 LLM 的實例。
    """

    question: str
    documents: List[str]
    generation: str
    web_search: str
    llm: Llm


def retrieve(state: GraphState):
    """
    根據問題從向量存儲中檢索相關文件。

    參數:
        state (GraphState): 當前節點的狀態。

    回傳:
        dict: 包含檢索到的文件的字典。
    """
    question = state["question"]
    llm = state["llm"]
    documents = llm.retriever(question)
    return {"documents": documents}


def grade_documents(state: GraphState):
    """
    根據問題評估文件的相關性，並決定是否需要網路搜尋。

    參數:
        state (GraphState): 當前節點的狀態。

    回傳:
        dict: 包含篩選後的文件列表和網路搜尋標記的字典。
    """
    llm = state["llm"]
    question = state["question"]
    documents = state["documents"]
    filtered_docs, web_search = llm.evaluate_documents(question, documents)
    return {"documents": filtered_docs, "web_search": web_search}


def generate(state: GraphState):
    """
    使用 RAG 流程生成答案。

    參數:
        state (GraphState): 當前節點的狀態。

    回傳:
        dict: 包含生成結果的字典。
    """
    question = state["question"]
    llm = state["llm"]
    result = llm.rag_chain(question)
    return {"generation": result}


def transform_query(state: GraphState):
    """
    改寫問題以提高檢索和生成的準確性。

    參數:
        state (GraphState): 當前節點的狀態。

    回傳:
        dict: 包含改寫後問題的字典。
    """
    llm = state["llm"]
    question = state["question"]
    better_question = llm.rewrite_question(question)
    return {"question": better_question}


def web_search(state: GraphState):
    """
    使用網路搜尋工具查詢問題，並將結果添加到現有文件列表中。

    參數:
        state (GraphState): 當前節點的狀態。

    回傳:
        dict: 包含更新後文件列表的字典。
    """
    llm = state["llm"]
    question = state["question"]
    web_result = llm.web_search(question)
    documents = state["documents"] + web_result
    return {"documents": documents}


def summarize_web_result(state: GraphState):
    """
    根據網路搜尋結果生成摘要。

    參數:
        state (GraphState): 當前節點的狀態。

    回傳:
        dict: 包含摘要結果的字典。
    """
    llm = state["llm"]
    question = state["question"]
    documents = state["documents"]
    generation = llm.summarize_web_result(documents, question)
    return {"generation": generation}


def decide_to_generate(state: GraphState):
    """
    根據文件評估結果決定下一步操作。

    參數:
        state (GraphState): 當前節點的狀態。

    回傳:
        str: 下一步操作的節點名稱。
    """
    web_search = state["web_search"]
    if web_search == "Yes":
        return "web_search_node"
    else:
        return "generate"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("summarize_web_result", summarize_web_result)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "web_search_node": "web_search_node",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "summarize_web_result")
workflow.add_edge("generate", END)
workflow.add_edge("summarize_web_result", END)
app = workflow.compile()

# save_grpah_as_image(app, "data/graph.png")
