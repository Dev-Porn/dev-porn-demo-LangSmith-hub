from langgraph.graph import START, END, StateGraph
from llm import Llm
from typing import List, TypedDict


class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    web_search: str
    llm: Llm


def retrieve(state: GraphState):
    question = state["question"]
    llm = state["llm"]
    documents = llm.retriever(question)
    return {"documents": documents}


def grade_documents(state: GraphState):
    llm = state["llm"]
    question = state["question"]
    documents = state["documents"]
    filtered_docs, web_search = llm.evaluate_documents(question, documents)
    return {"documents": filtered_docs, "web_search": web_search}


def generate(state: GraphState):
    question = state["question"]
    llm = state["llm"]
    result = llm.rag_chain(question)
    return {"generation": result}


def transform_query(state: GraphState):
    llm = state["llm"]
    question = state["question"]
    better_question = llm.rewrite_question(question)
    return {"question": better_question}


def web_search(state: GraphState):
    llm = state["llm"]
    question = state["question"]
    web_result = llm.web_search(question)
    documents = state["documents"] + web_result
    return {"documents": documents}


def decide_to_generate(state: GraphState):
    web_search = state["web_search"]
    if web_search == "Yes":
        return "transform_query"
    else:
        return "generate"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
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
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
