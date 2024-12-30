from typing import Annotated

from langchain_openai import ChatOpenAI


from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

import streamlit as st
from PIL import Image
import io
import os

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

search_tool = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
)


tools = [search_tool]

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.25)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# 可以自定義class BasicToolNode或使用預定義的ToolNode, 封裝和抽象中間過程
# ref : https://langchain-ai.github.io/langgraph/tutorials/introduction/#__codelineno-18-6
graph_builder.add_node("tools", ToolNode(tools))

# 可以自定義route_tool function 或直接使用tools_condition
# ref: https://langchain-ai.github.io/langgraph/tutorials/introduction/#__codelineno-19-1
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()


graph_png = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(graph_png))

st.image(image)


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            st.write(value["messages"][-1].content)


user_input = st.text_input("輸入訊息:")

if st.button("送出"):
    if user_input:
        stream_graph_updates(user_input)
    else:
        st.error("請輸入訊息")
