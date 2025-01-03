from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage
from langchain.tools import StructuredTool
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

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


memory = MemorySaver()
graph_builder = StateGraph(State)


def search_func(query: str):
    search_tool = TavilySearchResults(
        max_results=2,
        include_answer=True,
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
    return search_tool.run(query)


search_tool = StructuredTool.from_function(
    func=search_func,
    name="search_external_info",
    description="用於搜尋網路資訊。當你無法從你的知識庫中找到答案時，可以使用此工具",
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

graph = graph_builder.compile()


graph_png = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(graph_png))
with st.sidebar:
    st.markdown("**work flow**")
    st.image(image)


user_input = st.text_input("輸入訊息:")

if st.button("送出"):
    if user_input:
        init_state = {"messages": [("user", user_input)]}
        for event in graph.stream(init_state, stream_mode="values"):
            last_message = event["messages"][-1]
            st.write(event["messages"][-1])
        with st.expander("AI摘要", expanded=True):
            if isinstance(last_message, AIMessage):
                st.write(last_message.content)
    else:
        st.error("請輸入訊息")
