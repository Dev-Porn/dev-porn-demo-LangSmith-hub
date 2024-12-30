from typing import Annotated

from langchain_openai import ChatOpenAI

from langchain_core.messages import ToolMessage

from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

import streamlit as st
from PIL import Image
import io
import os
import json
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


class BasicNode:
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}
        logger.info(f"工具初始化: {list(self.tools_by_name.keys())}")

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("無訊息")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


graph_builder = StateGraph(State)

search_tool = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
)

tools = [search_tool]


llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.25)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()


graph_png = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(graph_png))

st.image(image)


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            st.write("Assistant:", value["messages"][-1].content)


user_input = st.text_input("輸入訊息:")

if st.button("送出"):
    if user_input:
        stream_graph_updates(user_input)
    else:
        st.error("請輸入訊息")
