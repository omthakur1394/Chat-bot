from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from .config import llm 
from .dotenv import load_dotenv
import os 
from tools import tools
load_dotenv ()
llm_with_tools = llm.bind_tools(tools)
DB_URI = os.getenv("DATABASE_URL")

class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

def tools_calling(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}
bulider = StateGraph(State)

bulider.add_node("tools_calling",tools_calling)
bulider.add_node("tools",ToolNode(tools))
bulider.add_edge(START,"tools_calling")
bulider.add_conditional_edges("tools_calling",tools_condition)
bulider.add_edge("tools","tools_calling")

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = bulider.compile(checkpointer=checkpointer)

