import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from psycopg_pool import AsyncConnectionPool

from src.config import llm 
from src.tools import tools

load_dotenv()
llm_with_tools = llm.bind_tools(tools)
DB_URI = os.getenv("DATABASE_URL")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def tools_calling(state: State):
    return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("tools_calling", tools_calling)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tools_calling")
builder.add_conditional_edges("tools_calling", tools_condition)
builder.add_edge("tools", "tools_calling")

pool = AsyncConnectionPool(
    conninfo=DB_URI,
    max_size=10,
    max_idle=300,
    max_lifetime=3600,
    reconnect_timeout=30,
    kwargs={"autocommit": True, "sslmode": "require"},
    open=False,
)

checkpointer = None
graph = None


async def init_graph():
    global checkpointer, graph
    if graph is None:
        await pool.open()
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
    return graph


async def get_graph():
    global graph
    if graph is None:
        await init_graph()
    return graph


async def get_checkpointer():
    global checkpointer
    if checkpointer is None:
        await init_graph()
    return checkpointer
