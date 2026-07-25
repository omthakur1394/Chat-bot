from dotenv import load_dotenv
from langchain.agents import create_agent
from src.tools import tools
from langgraph.checkpoint.memory import InMemorySaver
from src.config import llm

load_dotenv()

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=InMemorySaver(),
)


def invoke_agentv2(message: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return result["messages"][-1].content
