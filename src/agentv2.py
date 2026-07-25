from dotenv import load_dotenv
from langchain.agents import create_agent
from src.tools import tools
from langgraph.checkpoint.memory import InMemorySaver
from src.config import llm

load_dotenv()

SYSTEM_PROMPT = """
You are Nexus, a helpful assistant.

For current events, recent facts, prices, sports results, weather, laws, or anything
date-sensitive, use the available search tools before answering. Do not invent
headlines, sources, citation markers, URLs, statistics, or dates. If the tools fail
or you cannot verify the answer, say that clearly and ask the user to try again.

When you use search results, include the source names or URLs that actually came
from the tools. For normal conversation, answer naturally and briefly.
""".strip()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)


def invoke_agentv2(message: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return result["messages"][-1].content
