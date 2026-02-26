# Modular LangGraph & FastAPI Chatbot

A production-ready, modular AI chatbot built with FastAPI, LangGraph, and LangChain. This agent leverages Groq's blazing-fast inference and is equipped with external tools (Wikipedia, ArXiv, and Tavily Search) to actively research and answer complex user queries.

## 🚀 Features

- **Agentic Framework:** Built using LangGraph for state management and cyclic routing.
- **Tool Calling:** The LLM can dynamically search Wikipedia, query academic papers via ArXiv, and browse the web using Tavily.
- **Memory Management:** Implements `MemorySaver` to maintain thread-level conversational history.
- **Modular Architecture:** Clean separation of concerns (API routes, tools, agent logic, and config) for easy scaling.
- **Production Ready:** Includes a `Dockerfile` and Uvicorn setup optimized for cloud deployment.

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **AI/LLM:** LangChain, LangGraph, Groq (Llama-3.1)
- **Tools:** ArxivAPI, WikipediaAPI, Tavily Search
- **Deployment:** Docker, Uvicorn

## 📂 Project Structure

```text
├── src/
│   ├── __init__.py
│   ├── agent.py       # LangGraph state machine and memory compilation
│   ├── config.py      # LLM initialization and environment loading
│   ├── schemas.py     # Pydantic models for API validation
│   └── tools.py       # External tool initialization (Arxiv, Wiki, Tavily)
├── main.py            # FastAPI application entry point
├── Dockerfile         # Containerization instructions
├── requirements.txt   # Python dependencies
└── .env               # API keys (Not pushed to version control)
```
