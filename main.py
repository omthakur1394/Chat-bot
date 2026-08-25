import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from src.agent import init_graph, get_graph, get_checkpointer, pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_graph()
    yield
    if pool:
        await pool.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    graph = await get_graph()
    config = {"configurable": {"thread_id": request.thread_id}}

    async def event_generator():
        async for event in graph.astream_events(
            {"messages": [("user", request.message)]},
            config=config,
            version="v2",
        ):
            event_type = event.get("event")

            # Stream LLM tokens as they are generated
            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk:
                    # Stream reasoning tokens (e.g. Groq reasoning models like gpt-oss-120b / deepseek)
                    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                    if reasoning:
                        yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning})}\n\n"

                    # Stream answer content tokens
                    content = chunk.content
                    if content:
                        if isinstance(content, list):
                            content = "".join(
                                c.get("text", "") if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            # Stream tool calls
            elif event_type == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': event.get('name'), 'input': event.get('data', {}).get('input')})}\n\n"

            elif event_type == "on_tool_end":
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': event.get('name'), 'output': str(event.get('data', {}).get('output'))})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    messages = state.values.get("messages", [])
    history = []
    for msg in messages:
        if hasattr(msg, "type"):
            history.append({
                "role": msg.type,
                "content": msg.content
            })
    return {"history": history}


@app.delete("/history/{thread_id}")
async def delete_history(thread_id: str):
    checkpointer = await get_checkpointer()
    await checkpointer.adelete_thread(thread_id)
    return {"status": "deleted", "thread_id": thread_id}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
