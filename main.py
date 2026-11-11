import os
import json
from dotenv import load_dotenv

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.agent import init_graph, get_graph, get_checkpointer, pool
load_dotenv(override= True)

def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(
    key_func=get_client_ip,
    storage_uri=os.getenv("REDIS_URL", "memory://"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_graph()
    yield
    if pool:
        await pool.close()


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("3/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    graph = await get_graph()
    config = {"configurable": {"thread_id": body.thread_id}}

    async def event_generator():
        async for event in graph.astream_events(
            {"messages": [("user", body.message)]},
            config=config,
            version="v2",
        ):
            event_type = event.get("event")

            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk:
                    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                    if reasoning:
                        yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning})}\n\n"

                    content = chunk.content
                    if content:
                        if isinstance(content, list):
                            content = "".join(
                                c.get("text", "") if isinstance(c, dict) else str(c)
                                for c in content
                            )
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

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
@limiter.limit("30/minute")
async def get_history(request: Request, thread_id: str):
    graph = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    messages = state.values.get("messages", [])
    history = []
    for msg in messages:
        if hasattr(msg, "type"):
            history.append({"role": msg.type, "content": msg.content})
    return {"history": history}


@app.delete("/history/{thread_id}")
@limiter.limit("3/minute")
async def delete_history(request: Request, thread_id: str):
    checkpointer = await get_checkpointer()
    await checkpointer.adelete_thread(thread_id)
    return {"status": "deleted", "thread_id": thread_id}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)