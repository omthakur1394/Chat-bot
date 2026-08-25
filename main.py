import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from src.agent import graph


app = FastAPI()
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
    config = {"configurable": {"thread_id": request.thread_id}}

    def event_generator():
        for event in graph.stream_events(
            {"messages": [("user", request.message)]},
            config=config,
            version="v2",
        ):
            event_type = event.get("event")

            # Stream LLM tokens as they are generated
            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and chunk.content:
                    content = chunk.content
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

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
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
    graph.checkpointer.delete_thread(thread_id)
    return {"status": "deleted", "thread_id": thread_id}
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
