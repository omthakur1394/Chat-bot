from pydantic import BaseModel
from src.agent import graph
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    message:str
    thread_id: str = "1"

@app.post("/chat")
async def chat_endpoint(request:ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    result = graph.invoke(
        {"messages": [("user", request.message)]}, 
        config=config
    )
    return {"response": result["messages"][-1].content}
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