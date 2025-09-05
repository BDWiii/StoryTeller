from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional
from main import RunStoryTeller


app = FastAPI()
runner = RunStoryTeller()


class ChatRequest(BaseModel):
    task: str
    thread_id: Optional[str] = None


class AgentResponse(BaseModel):
    story: str
    thread_id: str


@app.post("/run", response_model=AgentResponse)
async def run_agent(request: ChatRequest):

    input_text = request.task
    thread_id = request.thread_id

    if thread_id:
        runner.thread_id = thread_id
        runner.config = {"configurable": {"thread_id": thread_id}}
        result = runner.existing_thread(input_text)
    else:
        result = runner.new_thread(input_text)

    return {"story": result["story"], "thread_id": result["session_id"]}


@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    try:
        snapshot = runner.get_current_state(thread_id)
        return snapshot
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "✅ ✅ ✅"}
