from fastapi import FastAPI
from pydantic import BaseModel
from chat import handle_chat

app = FastAPI()


class QueryModel(BaseModel):
    message: str


@app.post("/api/chat")
def chat(body: QueryModel):
    answer, sources = handle_chat(body.message)
    return {"response": answer, "sources": sources}
