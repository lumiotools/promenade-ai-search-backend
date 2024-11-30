from fastapi import FastAPI
from pydantic import BaseModel
from chat import handle_chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    message: str


@app.post("/api/chat")
def chat(body: QueryModel):
    answer, sources = handle_chat(body.message)
    return {"response": answer, "sources": sources}
