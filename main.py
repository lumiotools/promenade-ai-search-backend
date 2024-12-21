from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from search import handle_search
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
    
class NodeModel(BaseModel):
    node_id: str
    content: str
    source: str
    doc_type: str
    
class MessageModel(BaseModel):
    role: str
    content: str
    
class ChatModel(BaseModel):
    message: str
    chat_history: List[MessageModel]
    search_query: str
    nodes: List[NodeModel]

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/api/search")
def search(body: QueryModel):
    try:
        answer, valid_sources, invalid_sources = handle_search(body.message)
        return {"response": answer, "sources": [], "valid_sources": valid_sources, "invalid_sources": invalid_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat(body: ChatModel):
    try:
        answer = handle_chat(body.nodes, body.search_query, body.chat_history, body.message)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))