from fastapi import FastAPI
from pydantic import BaseModel
from chat import handle_chat
from live_search import handle_live_search
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

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/api/chat")
def chat(body: QueryModel):
    # chat_answer, chat_valid_sources, chat_invalid_sources = handle_chat(
    #     body.message)
    # live_search_answer, live_search_valid_sources, live_search_invalid_sources = handle_live_search(
    #     body.message)

    # answer, valid_sources, invalid_sources = chat_answer+live_search_answer, chat_valid_sources + \
    #     live_search_valid_sources, chat_invalid_sources+live_search_valid_sources
    
    answer, valid_sources, invalid_sources = handle_chat(body.message)
    return {"response": answer, "sources": [], "valid_sources": valid_sources, "invalid_sources": invalid_sources}
