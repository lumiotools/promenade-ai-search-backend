from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from search import handle_search
from chat import handle_chat
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from uuid import uuid4

DOCUMENTS_UPLOAD_DIR = "document_uploads"

if not os.path.exists(DOCUMENTS_UPLOAD_DIR):
    os.makedirs(DOCUMENTS_UPLOAD_DIR)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileModel(BaseModel):
    name: str
    url: str

class QueryModel(BaseModel):
    message: str
    files: Optional[List[FileModel]] = []


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
async def search(body: QueryModel):
    try:
        nodes, valid_sources, invalid_sources = handle_search(body.message, files=body.files)
        summary = handle_chat([NodeModel(**node) for node in nodes], body.message, [], """
                            Provide a professional research memo as if you are a 3rd-year Business Analyst at Goldman Sachs' Investment Banking Division and an HBS graduate. The memo should meet the following criteria:
                            - Start with a one-sentence summary that provides a clear overview of the investment thesis or opportunity.
                            - Structure the content using key topics, written in bold bullet points.
                            - Each bullet point should have: A clear title (in bold). Concise supporting points explaining the topic. Conclude with a summary statement, summarizing the recommendation or key takeaway in a professional tone.
                            - Keep the memo 100 words maximum. Ensure the tone is formal and professional. Write in polished business English.  
                            - Do not include the titles like "Memo:" or "Conclusion:" in the memo.
                            """)
        return {"response": nodes, "summary": summary, "valid_sources": valid_sources, "invalid_sources": invalid_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def chat(body: ChatModel):
    try:
        answer = handle_chat(body.nodes, body.search_query,
                             body.chat_history, body.message)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{file_name}")
def get_file(file_name: str):

    file_id = file_name.split(".")[0]

    if not os.path.exists(f"{DOCUMENTS_UPLOAD_DIR}/{file_id}"):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(f"{DOCUMENTS_UPLOAD_DIR}/{file_id}/metadata.json", "r") as file:
            metadata = json.load(file)
        return FileResponse(f"{DOCUMENTS_UPLOAD_DIR}/{file_id}/document.pdf", filename=metadata["filename"], content_disposition_type="inline")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
