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


class QueryModel(BaseModel):
    message: str
    files: Optional[List[str]] = []


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
        answer, valid_sources, invalid_sources = handle_search(body.message, file_ids=body.files)
        return {"response": answer, "valid_sources": valid_sources, "invalid_sources": invalid_sources}
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


@app.post("/api/upload_files")
def upload_file(files: Optional[List[UploadFile]] = []):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    try:
        file_ids = []
        for file in files:
            file_id = str(uuid4())
            file_path = f"{DOCUMENTS_UPLOAD_DIR}/{file_id}"
            os.makedirs(file_path, exist_ok=True)
            with open(f"{file_path}/document.pdf", "wb") as f:
                f.write(file.file.read())

            metadata = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file.file.tell(),
                "file_id": file_id,
            }

            with open(f"{file_path}/metadata.json", "w") as metadata_file:
                metadata_file.write(json.dumps(metadata, indent=4))
                
            file_ids.append(file_id)

        return JSONResponse(status_code=200, content={"success": True, "message": "File uploaded successfully","files": file_ids})

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
