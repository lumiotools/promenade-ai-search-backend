from uuid import uuid4
from typing import List
from fastapi import UploadFile
from pypdf import PdfReader
import os
import json

UPLOAD_DIR = "document_uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def handle_live_document_search(file_ids: List[str]):
    try:
        nodes = []
        
        for file_id in file_ids:
            node_id = file_id
            
            if not os.path.exists(os.path.join(UPLOAD_DIR, file_id)):
                continue
                
            reader = PdfReader(os.path.join(UPLOAD_DIR, file_id, "document.pdf"))
            content = ""
            
            for index, page in enumerate(reader.pages):
                if index > 0:
                    content += "\n"
                content += page.extract_text(extraction_mode="layout",space_width=10)
            
            with open(os.path.join(UPLOAD_DIR, file_id, "metadata.json"), "r") as metadata_file:
                metadata = json.load(metadata_file)
                
            nodes.append({
                "node_id": node_id,
                "content": content,
                "title": metadata["filename"],
                "source": f"{os.getenv('API_BASE_URL')}/files/{node_id}.pdf",
            })
        
        return nodes
        
    except Exception as e:
        print(e)

        return []