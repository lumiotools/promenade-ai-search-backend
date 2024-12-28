from uuid import uuid4
from typing import List
from pypdf import PdfReader
import os
import json
import requests
from pydantic import BaseModel

UPLOAD_DIR = "document_uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
class FileModel(BaseModel):
    name: str
    url: str

def handle_live_document_search(files: List[FileModel]):
    try:
        nodes = []
        file_ids = []
        
        for file in files:
            file_id = str(uuid4())
            
            os.makedirs(os.path.join(UPLOAD_DIR, file_id))
            
            response = requests.get(file.url)
            if response.status_code != 200:
                continue
            
            with open(os.path.join(UPLOAD_DIR, file_id, "document.pdf"), "wb") as document_file:
                document_file.write(response.content)
            
            with open(os.path.join(UPLOAD_DIR, file_id, "metadata.json"), "w") as metadata_file:
                metadata = {
                    "filename": file.name,
                    "file_url": file.url,
                    "file_id": file_id,
                }
                metadata_file.write(json.dumps(metadata, indent=4))
                
            file_ids.append(file_id)

        
        for file_id in file_ids:
            
            if not os.path.exists(os.path.join(UPLOAD_DIR, file_id)):
                continue
            
            with open(os.path.join(UPLOAD_DIR, file_id, "metadata.json"), "r") as metadata_file:
                metadata = json.load(metadata_file)
            
            
            print(f"Processing Uploaded Document: {metadata['filename']}")
                
            reader = PdfReader(os.path.join(UPLOAD_DIR, file_id, "document.pdf"))
            page_count = len(reader.pages)
            for start in range(0, page_count, 10):
                end = min(start + 10, page_count)
                node_content = ""
                for index in range(start, end):
                    if index > start:
                        node_content += "\n"
                    node_content += reader.pages[index].extract_text(extraction_mode="layout", space_width=10)
                
                print(f"Extracted content from pages {start} to {end} of {metadata['filename']}")
                
                nodes.append({
                    "node_id": str(uuid4()),
                    "content": node_content,
                    "title": metadata["filename"],
                    "source": metadata["file_url"],
                })
            
            os.remove(os.path.join(UPLOAD_DIR, file_id, "document.pdf"))
            os.remove(os.path.join(UPLOAD_DIR, file_id, "metadata.json"))
            os.rmdir(os.path.join(UPLOAD_DIR, file_id))
                

        return nodes
        
    except Exception as e:
        print(e)

        return []