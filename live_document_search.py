from typing import List
import os
from pydantic import BaseModel
from live_search.pdf import get_pdf_content_nodes
from concurrent.futures import ThreadPoolExecutor

UPLOAD_DIR = "document_uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
class FileModel(BaseModel):
    name: str
    url: str

def handle_live_document_search(files: List[FileModel]):
    try:
        nodes = []
        
        with ThreadPoolExecutor() as executor:
            file_nodes_list = list(executor.map(get_pdf_content_nodes, files))
            for file_nodes in file_nodes_list:
                nodes.extend(file_nodes)
            
        return nodes
        
    except Exception as e:
        print(e)

        return []