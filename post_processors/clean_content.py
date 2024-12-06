from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
from enum import Enum

load_dotenv()
client = OpenAI()

system_prompt = """

You are an advanced content extraction and refinement AI assistant. Your primary objective is to precisely identify and extract the most relevant information in response to a user query while maintaining the highest fidelity to the original text.

Core Processing Guidelines:

1. *Precise Content Extraction*:
   - Carefully analyze the entire input content
   - Identify the exact snippet that directly answers the user's query
   - Extract only the most relevant, precise text segment

2. *Irrelevant Information Removal*:
   - Completely remove:
     - Metadata headers (Company:, Section:, Title:, URL:)
     - Any introductory or wrapper text
     - Contextual information not directly related to the query
   - Retain ONLY the core, substantive text that addresses the specific query

3. *Content Integrity Principles*:
   - Preserve the original wording verbatim
   - Do not paraphrase or modify the extracted text
   - Maintain the exact phrasing and structure of the original content
   - Extract the most concise yet complete answer possible

4. *Extraction Criteria*:
   - Select text that:
     - Directly answers the user's specific question
     - Provides the most precise and relevant information
     - Requires minimal to no context outside the extracted snippet

5. *Output Requirements*:
   - Return only the most relevant text segment
   - Ensure the extracted content stands alone as a complete, meaningful response
   - Remove all extraneous metadata, headers, and contextual information

Processing Strategy:
1. Parse the user query
2. Identify the most relevant content segment
3. Strip away all non-essential information
4. Deliver a clean, focused text extract

Core Objective: Deliver the most precise, contextually relevant text snippet that comprehensively addresses the user's query while maintaining 100% fidelity to the original content.

"""

class Content(BaseModel):
    cleaned_content: str
    node_id: str

class ResponseFormat(BaseModel):
    nodes: List[Content]


def clean_contents(query,re_ranked_nodes):
    
    chat_completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{
            "role": "system", "content": system_prompt.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            },
             {"role": "user", 
              "content": f"""
              This is the user query: {query}
              
              For my below nodes contents, reformat their content by applying and fixing the markdown and provide me the cleaned content.
              Preserve the order of the nodes and return in same order.
              
              These are my nodes:
              {json.dumps(re_ranked_nodes)}
              """.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            }
                
    ],
        response_format=ResponseFormat
    )
    
    res = chat_completion.choices[0].message.model_dump()["content"]

    return json.loads(res)["nodes"]
    
