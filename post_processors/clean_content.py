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

You are a precise content extraction AI. Your goal is to find and extract the smallest, most relevant text snippet that directly answers the user's query.

Extraction Guidelines:
1. Identify the exact text segment that most precisely answers the query
2. Remove all metadata, headers, and contextual information
3. Extract only the minimal, essential text needed to address the query
4. Preserve the original wording exactly
5. Return the shortest possible snippet that fully answers the question
6. If the answer is not present in the text, return the most relevant segment

Removal Criteria:
- Delete all headers (Company:, Section:, Title:, URL:)
- Eliminate introductory or surrounding text
- Keep only the core, direct answer
- If no exact mathcing snippet found, return the most relevant segment

Core Objective: Surgically extract the most concise, relevant text snippet that comprehensively answers the user's query.

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
    
