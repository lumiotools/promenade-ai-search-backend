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

You are a precise content extraction AI. Your goal is to find and extract the most relevant text snippet in response to the user's query.

Extraction Guidelines:
1. Primary Goal: Identify the exact text segment that most precisely answers the query
2. Secondary Strategy: If no exact snippet found, extract the most relevant and informative segment
3. Remove all metadata, headers, and contextual information
4. Preserve the original wording exactly

Extraction Hierarchy:
- First Priority: Exact snippet directly answering the query
- Second Priority: Most relevant and informative text segment
- Last Resort: Most closely related content, even if partially relevant

Removal Criteria:
- Delete all headers (Company:, Section:, Title:, URL:)
- Eliminate introductory or surrounding text
- Focus on content most closely aligned with the query

Fallback Mechanism:
- If no relevant content is found, return a segment that provides the closest contextual information
- Ensure the returned text offers some value or insight related to the query

Core Objective: Deliver the most precise, contextually relevant text that provides meaningful information in response to the user's query.
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
    
