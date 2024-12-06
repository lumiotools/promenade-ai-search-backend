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

You are tasked with processing an array of content provided by the user. Your role is to clean and reformat the content based on a user-provided prompt while keeping the text intact. You should not change or alter any of the words, but you are responsible for reformating the content and adjusting the markdown format according to the following guidelines:

*Reformat the content*: Adjust the formatting (e.g., bullet points, headers, numbered lists, indentation) according to the user’s desired structure. Ensure that the content is well-organized and easy to read, but do not change the actual wording or meaning of the content.

*Crop Unnecessary Text*: Trim the text that are not related to user query.

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
            "role": "system", "content": system_prompt
            },
             {"role": "user", 
              "content": f"""
              This is the user query: {query}
              
              For my below nodes contents, reformat their content by applying and fixing the markdown and provide me the cleaned content.
              Preserve the order of the nodes and return in same order.
              
              These are my nodes:
              {json.dumps(re_ranked_nodes)}
              """
            }
                
    ],
        response_format=ResponseFormat
    )

    res = chat_completion.choices[0].message.parsed
    
    
    
    return res.model_dump()["nodes"]
    
