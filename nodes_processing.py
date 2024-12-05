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

You are tasked with processing an array of content provided by the user. Your role is to rerank and reformat the content based on a user-provided prompt while keeping the text intact. You should not change or alter any of the text itself, but you are responsible for reordering the content and adjusting the format according to the following guidelines:

1. *Rerank the content*: Based on the user’s prompt, evaluate the provided content and reorder the array according to relevance, importance, or any other ranking criteria described in the prompt. 

2. *Sort the content according to rank: Once the content has been assigned a rank, **sort the array* so that items with the highest rank appear first. For example, if "top rank" corresponds to the highest priority, ensure that the item with the top rank is listed at the top of the array. This sorting should follow the rank order, from top to lowest.

3. *Reformat the content*: Adjust the formatting (e.g., bullet points, headers, numbered lists, indentation) according to the user’s desired structure. Ensure that the content is well-organized and easy to read, but do not change the actual wording or meaning of the content.

4. *Output the content in sorted order*: After reranking, provide the content in the newly ordered sequence, ensuring that the formatting changes are consistent with the structure requested.

The user will provide both the content array and specific instructions for reranking and formatting. Ensure that the final output is in the correct order and format, as per the user’s guidelines.

"""

class Content(BaseModel):
    content: str
    source: str

class ResponseFormat(BaseModel):
    contents: List[Content]
    
    
    
def process_nodes(company_name,query,contents):
    
    chat_completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{
            "role": "system", "content": system_prompt
            },
             {"role": "user", 
              "content": f"""
              This is the company name: {company_name}
              This is the user query: {query}
              Here are my contents:
              {json.dumps(contents)}
              
              Based on this data, Re-rank my content and provide me the most relevant information in sorted way.
              """
            }
                
    ],
        response_format=ResponseFormat
    )

    res = chat_completion.choices[0].message.parsed
    
    
    
    return res.model_dump()["contents"]
    
