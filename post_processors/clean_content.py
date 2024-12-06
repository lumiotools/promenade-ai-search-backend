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

You are tasked with processing an array of content provided by the user. Your role is to clean and reformat the content based on a user-provided prompt while keeping the text intact. You should not change or alter any of the words or sentences, but you are responsible for reformatting the content and adjusting the markdown format according to the following guidelines:

*Reformat the content*: Adjust the formatting according to the user's desired structure. If the content is in a table format, replace it with a more suitable paragraph format or other preferred formats. Ensure that the content is well-organized and easy to read, but do not change the actual wording or meaning of the content.

*Crop Unnecessary Text*: Trim any text that is not directly related to the user's query.

*Maintain Integrity of the Text*: The words and sentences should remain unchanged; only formatting and structure should be modified to improve clarity and readability.

Remove this kind of headers if present:
    Company: name...
    Section: section name...
    Title: something...
    URL: https:.....

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
    
