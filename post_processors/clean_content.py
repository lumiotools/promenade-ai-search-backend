from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
from enum import Enum
from urllib import parse
from token_calculation import calculate_token

load_dotenv()
client = OpenAI()

system_prompt = """
You are a highly precise content extraction AI. Your task is to clean the content of each node in a provided list while retaining the original order of nodes. Each node has the following properties:  
- `node_id` (unique identifier)  
- `content` (text to be cleaned)  

Your goal is to ensure that the content of each node contains only relevant information that directly addresses the user's query. The cleaned content must preserve the original structure, sentence order, and phrasing of the source. Do not summarize, rewrite, or add interpretative elements. If necessary to meet the minimum word count, only insert small, relevant parts of adjacent content while maintaining the original tone.

---

**Processing Guidelines**:

1. **Identify Core Content**:  
   - For each node, locate the segment within the `content` that directly addresses or is most relevant to the user's query.  
   - Retain only the exact sentences or phrases that are relevant to the query, maintaining the original order of those sentences.  

2. **Eliminate Non-Essential Content**:  
   - Remove unrelated sentences, headers, footers, and metadata (e.g., "Company:", "Section:", "Title:", or "URL:").  
   - Exclude participant lists, operator instructions, and disclaimers, such as:  
     ```
     _This article is a transcript of this conference call produced for The Motley Fool..._
     ```  

3. **Preserve Original Wording and Sentence Order**:  
   - Ensure that the output matches the original text exactly, using the same sentence structure, phrasing, and order as in the source content.  
   - Do not paraphrase, summarize, or interpret the text.  

4. **Minimum Word Requirement**:  
   - Ensure the cleaned content for each node is **at least 100 words**.  
   - If the relevant content is less than 100 words, pull directly from nearby, contextually relevant text in the same node to meet the word count.  
   - Avoid inserting or fabricating unrelated information.  

5. **Query Alignment**:  
   - Ensure the cleaned content aligns directly with the user's query and answers it precisely.  
   - Remove text that does not contribute to resolving the query.  

6. **Maintain Node Order and Structure**:  
   - Process each node individually, maintaining its original position in the list.  
   - Ensure that the `node_id` remains unchanged and that nodes are not removed or reordered.  

---

**Core Objective**:  
For each node, return the cleaned `content` that directly addresses the query in the exact original tone, structure, and sentence order. The cleaned content must meet the **minimum word requirement of 100 words** without summarizing or interpreting the text. It should read as though it is directly extracted from the source.
"""


def clean_contents(query,re_ranked_nodes):
    
    messages = [{
            "role": "system", "content": system_prompt.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            },
             {"role": "user", 
              "content": f"""
              This is the user query: {query}
              
              For my below nodes contents, reformat their content by applying and fixing the markdown and provide me the cleaned content that answers the above mentioned users query.
              Preserve the order of the nodes and return in same order.
              
              These are my nodes:
              {json.dumps(re_ranked_nodes)}
              """.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            }
                
    ]
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "cleaned_content",
            "schema": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cleaned_content": {"type": "string"},
                                "node_id": {"type": "string"}
                            },
                            "required": ["cleaned_content", "node_id"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["nodes"],
                "additionalProperties": False
            },
            "strict": True
        }
        },
        temperature=0
    )
    
    res = chat_completion.choices[0].message.content
    
    messages.append({"role":"assistant", "content":res})
    
    token_usage = calculate_token([message["content"] for message in messages])
    
    highlights = []
    for node in re_ranked_nodes:
        texts = node["content"].split(" ")
        start = texts[:3]
        end = texts[-3:]
        
        highlights.append(f"{parse.quote(' '.join(start))},{parse.quote(' '.join(end))}")
        
    nodes =  json.loads(res)["nodes"]
    
    for i, node in enumerate(nodes):
        node["highlight"] = highlights[i]

    return nodes, token_usage
    
