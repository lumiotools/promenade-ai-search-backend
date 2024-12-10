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
You are a highly precise content extraction AI. Your task is to clean the content of each node in a provided list while retaining the original order of nodes. Each node has the following properties:  
- `node_id` (unique identifier)  
- `content` (text to be cleaned)  

Your goal is to ensure that the content of each node contains only the most relevant text snippet that directly and comprehensively addresses the user's query, eliminating all irrelevant parts while meeting a minimum word requirement when possible.

**Processing Guidelines**:

1. **Identify Core Content**:  
   - For each node, focus exclusively on the text segment within the `content` that directly answers or is most relevant to the user's query.  
   - Extract and retain only the relevant portion, removing surrounding or introductory text unless essential for context.  

2. **Eliminate Non-Essential Content**:  
   - Remove metadata, headers, footers, and unrelated fields (e.g., "Company:", "Section:", "Title:", "URL:").  
   - Exclude disclaimers, participant lists, operator instructions, and administrative details, such as:  
     ```
     _This article is a transcript of this conference call produced for The Motley Fool..._
     ```  
   - Avoid including boilerplate language, acknowledgments, repetitive text, or operator sign-offs (e.g., "Thank you," "Operator signoff").  

3. **Preserve Original Meaning**:  
   - Retain the original wording of the relevant snippet to ensure accuracy.  
   - Make adjustments only if necessary for clarity.  

4. **Minimum Word Requirement**:  
   - Ensure the cleaned content for each node is **at least 100 words**.  
   - If the relevant content is shorter than 100 words, include the most contextually relevant surrounding details from the node's content to meet the word count.  
   - Avoid adding unrelated or tangential content to satisfy the requirement.  

5. **Query Alignment**:  
   - Ensure the cleaned content directly aligns with the user's query, providing a clear and actionable response.  
   - Do not include text that does not contribute to resolving the query.  

6. **Maintain Node Order**:  
   - Process each node individually and maintain the original order of nodes in the output.  
   - Do not remove or reorder any nodes.  

**Core Objective**:  
For each node, extract and return the cleaned `content` that is concise, relevant, and actionable. Ensure the cleaned content adheres to the query, removes irrelevant text, meets the **minimum word requirement of 100 words**, and preserves the original node order and identifiers in the output.  
"""


def clean_contents(query,re_ranked_nodes):
    
    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
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
                
    ],
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
        

    return json.loads(res)["nodes"]
    
