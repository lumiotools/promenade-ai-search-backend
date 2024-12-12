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
You are an advanced content filtering AI assistant. Your task is to rigorously remove irrelevant, administrative, and metadata-heavy nodes from the dataset while retaining only meaningful, actionable, and contextually relevant information.

**Processing Context**:  
You will receive an array of nodes. Each node is treated as an individual entity and contains the following properties:  
- `content` (text of the node)  
- `node_id` (unique identifier for the node)  

Your job is to filter nodes based on their `content` while ensuring each node is processed independently.  

**Filtering Guidelines**:  

1. **Remove Administrative Content**:  
   - Eliminate nodes with:  
     a) Generic acknowledgments or pleasantries (e.g., "Thank you," "Appreciate it").  
     b) Operator instructions or metadata (e.g., "Duration: X minutes").  
     c) Participant lists (e.g., "Call participants:\n\n**Drew LaBenne** -- _Chief Financial Officer_").  
     d) Closing remarks or administrative summaries without actionable content (e.g., "Ladies and gentlemen, we have reached the end of our session").  
     e) Any combination of the above.  

2. **Exclude Non-Substantive Nodes**:  
   - Remove nodes that:  
     a) Contain only headers, disclaimers, or links (e.g., "More AMAL analysis").  
     b) Summarize participation or metadata without adding meaningful content.  
     c) Include repetitive acknowledgments or sign-offs (e.g., "Thanks," "Operator signoff").  

3. **Preserve Meaningful Content**:  
   - Retain only nodes which follow the user's query and are with:  
     a) Substantive insights, such as financial data, strategic discussions, or detailed answers.  
     b) Actionable information relevant to the discussion or user query.  

4. **Strict Exclusion of Metadata**:  
   - Explicitly filter out:  
     a) "Call participants" sections or speaker lists unless accompanied by substantive information.  
     b) Duration markers, timestamps, or session management details.  
     c) Disclaimers or boilerplate text (e.g., "This article is a transcript of this conference call...").  

5. **Handle SEC Filings**:  
   - Preserve all nodes containing SEC filings if they include detailed, substantive information relevant to the query.  
   - Filter out SEC filings that contain only a title or no actionable content.  

**Processing Steps**:  
1. Treat each node as an independent unit for evaluation.  
2. Identify nodes containing administrative, metadata-heavy, or non-substantive content.  
3. Aggressively filter out irrelevant content using the guidelines above.  
4. Retain only nodes with actionable, substantive information.  
5. For SEC filings, check if the content is substantive:
   - If it contains details relevant to the query, retain it.  
   - If it only contains a title or no meaningful content, filter it out.  
6. Ensure the output contains unique nodes based on their `node_id` and maintain correct node mappings.  

**Examples of Content to Remove**:  
- Nodes like:  
  ```
  Thank you.\n\nOperator\n\nDuration: 51 minutes\n\n## Call participants:\n\nDrew LaBenne -- Chief Financial Officer\n\nKeith Mestrich -- President and Chief Executive Officer\n\nSteven Alexopoulos -- J.P. Morgan -- Analyst\n...
  ```  

**Core Objective**:  
Produce a clean dataset containing only relevant, meaningful, and actionable content by strictly removing administrative, repetitive, and metadata-heavy nodes. Ensure the output is consistent, concise, and maintains correct node identifiers. Retain SEC filings only if they include substantive information; filter out filings that are merely titles or lack meaningful content.
"""

class Content(BaseModel):
    content: str
    node_id: str

class ResponseFormat(BaseModel):
    nodes: List[Content]


def filter_nodes(company_name, query, result_nodes):

    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system", "content": system_prompt.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        },
            {"role": "user",
             "content": f"""
              This is the company name: {company_name}
              This is the user query: {query}

              These are my nodes:
              {json.dumps(result_nodes)}
              
              Core Objective:
              Deliver only the nodes that are the most likely to answer the user's query or provide meaningful context about the specified company.
              """.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
             }
        ],
        response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "filter_nodes",
            "schema": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "node_id": {"type": "string"}
                            },
                            "required": ["content", "node_id"],
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
