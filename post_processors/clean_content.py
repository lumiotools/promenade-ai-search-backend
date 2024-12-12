from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
from enum import Enum
from urllib import parse

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
   - Remove unrelated sentences, headers, footers, and metadata (e.g., "Company:", "Section:", "Title:", "URL:", "Filed On:", "SEC Filing Form Type:", "Period:", or "URL:").  
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

7. **Special Handling only for SEC Filings**:  
   - If a node contains information about SEC filings, ensure the content follows all the above rules and additionally:  
     a) Extract and display the following details in proper markdown format, if available:  
        - **Form Type**: Mention the SEC form type (e.g., 10-K, 8-K, etc.).  
        - **Filing Date**: Include the date the filing was made.  
        - **Important Data**: Extract and include relevant sections, such as financial highlights, summaries, or other key details.  
     b) Format the SEC filing content as follows:  
        ```
        ### SEC Filing Details:
        - **Form Type**: [Form Type Here]  
        - **Filing Date**: [Filing Date Here]  
        - **Key Information**:  
          [Important data or summary of the filing, presented as bullet points or clean paragraphs.]  
        ```

---

**For Start and End words (not to be part of cleaned_content, its part of startWords and endWords)**:
    1) For the startWords string give the string containing the initial words of our snippet from original content.
    2) For the endWords string give the string containing the ending words of our snippet from original content.
    3) The words should be as in the original content and maximum 3 words are allowed.
    4) The startWords and endWords should be from the original content from where we have captured the snippet.
    5) The startWords and endWords should start and end on plain text only. (as it will be used to highlight content on original source page)

**Core Objective**:  
For each node, return the cleaned `content` that directly addresses the query in the exact original tone, structure, and sentence order. The cleaned content must meet the **minimum word requirement of 100 words** without summarizing or interpreting the text. 
Try to keep the cleaned content short and focused on the user query. For SEC filings, ensure proper markdown formatting and include only substantive information. The output must be clear, concise, and safe for all audiences.

**Note**:  
Remove any words that can trigger content filtering or are not safe for work. Ensure that the content is safe for all audiences.
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
              {json.dumps(re_ranked_nodes).replace("UNITED STATES SECURITIES AND EXCHANGE COMMISSION","").replace("Washington, D.C.","")}
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
                                "node_id": {"type": "string"},
                                "startWords": {"type":"string"},
                                "endWords": {"type":"string"}
                            },
                            "required": ["cleaned_content", "node_id","startWords","endWords"],
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
        
    nodes =  json.loads(res)["nodes"]
    
    for node in nodes:
        start = node["startWords"]
        end = node["endWords"]
        highlight = f"{parse.quote(start)},{parse.quote(end)}"
        node["highlight"] = highlight

    return nodes
    
