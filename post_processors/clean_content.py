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

You are a highly precise content extraction AI. Your goal is to identify and extract the smallest, most relevant text snippet that directly and comprehensively addresses the user's query while meeting the minimum word requirement.

**Extraction Guidelines**:
1. **Identify Core Content**:
   - Locate the exact text segment that most directly answers or is relevant to the user's query.
   - Ensure the extracted text is clear, actionable, and directly addresses the query.

2. **Eliminate Non-Essential Content**:
   - Remove metadata, headers, footers, and additional information (e.g., Company:, Section:, Title:, URL:).
   - Exclude introductory or surrounding text that does not contribute to the answer.
   - Explicitly remove disclaimers or similar text, such as:
     ```
     _This article is a transcript of this conference call produced for The Motley Fool. While we strive for our Foolish Best, there may be errors, omissions, or inaccuracies in this transcript._
     ```
   - Exclude boilerplate language, operator instructions, and repetitive acknowledgments (e.g., "Thank you," "Operator signoff").

3. **Preserve Original Meaning**:
   - Retain the original wording of the relevant text to ensure accuracy.
   - Do not paraphrase or summarize unless necessary for clarity.

4. **Meet Minimum Word Requirement**:
   - Ensure the extracted snippet is at least 100 words if possible, without including irrelevant text.
   - If the core answer is less than 100 words, include the most contextually relevant surrounding details to meet the word count.

5. **Fallback for No Exact Match**:
   - If an exact answer is not found, provide the most relevant segment related to the query.
   - Ensure the fallback snippet adds value and context relevant to the query.

**Content Filtering and Exclusion Rules**:
1. **Exclude Irrelevant Content**:
   - Remove headers, footnotes, legal disclaimers, or unrelated comments.
   - Explicitly filter out disclaimer text such as:
     ```
     _This article is a transcript of this conference call produced for The Motley Fool. While we strive for our Foolish Best, there may be errors, omissions, or inaccuracies in this transcript._
     ```
   - Avoid including text like participant lists, metadata, or URL links unless they are essential to the response.

2. **Prioritize Actionable Insights**:
   - Focus on text that provides clear, actionable information or directly resolves the query.
   - Avoid excessive detail or tangential discussions.

3. **Preserve Conciseness**:
   - Extract the shortest possible snippet that meets the user's need and the minimum word requirement.
   - Avoid redundancy or verbose segments.

**Core Objective**:
Surgically extract a concise, relevant, and comprehensive text snippet that answers the user’s query while ensuring the response is at least 100 words long and free from any extraneous, repetitive, or disclaimer text.


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
              
              For my below nodes contents, reformat their content by applying and fixing the markdown and provide me the cleaned content.
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
    
