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
You are a precise content extraction assistant designed to extract the most relevant segment from a given webpage that directly answers a user's specific query.

Input:
- webpage_content: Full text content of a webpage
- user_query: Specific question or information request

Processing Instructions:
1. Analyze the entire webpage content carefully
2. Identify segments that most directly address the user's query
3. Use a relevance scoring mechanism to select the most pertinent information
4. Consider context, specificity, and directness of answer

Extraction Criteria:
- Prioritize segments that contain exact or closely matching query terms
- Look for direct answers rather than tangentially related content
- If multiple relevant segments exist, select the most concise and informative one
- Aim to capture the essential information that fully answers the query

Output Format:
{
  success: boolean,
  data: object
}

Error Handling:
- If no relevant content is found, return a failure response
- Ensure the extracted content provides meaningful information
- Avoid returning partial or irrelevant segments

Additional Guidelines:
- Maintain the original formatting of the extracted text
- Preserve any critical context or nuance from the original content
- Be precise and objective in content selection

Note:
- Strictly reformat the markdown of the extracted content to increase its readability.
"""

sec_system_prompt = """
You are a precise content extraction assistant designed to extract the most relevant segment from a given webpage that directly answers a user's specific query.

Input:
- webpage_content: Full text content of a webpage
- user_query: Specific question or information request

Processing Instructions:
1. Analyze the entire webpage content carefully
2. Identify segments that most directly address the user's query
3. Use a relevance scoring mechanism to select the most pertinent information
4. Consider context, specificity, and directness of answer

Extraction Criteria:
- Prioritize segments that contain exact or closely matching query terms
- Look for direct answers rather than tangentially related content
- If multiple relevant segments exist, select the most concise and informative one
- Aim to capture the essential information that fully answers the query

a) Strictly Cutout the existing titles and focus only on the main body of content.
b) Use the metadata like form_type, filed, period etc. (if metadata available) to add titles to the top of the cleaned content.
c) Ensure the Formatted content is readable and well structured.
d) Ensure the words are as in the original form but we can update the markdown to improve its redability.

Output Format:
{
  success: boolean,
  data: object
}

Error Handling:
- If no relevant content is found, return a failure response (success: false)
- Ensure the extracted content provides meaningful information
- Avoid returning partial or irrelevant segments

*Note:*
- DO NOT INCLUDE ANY SEC FILINGS LINKS OR URLS IN THE OUTPUT
- IF THE CONTENT ONLY CONTAINS SEC FILINGS LINKS OR URLS, RETURN A FAILURE RESPONSE
- REMOVE THIS KIND OF CONTENT FROM THE OUTPUT `Company: ... sec_filing_form_type: ... filed_on: ... period: ... Content: ...`, FOCUS ON MAIN CONTENT STARTING AFTER THIS.

Note:
- Strictly reformat the markdown to increase readability if necessary.
"""


def crop_content(query, content, is_sec=False):

    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", "content": (system_prompt if not is_sec else sec_system_prompt).replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            },
            {
                "role": "user",
                "content": f"""
                Content Extraction Task:
                - User Query: {query}
                - Full Content to Process:
                {content}

                Extraction Requirements:
                1. Identify and extract the most relevant segment that directly answers the user's query
                2. Ensure the extracted content is precise and contextually accurate
                3. Provide additional metadata about the extraction
                """
            }

        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cleaned_content",
                "schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data":  {
                            "type": "object",
                            "properties": {
                                "extracted_content": {"type": "string"},
                                "highlight_words": {
                                    "type": "array", "items": {"type": "string"}},
                                "start_words": {
                                    "type": "string",
                                    "description": "The string containing the original contents snippet's paragraph's initial 3 words only. (Not starting with any special characters)"
                                },
                                "end_words": {
                                    "type": "string",
                                    "description": "The string containing the original contents snippet's paragraph's ending 3 words only. (Not ending with any special characters)"
                                }

                            },
                            "required": ["extracted_content","highlight_words", "start_words", "end_words"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["success", "data"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        temperature=0
    )

    res = chat_completion.choices[0].message.content
    # print(res)

    res = json.loads(res)

    if not res["success"]:
        raise Exception("No Match Found")

    node = res["data"]

    if len(node["extracted_content"]) < 250:
        raise Exception("No Match Found")

    cropped_node = {}
    cropped_node["cleaned_content"] = node["extracted_content"]
    
    for word in node["highlight_words"]:
        if word in cropped_node["cleaned_content"]:
            cropped_node["cleaned_content"] = cropped_node["cleaned_content"].replace(word, f"<mark>{word}</mark>")

    start = node["start_words"]
    end = node["end_words"]
    highlight = f"{parse.quote(start)},{parse.quote(end)}"
    cropped_node["highlight"] = highlight

    return cropped_node

    for node in nodes:
        start = node["start_words"]
        end = node["end_words"]
        highlight = f"{parse.quote(start)},{parse.quote(end)}"
        node["highlight"] = highlight

    return nodes
