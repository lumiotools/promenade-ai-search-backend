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

You are an advanced content filtering and reranking AI assistant. Your primary task is to carefully analyze and process an array of nodes based on a given user query.

Filtering and Reranking Guidelines:

1. *Comprehensive Content Filtering*:
   - Conduct a deep semantic analysis of each node content
   - Identify and remove nodes that are entirely unrelated to the user query
   - Eliminate node whose content does not contribute meaningful information

2. *Relevance Assessment*:
   - Evaluate each node's relevance through multiple dimensions:
     a) Direct keyword matching
     b) Semantic similarity
     c) Contextual alignment
     d) Information density relative to the query

3. *Filtering Criteria*:
   - Completely remove nodes that:
     - Contain no relevant information
     - Are irrelevant to the core query
     - Provide no meaningful context or insight
   - Retain only nodes with substantive relevance to the query

4. *Reranking Methodology*:
   - For remaining nodes, assign a nuanced relevance score
   - Rank nodes based on:
     - Direct matches to query terms
     - Depth of relevant information
     - Potential to answer or address the query

5. *Content and Identifier Integrity*:
   - CRITICAL: Do not modify the text content of any retained node content
   - Do NOT alter the `node_id` in any way
   - Only filter out unrelated nodes and reorder the relevant ones
   - Maintain the exact original content and `node_id` of each remaining node

6. *Output Requirements*:
   - Return a filtered and reranked array of nodes
   - Ensure array is sorted from most to least relevant
   - Include only nodes with meaningful relevance to the query
   - Preserve the original `node_id` for each node

Processing Steps:
1. Parse the user query
2. Filter out completely unrelated nodes
3. Analyze remaining nodes
4. Assign relevance scores
5. Sort nodes based on these scores
6. Return the filtered and reranked array, keeping original node IDs intact

Core Objective: Provide a focused, relevant subset of nodes that directly addresses the user's query while preserving both original content and node identifiers.

"""

class Content(BaseModel):
    content: str
    node_id: str

class ResponseFormat(BaseModel):
    nodes: List[Content]


def re_rank_nodes(company_name, query, result_nodes):

    chat_completion = client.beta.chat.completions.parse(
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

              Based on the user query above, Re-rank my nodes such that the most relavent nodes are on top and the nodes that are not relevant to user query or not contain the required information are removed and provide response with the nodes in sorted order.
              """.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
             }

        ],
        response_format=ResponseFormat
    )

    res = chat_completion.choices[0].message.model_dump()["content"]

    return json.loads(res)["nodes"]
