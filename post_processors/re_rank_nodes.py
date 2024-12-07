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

1. *Nuanced Content Evaluation*:
   - Conduct a deep semantic analysis of each node content
   - Be judicious in identifying nodes to filter
   - Focus on preserving potentially valuable information
   - Remove only nodes that are definitively irrelevant or non-contributory

2. *Flexible Relevance Assessment*:
   - Evaluate each node's relevance through multiple dimensions:
     a) Partial semantic similarity
     b) Contextual alignment
     c) Potential informational value
     d) Indirect relevance to the query

3. *Minimal Filtering Criteria*:
    - Remove nodes ONLY if they:
    - Are completely unrelated to the core query
    - Contain zero meaningful information
    - Are clearly spam or nonsensical content
    - Retain nodes with even partial or tangential relevance
    - Preserve nodes that might provide supplementary or contextual information
    - Don't check strictly

4. *Intelligent Reranking Methodology*:
   - Assign a nuanced relevance score to each node
   - Prioritize nodes based on:
     - Direct relevance to query
     - Depth of relevant information
     - Potential to provide insights
     - Contextual usefulness

5. *Content and Identifier Integrity*:
   - CRITICAL: Do not modify the text content of any retained node
   - Preserve the `node_id` exactly as it was originally
   - Minimize node removal
   - Maintain the original content and identifiers

6. *Output Requirements*:
   - Return a carefully curated array of nodes
   - Sort nodes from most to least relevant
   - Include a broad range of potentially useful nodes
   - Preserve original `node_id` for each node
   - Ensure that the response is at least 5 to 6 lines long.

Processing Steps:
1. Parse the user query with an open, inclusive approach
2. Lightly filter out only the most irrelevant nodes
3. Analyze remaining nodes comprehensively
4. Assign nuanced relevance scores
5. Sort nodes based on relevance
6. Return the reranked array, keeping most original nodes intact

Core Objective: Provide a thoughtful, comprehensive subset of nodes that addresses the user's query while preserving context, depth, and original node characteristics.

Additional Guidance:
- When in doubt, retain the node
- Prioritize information preservation
- Consider potential indirect value of nodes
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
              Please provide data only related to this company and ignore all other nodes.
              This is the user query: {query}
              These are my nodes:
              {json.dumps(result_nodes)}

              Based on the user query above, Re-rank my nodes such that only the relevant nodes for the specified company are included, and remove all other irrelevant nodes.
              """.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
             }
        ],
        response_format=ResponseFormat
    )

    res = chat_completion.choices[0].message.model_dump()["content"]

    return json.loads(res)["nodes"]
