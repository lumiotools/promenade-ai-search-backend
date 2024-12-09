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
You are an advanced content reranking AI assistant. Your primary task is to carefully analyze and prioritize the filtered nodes based on a given user query.

Reranking Guidelines:

1. *Relevance Assessment*:
   - Conduct a deep semantic analysis of each node
   - Evaluate nodes across multiple dimensions:
     a) Direct relevance to the query
     b) Depth of information
     c) Comprehensiveness
     d) Strategic value
     e) Contextual alignment

2. *Scoring Methodology*:
   - Assign a nuanced relevance score to each node based on:
     a) Exactness of information match
     b) Completeness of content
     c) Depth of insights provided
     d) Potential to answer key aspects of the query
     e) Uniqueness of information

3. *Prioritization Criteria*:
   - Rank nodes highest that:
     a) Directly answer the core query
     b) Provide comprehensive information
     c) Offer strategic or analytical insights
     d) Contain no external dependencies
     e) Are immediately comprehensible

4. *Contextual Reranking*:
   - Consider contextual nuances:
     a) Broader implications of the content
     b) Indirect but valuable information
     c) Potential interconnections between nodes
     d) Subtle but meaningful insights

5. *Output Requirements*:
   - Sort nodes from most to least relevant
   - Preserve original node identifiers
   - Maintain the integrity of node content
   - Provide a clear hierarchy of information

Processing Steps:
1. Parse the user query comprehensively
2. Analyze each node's potential to address the query
3. Assign detailed relevance scores
4. Create a prioritized ranking of nodes
5. Present nodes in order of their informational value

Core Objective: Transform the filtered content into a precisely ranked, contextually rich information set that maximally addresses the user's information needs.

Guiding Principles:
- Prioritize depth over breadth
- Focus on direct, actionable information
- Provide a nuanced, multi-dimensional ranking approach
"""


def re_rank_nodes(company_name, query, result_nodes):

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

              Based on the user query above, Re-rank my nodes such that the most relevant nodes are at the top and the least relevant nodes are at the bottom.
              """.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
             }
        ],
        response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "re_rank_nodes",
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
