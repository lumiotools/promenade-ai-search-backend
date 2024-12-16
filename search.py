import urllib.parse
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from dotenv import load_dotenv
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from typing import List
from typing import Optional
from extract_query_details import extract_query_details
from post_processors.filter import filter_nodes
from post_processors.re_rank_nodes import re_rank_nodes
from post_processors.clean_content import clean_contents
from live_news_search import handle_live_news_search
from live_sec_search import handle_live_sec_search
import json

load_dotenv()

pinecone = Pinecone()
pinecone_index_name = "nasdaq-companies"

pinecone_index = pinecone.Index(pinecone_index_name)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Instantiate VectorStoreIndex object from your vector_store object
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=OpenAIEmbedding(model="text-embedding-3-small"))


def handle_search(query):
  try:
    filters =extract_query_details(query)
    
    print(json.dumps(filters))

    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=30,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="symbol",
                    operator=FilterOperator.IN,
                    value=[company["symbol"] for company in filters["companies"]
                           ]),
            ]
        )
    )
    nodes = retriever.retrieve(query)
    
    result_nodes = []
    for node in nodes:      
        
      result_nodes.append({
        "content": node.get_content(),
        "node_id":node.node.node_id,
        "source":node.node.metadata["url"],
        "filed":node.node.metadata["filed"] if "filed" in node.node.metadata.keys() else None,
        "title":node.node.metadata["title"] if "title" in node.node.metadata.keys() else None,
        "doc_type":"SEC Filing" if "form_type" in node.node.metadata.keys() else 
        "IR Page" if "section_name" in node.node.metadata.keys() else "Earnings Call"
      })
      
    if filters["query_contains_sec_filings"]:
      result_nodes = [node for node in result_nodes if node["doc_type"] == "SEC Filing"]
      
    for node in result_nodes:
      print(node["node_id"])
      
    print()
      
    print("Filtering")
    filtered_nodes = filter_nodes(filters["companies"][0]["company_name"],query,result_nodes)
    
    for node in filtered_nodes:
      print(node["node_id"])
      for result_node in result_nodes:
        if node["node_id"] == result_node["node_id"]:
          node["content"] = result_node["content"]
          node["filed"] = result_node["filed"]
          node["title"] = result_node["title"]
          node["doc_type"] = result_node["doc_type"]
      
    print()
    
    print("Performing Live News Search...")
    live_search_nodes = handle_live_news_search(query)
    
    for node in live_search_nodes:
      print(node["node_id"])
      
    print()
    for node in live_search_nodes:      
      result_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "source":node["source"],
        "filed": None,
        "title": None,
        "doc_type":"Industry Report"
      })
      filtered_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "filed": None,
        "title": None,
        "doc_type":"Industry Report"
      })
      
    print("Performing Live SEC Filing Search...")
    live_search_nodes = []
    for company in filters["companies"]:
      live_search_nodes.extend(handle_live_sec_search(company["symbol"]))
    
    for node in live_search_nodes:
      print(node["node_id"])
      
    print()
    for node in live_search_nodes:      
      result_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "source":node["source"],
        "filed": node["filed"],
        "title": None,
        "doc_type":"SEC Filing"
      })
      filtered_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "filed": node["filed"],
        "title": None,
        "doc_type":"SEC Filing"
      })
      
    print("Cleaning")
    cleaned_nodes = []
    for node in filtered_nodes:
        try:
            cleaned_node = clean_contents(query, [node])[0]
            cleaned_node["content"] = cleaned_node["cleaned_content"]
            cleaned_node["filed"] = node["filed"]
            cleaned_node["title"] = node["title"]
            cleaned_node["doc_type"] = node["doc_type"]
            cleaned_nodes.append(cleaned_node)
            print(cleaned_node["node_id"])
        except Exception as e:
            continue

    print()
    
    print("Re-Ranking")
    re_ranked_nodes = re_rank_nodes(filters["companies"][0]["company_name"], query, cleaned_nodes)

    for node in re_ranked_nodes:
        print(node["node_id"])
        
    print()
    
    final_nodes = []
    
    valid_sources = []
    invalid_sources = []
    
    for node in re_ranked_nodes:
      for item in result_nodes:
        if item["node_id"] == node["node_id"]:
          node["source"] = item["source"]
          node["doc_type"] = item["doc_type"]
          break
        
      for item in cleaned_nodes:
        if item["node_id"] == node["node_id"]:
          node["highlight"] = item["highlight"]
          break
        
      if not "source" in node.keys():
        continue
      
      if any(final_node["node_id"] == node["node_id"] for final_node in final_nodes):
        continue
      
      final_nodes.append({
        "node_id":node["node_id"],
        "content": node["content"],
        "source":node["source"]+"#:~:text="+item["highlight"],
        "doc_type":node["doc_type"]
      })
      
      if not node["source"] in [source["url"] for source in valid_sources]:
        valid_sources.append({
          "doc_type":node["doc_type"],
          "url":node["source"]
        })
      
    for node in result_nodes:
      if not node["source"] in [source["url"] for source in valid_sources] and not node["source"] in [source["url"] for source in invalid_sources]:
        invalid_sources.append({
                                "doc_type":node["doc_type"],
                                "url":node["source"]
                               })
      
    # final_nodes = {
    #   "original":result_nodes,
    #   "filtered":filtered_nodes,
    #   "re_ranked":re_ranked_nodes,
    #   "cleaned":cleaned_nodes,
    #   "final":final_nodes
    # }   
      
    return final_nodes,valid_sources,invalid_sources
  except Exception as e:
    print("error",e)
    return [],[],[]