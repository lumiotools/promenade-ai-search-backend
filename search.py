from fastapi import UploadFile
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from typing import List
from extract_query_details import extract_query_details
from post_processors.filter import filter_nodes
from post_processors.re_rank_nodes import re_rank_nodes
from post_processors.crop_content import crop_content
from live_industry_report_search import handle_live_industry_reports_search
from live_google_news_rss_search import handle_live_google_news_rss_search
from live_sec_search import handle_live_sec_search
from live_document_search import handle_live_document_search
import json
import concurrent.futures
from pydantic import BaseModel
    
load_dotenv()

pinecone = Pinecone()
pinecone_index_name = "nasdaq-companies"

pinecone_index = pinecone.Index(pinecone_index_name)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Instantiate VectorStoreIndex object from your vector_store object
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=OpenAIEmbedding(model="text-embedding-3-small"))

class FileModel(BaseModel):
    name: str
    url: str
    
def handle_search(query, files: List[FileModel]):
  try:
    filters =extract_query_details(query)
    
    print(json.dumps(filters))
    
    if len(filters["companies"]) > 0:
      retriever = VectorIndexRetriever(
          index=vector_index,
          similarity_top_k=40,
          filters=MetadataFilters(
              filters= [ 
                  MetadataFilter(
                      key="symbol",
                      operator=FilterOperator.IN,
                      value=[company["symbol"] for company in filters["companies"]
                             ]),
              ]
          )
      )
      
    else:
      retriever = VectorIndexRetriever(
          index=vector_index,
          similarity_top_k=40
      )
    
    nodes = retriever.retrieve(query)
    
    result_nodes = []
    for node in nodes:      
        
      result_nodes.append({
        "content": node.get_content(),
        "node_id":node.node.node_id,
        "source":node.node.metadata["url"],
        "company_name":node.node.metadata["company_name"] if "company_name" in node.node.metadata.keys() else None,
        "form_type":node.node.metadata["form_type"] if "form_type" in node.node.metadata.keys() else None,
        "filed":node.node.metadata["filed"] if "filed" in node.node.metadata.keys() else None,
        "title":node.node.metadata["title"] if "title" in node.node.metadata.keys() else None,
        "doc_type":"SEC Filing" if "form_type" in node.node.metadata.keys() else 
        "IR Page" if "section_name" in node.node.metadata.keys() else "Earnings Call"
      })
      
      for i,node in enumerate(result_nodes):
        if node["doc_type"] == "SEC Filing":
          result_nodes[i]["title"] = result_nodes[i]["company_name"] + ". Form " + result_nodes[i]["form_type"] + " - " + result_nodes[i]["filed"] if result_nodes[i]["form_type"] else result_nodes[i]["company_name"] + ". Form SEC Filing"
    
    result_nodes_sec = [node for node in result_nodes if node["doc_type"] == "SEC Filing"]
      
    for node in result_nodes:
      print(node["node_id"])
      
    print()
      
    print("Filtering")
    filtered_nodes = filter_nodes(filters["companies"][0]["company_name"] if len(filters["companies"])>0 else None,query,result_nodes)
    
    for node in filtered_nodes:
      print(node["node_id"])
      for result_node in result_nodes:
        if node["node_id"] == result_node["node_id"]:
          node["content"] = result_node["content"]
          node["filed"] = result_node["filed"]
          node["title"] = result_node["title"]
          node["doc_type"] = result_node["doc_type"]
      
    print()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
      print("Performing Live Uploaded Documents Search...")
      future_document_search = executor.submit(handle_live_document_search, files)
      print("Performing Live Industry Reports Search...")
      future_industry_reports_search = executor.submit(handle_live_industry_reports_search, query)
      print("Performing Live Google News RSS Search...")
      future_google_news_rss_search = executor.submit(lambda: handle_live_google_news_rss_search(query))
      print("Performing Live SEC Filing Search...")
      future_sec_search = executor.submit(lambda: [handle_live_sec_search(company["symbol"]) for company in filters["companies"]])

      document_nodes = future_document_search.result()
      live_industry_reports_search_nodes = future_industry_reports_search.result()
      live_google_news_rss_nodes = future_google_news_rss_search.result()
      live_sec_nodes = [node for sublist in future_sec_search.result() for node in sublist]
    
    # print("Performing Live Uploaded Documents Search...")
    # document_nodes = handle_live_document_search(files)
    
    print("Live Uploaded Documents Search Results")
    for node in document_nodes:
      print(node["node_id"])
      
    print()
    for node in document_nodes:      
      result_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "source":node["source"],
        "filed": None,
        "title": node["title"],
        "doc_type":"Uploaded Document"
      })
      filtered_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "filed": None,
        "title": node["title"],
        "doc_type":"Uploaded Document"
      })
    
    # print("Performing Live News Search...")
    # live_search_nodes = handle_live_news_search(query)
    
    print("Live Industry Reports Search Results")
    for node in live_industry_reports_search_nodes:
      print(node["node_id"])
      
    print()
    for node in live_industry_reports_search_nodes:      
      result_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "source":node["source"],
        "filed": None,
        "title": node["title"],
        "doc_type":"Industry Report"
      })
      filtered_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "filed": None,
        "title": node["title"],
        "doc_type":"Industry Report"
      })
      
    print("Live Google News RSS Search Results")
    for node in live_google_news_rss_nodes:
      print(node["node_id"])
      
    print()
    for node in live_google_news_rss_nodes:
      result_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "source":node["source"],
        "filed": None,
        "title": node["title"],
        "doc_type":"Press"
      })
      filtered_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "filed": None,
        "title": node["title"],
        "doc_type":"Press"
      })
      
    # if len(result_nodes_sec) <= 5:
    # print("Performing Live SEC Filing Search...")
    # live_search_nodes = []
    
    # for company in filters["companies"]:
    #   live_search_nodes.extend(handle_live_sec_search(company["symbol"]))
    
    print("Live SEC Filing Search Results")
    for node in live_sec_nodes:
      print(node["node_id"])
      
    print()
    
    for node in live_sec_nodes:      
      result_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "source":node["source"],
        "filed": node["filed"],
        "form_type": node["form_type"],
        "title": node["title"],
        "doc_type":"SEC Filing"
      })
      filtered_nodes.append({
        "content": node["content"],
        "node_id":node["node_id"],
        "filed": node["filed"],
        "form_type": node["form_type"],
        "title": node["title"],
        "doc_type":"SEC Filing"
      })

    print("Extracting Content")
    cropped_nodes = []

    def process_node(node):
      try:
        cropped_node = crop_content(query, node["content"], node["doc_type"] == "SEC Filing")
        cropped_node["node_id"] = node["node_id"]
        cropped_node["content"] = cropped_node["cleaned_content"]
        cropped_node["highlight_words"] = cropped_node["highlight_words"]
        cropped_node["filed"] = node["filed"]
        cropped_node["title"] = node["title"]
        cropped_node["doc_type"] = node["doc_type"]
        print(cropped_node["node_id"])
        return cropped_node
      except Exception as e:
        print(e)
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
      future_to_node = {executor.submit(process_node, node): node for node in filtered_nodes}
      for future in concurrent.futures.as_completed(future_to_node):
        cropped_node = future.result()
        if cropped_node:
          cropped_nodes.append(cropped_node)
    
    print()
          
    # print("Extracting Content")
    
    # cropped_nodes = []
    # for node in filtered_nodes:
    #     try:
    #         cropped_node = crop_content(query, node["content"], node["doc_type"] == "SEC Filing")
    #         cropped_node["node_id"] = node["node_id"]
    #         cropped_node["content"] = cropped_node["cleaned_content"]
    #         cropped_node["filed"] = node["filed"]
    #         cropped_node["title"] = node["title"]
    #         cropped_node["doc_type"] = node["doc_type"]
    #         cropped_nodes.append(cropped_node)
    #         print(cropped_node["node_id"])
    #     except Exception as e:
    #         print(e)
    #         continue

    # print()
    
    print("Re-Ranking")
    re_ranked_nodes = re_rank_nodes(filters["companies"][0]["company_name"] if len(filters["companies"]) > 0 else None, query, cropped_nodes)

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
          node["title"] = item["title"]
          node["doc_type"] = item["doc_type"]
          break
        
      for item in cropped_nodes:
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
        "highlight_words": node["highlight_words"],
        "title":node["title"],
        "source":node["source"]+"#:~:text="+item["highlight"],
        "doc_type":node["doc_type"]
      })
      
      if not node["source"] in [source["url"] for source in valid_sources]:
        valid_sources.append({
          "doc_type":node["doc_type"],
          "title":node["title"],
          "url":node["source"]
        })
      
    for node in result_nodes:
      if not node["source"] in [source["url"] for source in valid_sources] and not node["source"] in [source["url"] for source in invalid_sources]:
        invalid_sources.append({
                                "doc_type":node["doc_type"],
                                "title":node["title"],
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