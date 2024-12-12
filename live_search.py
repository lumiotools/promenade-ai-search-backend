from search.perplexity import get_news_articles
from search.scrape import get_pages_content
from post_processors.clean_content import clean_contents
import asyncio

def handle_live_search(query):
    try:
        news_articles = get_news_articles(query)

        nodes = []

        news_contents = asyncio.run( get_pages_content(news_articles))

        for i,news_content in enumerate(news_contents):
            nodes.append({
                "content":news_content["content"],
                "node_id":f"{i}"
            })

        cleaned_nodes = clean_contents(query,nodes)
        
        final_nodes = []
        valid_sources = []
        for i,cn in enumerate(cleaned_nodes):
            if cn["cleaned_content"] != "":
                final_nodes.append({
                "content":cn["cleaned_content"],
                "source":news_contents[i]["source"] +"#:~:text="+ cn["highlight"]
                })
                
        for i,cn in enumerate(cleaned_nodes):
            if cn["cleaned_content"] != "":
                valid_sources.append({
                    "doc_type":"Industry Report",
                    "url":news_contents[i]["source"]
                })
                

        return final_nodes, valid_sources,[]
    except Exception as e:
        print(e)
        
        return [],[],[]