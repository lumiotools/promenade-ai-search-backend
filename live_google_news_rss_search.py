import requests
import xml.etree.ElementTree as ET
from googlenewsdecoder import new_decoderv1
from live_search.scrape import get_pages_content
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor


def handle_live_google_news_rss_search(query):
    search_url = "https://news.google.com/rss/search?q=" + query +"&hl=en-US&gl=US&ceid=US:en"
    
    response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    
    if not response.status_code == 200:
        raise Exception("Failed to fetch data")
    
    root = ET.fromstring(response.content)
    
    encoded_articles = []
    
    for element in root[0]:
        if element.tag == "item":
            for child in element:
                if child.tag == "link":
                    encoded_articles.append(child.text)
                    
            if len(encoded_articles) == 20:
                break
            
    print("Decoding google news article urls...")
    
    with ThreadPoolExecutor() as executor:
        decoded_articles = list(executor.map(new_decoderv1, encoded_articles))
        
    articles = []
    
    for article in decoded_articles:
        try:
            articles.append(article["decoded_url"])
        except:
            pass
    
    print("Decoded google news article urls")
            
    nodes = []

    contents = get_pages_content([article for article in articles if (
        not "youtube.com" in article) and (not ".pdf" in article)])
    
    for i, content in enumerate(contents):
        nodes.append({
            "content": content["content"],
            "node_id": str(uuid4()),
            "source": content["source"],
            "title": content["title"],
        })
    
    return nodes