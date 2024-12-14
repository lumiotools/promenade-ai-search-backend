from search.perplexity import get_news_articles
from search.scrape import get_pages_content
from post_processors.clean_content import clean_contents
from uuid import uuid4


def handle_live_news_search(query):
    try:
        news_articles = get_news_articles(query)

        nodes = []

        news_contents = get_pages_content([news_article for news_article in news_articles if (
            not "youtube.com" in news_article) and (not ".pdf" in news_article)])

        for i, news_content in enumerate(news_contents):
            nodes.append({
                "content": news_content["content"],
                "node_id": str(uuid4()),
                "source": news_content["source"]
            })
        
        return nodes
        
    except Exception as e:
        print(e)

        return []

