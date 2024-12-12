import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_news_articles(query):
    try:
        messages = [
            {
                "role": "system",
                "content": "Search only for news articles from Trusted Sources"
            },
            {
                "role": "user",
                "content": f"News Articles from Trusted Sources for '{query}'"
            }
        ]

        body = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": messages,
            "temperature": 0.5,
        }

        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
        }

        response = requests.post("https://api.perplexity.ai/chat/completions",
                         json=body,headers=headers)

        if response.status_code != 200:
            raise Exception(response.content)

        response = response.json()
        
        citations = response["citations"]
       
        return citations
    except Exception as e:
        print(e)