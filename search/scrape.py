import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

def get_pages_content(urls):
    
    contents = []
        
    for url in urls:
        print(f"Fetching {url[:30]}...")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code != 200:
            print(f"Failed to fetch {url[:30]}...\n")
            continue
        
        print(f"Got response from {url[:30]}...")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        html_content = soup.find('body')
        markdown_content = md(str(html_content))
        
        print(f"Converted {url[:30]}... to markdown\n")
        
        contents.append({
            "content":markdown_content,
            "source": url
        })
        
    return contents
