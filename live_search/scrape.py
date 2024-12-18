import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

def get_pages_content(urls):
    
    contents = []
        
    for url in urls:
        print(f"Fetching {url[:30]}...")
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
            if response.status_code != 200:
                print(f"Failed to fetch {url[:30]}...\n")
                continue
        except Exception as e:
            print(f"Failed to fetch {url[:30]}...\n")
            continue
        
        print(f"Got response from {url[:30]}...")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "Industry Report"
        html_content = soup.find('body')
        markdown_content = md(str(html_content))
        
        if markdown_content == "":
            print(f"Failed to convert {url[:30]}... to markdown\n")
            continue
        
        print(f"Converted {url[:30]}... to markdown\n")
        
        contents.append({
            "content":markdown_content,
            "source": url,
            "title": title
        })
        
    return contents
