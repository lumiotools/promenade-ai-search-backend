from live_search.sec import get_sec_links
from live_search.scrape import get_pages_content
from post_processors.clean_content import clean_contents
from uuid import uuid4


def handle_live_sec_search(symbol):
    try:
        sec_links = get_sec_links(symbol)

        nodes = []

        sec_contents = get_pages_content([sec_link["view"]["htmlLink"] for sec_link in sec_links])

        for i, sec_content in enumerate(sec_contents):
            nodes.append({
                "content": sec_content["content"],
                "node_id": str(uuid4()),
                "source": sec_content["source"],
                "filed": sec_links[i]["filed"],
            })
        
        return nodes
        
    except Exception as e:
        print(e)

        return []