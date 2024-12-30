import os
import requests
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class FileModel(BaseModel):
    name: str
    url: str


def get_industry_reports(query):
    try:
        search_url = f"https://customsearch.googleapis.com/customsearch/v1?cx={os.getenv('GOOGLE_SEARCH_ENGINE_ID')}&q={query}&fileType=pdf&nums=20&key={os.getenv('GOOGLE_SEARCH_API_KEY')}"

        response = requests.get(search_url)
        if response.status_code != 200:
            raise Exception("Failed to get search results")

        search_results = response.json()
        search_items = search_results.get("items", [])

        industry_reports = [
            FileModel(name=item["title"], url=item["link"]) for item in search_items]

        return industry_reports

    except Exception as e:
        print(e)

        return []
