from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
from enum import Enum

load_dotenv()
client = OpenAI(
    # Defaults to os.environ.get("OPENAI_API_KEY")
)

company_data = pd.read_csv("data/companies.csv")
company_json = []
for i in range(len(company_data)):
    company_json.append(
        {"symbol": company_data.iloc[i]["Symbol"], "company_name": company_data.iloc[i]["Description"]})

system_prompt = f"""
You are a specialized AI tasked with extracting only the company name from a given text. Your response should focus solely on identifying and returning the name of the company mentioned in the input, ignoring any irrelevant details or additional information. You should interpret partial or colloquial mentions as the full corporate entity (e.g., "apple" should be understood as "Apple").

**Memory:**

{json.dumps(company_json)}


**Instructions:**

- Use the memory to access the company names with their corresponding symbols.
- Identify and extract the exact name of the company from the query.
- If the company is mentioned by an abbreviation or partial name (e.g., "apple", "tesla", "microsoft"), interpret it as the full formal name (e.g., "Apple", "Tesla", "Microsoft Corporation").
- If the text includes multiple companies, return all company names found.
- Do not provide any additional context or information other than the extracted company name(s).
- Ensure that the name is in the correct format as a formal company entity, including appropriate capitalization.
- If no company name is found, simply respond with "No company name found."

**Example Input:**
"Apple  reported a 6% increase in quarterly revenue."

**Example Output:**
"Apple"
"""

# Enum for QueryType
class QueryType(Enum):
    IR = "IR"
    SEC_FILINGS = "SEC_FILINGS"
    OTHERS = "OTHERS"

class CompanyDetails(BaseModel):
    company_name: str
    symbol: str
    
class ResponseFormat(BaseModel):
    companies: List[CompanyDetails]
    query_type: QueryType
    
    # Query Type only IR, SEC_FILINGS, OTHERS
    


def extract_query_details(query):
    chat_completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{
            "role": "system", "content": system_prompt}, {"role": "user", "content": query}],
        response_format=ResponseFormat
    )

    res = chat_completion.choices[0].message.parsed
    # company_name = res.company_name
    # symbol = res.symbol
    # query_type = res.query_type

    print(res)

    return "Apple Inc"
    # res=chat_completion.choices[0].message.content
    # print(f"gpt res {chat_completion.choices[0].message.content}")
    # if res =="No company name found.":
    #     return ""
    # else:
    #     return res
