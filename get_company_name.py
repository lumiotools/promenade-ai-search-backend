from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    # Defaults to os.environ.get("OPENAI_API_KEY")
)

system_prompt = """
You are a specialized AI tasked with extracting only the company name from a given text. Your response should focus solely on identifying and returning the name of the company mentioned in the input, ignoring any irrelevant details or additional information. You should interpret partial or colloquial mentions as the full corporate entity (e.g., "apple" should be understood as "Apple").

**Instructions:**

- Identify and extract the exact name of the company.
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

def handle_chat_for_company_name(query):
    chat_completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": system_prompt},{"role":"user","content":query}]
    )
    res=chat_completion.choices[0].message.content
    print(f"gpt res {chat_completion.choices[0].message.content}")
    if res =="No company name found.":
        return ""
    else:
        return res