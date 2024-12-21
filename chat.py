from openai import OpenAI
from dotenv import load_dotenv
from live_search.scrape import get_pages_content

load_dotenv()

openai = OpenAI()


def handle_chat(nodes, search_query, chat_history, message):
    try:
        # original_content = get_pages_content([node.source])[0]["content"]

        system_prompt = f"""
        You are provided with the following context information:

        *User Search Query:* {search_query}
        *Snippets of Original nodes Content that answers above search query:* {[node.content for node in nodes]}

        Based on the above information, provide a response in to the user's message.
        If users message is out of context, then say you dont have information on that.
        Only answer from the context provided above.
        Answer should be in bullet points.
        """

        messages = [
            {
                    "role": "system",
                    "content": system_prompt
            }
        ]

        messages.extend(chat_history)
        messages.extend([
            {
                "role": "user",
                "content": message
            }
        ])

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return "An error occured while processing the request."