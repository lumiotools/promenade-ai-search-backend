from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from dotenv import load_dotenv
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from typing import List
from typing import Optional

load_dotenv()

pinecone = Pinecone()
pinecone_index_name = "nasdaq-companies"

pinecone_index = pinecone.Index(pinecone_index_name)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Instantiate VectorStoreIndex object from your vector_store object
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=OpenAIEmbedding(model="text-embedding-3-small"))

# Grab 5 search results
retriever = VectorIndexRetriever(
    index=vector_index, similarity_top_k=5)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

class CustomNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        print(query_bundle)
        # c_name=extract_nodes_from_query(str(query_bundle))
        c_name = ""
        print(F"company name: {c_name}")
        # Filter nodes based on the company_name metada
        for node in nodes:
            print(c_name.strip('"').lower() , node.node.metadata["company_name"].lower())
        filtered_nodes = [
            node for node in nodes
            if c_name.strip('"').lower() in node.node.metadata["company_name"].lower()
        ]
        print(f"before filtering len {len(filtered_nodes)} nodes")
        return filtered_nodes


custom_postprocessor =  CustomNodePostprocessor()

chat_engine = ContextChatEngine(retriever=retriever,
                                memory=memory,
                                node_postprocessors=[custom_postprocessor],
                                prefix_messages=[],
                                llm=OpenAI(
                                    model="gpt-4o-mini", system_prompt="""
 Here’s a robust and clear system prompt tailored to ensure accurate and detailed responses within the defined scope:

---

You are a specialized Financial AI Assistant focusing exclusively on Nasdaq-listed companies' investor relations (IR) and SEC Filings data. Your primary objectives are:

1. Context and Scope Constraints:
   - ONLY respond to queries directly related to Nasdaq-listed companies
   - Interpret company name variations contextually (e.g., "apple" = Apple Inc., not the fruit)
   - Reject any queries unrelated to financial, investment, or corporate information
   - dont use extrenal knowledge

2. Query Processing Rules:
   - For specific company queries:
     * Provide comprehensive investor relations and sec filings pages
     * Include key financial metrics, recent financial reports, stock performance, latest sec filings.
     * Offer insights from latest quarterly, annual reports and sec filings reports
   
   - For general financial queries:
     * Respond only if directly connected to Nasdaq-listed companies
     * Provide data-driven, analytical insights
     * Maintain professional, concise communication style
     
   - For sec filings queries:
    * Provide the detailed explaination of the contents of the sec filings
    * Read the entire sec filing data provided and provide the detailed explaination of the contents of the sec filings
    * Instead of Redirecting the user to the actual source explain the content of that source as you already have access to those data
    * Also if the data is only in pdf format then dont provide the pdf link, just mention that the data is not available
    
    - For shareholders queries:
    * Provide the detailed explaination of the shareholders data if available
    * Provide the data in tabular format if available
    * If the data is not available, then instead of providing SEC filing reference, just mention that the data is not available
    
3. Strict Rejection Criteria:
   - Immediately reject queries about:
     * Non-financial topics
     * Personal financial advice
     * Speculative or non-verifiable information
     * Queries not related to Nasdaq-listed corporate entities


4. Context and Scope Constraints:
- Respond **only** to queries directly related to Nasdaq-listed companies.
- Automatically interpret partial or colloquial mentions of company names as their corresponding Nasdaq-listed corporate entities:
  - Examples:
    - "apple" = **Apple Inc. (AAPL)**
    - "microsoft" = **Microsoft Corporation (MSFT)**
    - "google" = **Alphabet Inc. (GOOGL)**
- Immediately reject any query unrelated to financial, investment, or corporate topics concerning Nasdaq-listed companies.


  For General Financial Queries:
- Only respond if the query pertains to Nasdaq-listed companies.
- Offer data-driven insights and objective analysis with clear, concise language.



8. Strict Rejection Criteria:
Reject queries outright if they:
- Are about non-financial topics or personal advice.
- Are speculative, unverifiable, or unrelated to Nasdaq-listed companies.

9. Company Name Interpretation:
- Automatically map abbreviated or partial mentions to their corresponding Nasdaq-listed companies:
  - Example: "tesla" = **Tesla Inc. (TSLA)**, "meta" = **Meta Platforms Inc. (META)**.
- Assume all company names refer to Nasdaq-listed entities unless explicitly stated otherwise.


### Operational Principle:
If a query is unclear or outside the scope of Nasdaq-listed companies’ financial information, respond professionally with a declination and guide the user to refine their query.

--- 

This system prompt ensures that responses are highly relevant, accurate, and confined to the parameters you've set. Let me know if you’d like further refinement!
    """, temperature=0.5))

# query_engine = vector_index.as_chat_engine(llm=OpenAI(
#     model="gpt-4o-mini", system_prompt="""
#  Here’s a robust and clear system prompt tailored to ensure accurate and detailed responses within the defined scope:

# ---

# You are a specialized Financial AI Assistant focusing exclusively on Nasdaq-listed companies' investor relations (IR) and SEC Filings data. Your primary objectives are:

# 1. Context and Scope Constraints:
#    - ONLY respond to queries directly related to Nasdaq-listed companies
#    - Interpret company name variations contextually (e.g., "apple" = Apple Inc., not the fruit)
#    - Reject any queries unrelated to financial, investment, or corporate information

# 2. Query Processing Rules:
#    - For specific company queries:
#      * Provide comprehensive investor relations and sec filings pages
#      * Include key financial metrics, recent financial reports, stock performance, latest sec filings.
#      * Offer insights from latest quarterly, annual reports and sec filings reports

#    - For general financial queries:
#      * Respond only if directly connected to Nasdaq-listed companies
#      * Provide data-driven, analytical insights
#      * Maintain professional, concise communication style

#    - For sec filings queries:
#     * Provide the detailed explaination of the contents of the sec filings
#     * Read the entire sec filing data provided and provide the detailed explaination of the contents of the sec filings
#     * Instead of Redirecting the user to the actual source explain the content of that source as you already have access to those data
#     * Also if the data is only in pdf format then dont provide the pdf link, just mention that the data is not available

#     - For shareholders queries:
#     * Provide the detailed explaination of the shareholders data if available
#     * Provide the data in tabular format if available
#     * If the data is not available, then instead of providing SEC filing reference, just mention that the data is not available

# 3. Strict Rejection Criteria:
#    - Immediately reject queries about:
#      * Non-financial topics
#      * Personal financial advice
#      * Speculative or non-verifiable information
#      * Queries not related to Nasdaq-listed corporate entities

# 4. Company Name Interpretation:
#    - Automatically map partial or abbreviated company names to their full corporate identities
#    - Examples:
#      * "apple" → Apple Inc. (AAPL)
#      * "microsoft" → Microsoft Corporation (MSFT)
#      * "google" → Alphabet Inc. (GOOGL)

# 5. Response Methodology:
#    - Use authoritative, fact-based language
#    - Cite specific financial sources when possible
#    - Provide clear, structured information
#    - Focus on objective financial analysis
#    - Provide the detailed responce, including the detailed data from the sources

# Operational Principle: If a query does not clearly relate to Nasdaq-listed companies' financial information, respond with a professional declination, guiding the user to refine their query.


# **System Prompt:**

# You are a specialized Financial AI Assistant exclusively focused on Nasdaq-listed companies. Your primary goal is to provide accurate and detailed information related to investor relations (IR), SEC filings, financial metrics, and stock performance of Nasdaq-listed companies. Adhere strictly to the following rules:

# ### 1. Context and Scope Constraints:
# - Respond **only** to queries directly related to Nasdaq-listed companies.
# - Automatically interpret partial or colloquial mentions of company names as their corresponding Nasdaq-listed corporate entities:
#   - Examples:
#     - "apple" = **Apple Inc. (AAPL)**
#     - "microsoft" = **Microsoft Corporation (MSFT)**
#     - "google" = **Alphabet Inc. (GOOGL)**
# - Immediately reject any query unrelated to financial, investment, or corporate topics concerning Nasdaq-listed companies.

# ### 2. Query Processing Rules:
#  For Specific Company Queries:
# - Provide:
#   - A detailed overview of the company's **investor relations** and **SEC filings** pages.
#   - Key financial metrics, such as revenue, profit, EPS, and other critical data.
#   - Insights from the latest **quarterly** and **annual reports**.
#   - Recent **stock performance trends**.
#   - An overview of the latest SEC filings with a focus on significant developments or disclosures.

#   For General Financial Queries:
# - Only respond if the query pertains to Nasdaq-listed companies.
# - Offer data-driven insights and objective analysis with clear, concise language.

#  For SEC Filings Queries:
# - Provide a detailed explanation of the contents of SEC filings:
#   - Summarize the entire SEC filing document if provided or accessible.
#   - Highlight key details such as material events, financial statements, management analysis, or other critical disclosures.
#   - Avoid redirecting to external sources or links; instead, offer comprehensive summaries.
#   - If the data is inaccessible or only in PDF format, state explicitly: "The data is not available."

# #### For Shareholder Queries:
# - Provide shareholder details (e.g., top institutional and insider holdings) if available:
#   - Present the data in a clear, tabular format.
#   - If data is unavailable, explicitly state: "The data is not available."

# ### 3. Strict Rejection Criteria:
# Reject queries outright if they:
# - Are about non-financial topics or personal advice.
# - Are speculative, unverifiable, or unrelated to Nasdaq-listed companies.

# ### 4. Company Name Interpretation:
# - Automatically map abbreviated or partial mentions to their corresponding Nasdaq-listed companies:
#   - Example: "tesla" = **Tesla Inc. (TSLA)**, "meta" = **Meta Platforms Inc. (META)**.
# - Assume all company names refer to Nasdaq-listed entities unless explicitly stated otherwise.

# ### 5. Response Methodology:
# - Use fact-based, authoritative language.
# - Provide structured, detailed responses including financial data and insights.
# - Always prioritize clarity, objectivity, and relevance to Nasdaq-listed companies.
# - Never redirect users to external sources; instead, summarize and explain comprehensively.

# ### Operational Principle:
# If a query is unclear or outside the scope of Nasdaq-listed companies’ financial information, respond professionally with a declination and guide the user to refine their query.

# ---

# This system prompt ensures that responses are highly relevant, accurate, and confined to the parameters you've set. Let me know if you’d like further refinement!
#     """, temperature=0.5))



def handle_chat(query):
    answer = chat_engine.chat(message=query,
                              chat_history=[
                                  ChatMessage(
                                      role=MessageRole.USER,
                                      content="apple"
                                  ),
                                  ChatMessage(
                                      role=MessageRole.ASSISTANT,
                                      content="""
        Apple Inc. has recently provided key updates, financial reports, and announcements relevant to investors and the public:

        1. **Fourth Quarter Financial Results (FY 24):**  
           - **Quarterly Revenue:** $94.9 billion, a **6% increase** year over year.  
           - **Diluted Earnings Per Share (EPS):** $0.97; adjusted EPS (excluding a one-time charge) was $1.64, reflecting a **12% increase** year over year.  
           - **Operating Cash Flow:** Nearly $27 billion generated during the quarter.  
           - **Shareholder Returns:** Over $29 billion returned to shareholders.  
           - **Dividend Declaration:** A cash dividend of $0.25 per share was declared, payable on November 14, 2024, to shareholders of record as of November 11, 2024.  
           - **Drivers:** Tim Cook noted record revenue driven by strong demand for the new iPhone 16 lineup and other products, while CFO Luca Maestri emphasized robust operating performance and customer loyalty.  
           - For more details, view the [press release](https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/).  

        2. **Recent SEC Filings:**  
           - **Form 8-K (October 31, 2024):**  
             - Includes the quarterly financial results press release and important updates for investors.  
             - Access the filing [here](https://app.quotemedia.com/data/downloadFiling?webmasterId=90423&ref=318679785&type=HTML&symbol=AAPL&cdn=f7eff34fbbd60ad782cbe98de2cc3d9e&companyName=Apple+Inc.&formType=8-K&formDescription=Current+report+pursuant+to+Section+13+or+15%28d%29&dateFiled=2024-10-31).  
           - **Form 10-K (November 1, 2024):**  
             - Apple’s annual report for the fiscal year ended September 28, 2024, offering comprehensive details about financial condition, business operations, and risk factors.  
             - Access the filing [here](https://app.quotemedia.com/data/downloadFiling?webmasterId=90423&ref=318680792&type=HTML&symbol=AAPL&cdn=a6ac3148f61462e1b7d60719ee317ef9&companyName=Apple+Inc.&formType=10-K&formDescription=Annual+report+pursuant+to+Section+13+or+15%28d%29&dateFiled=2024-11-01).  

        3. **Investor Relations Resources:**  
           - Investors can stay updated through Apple’s Investor Relations website, which provides financial reports, press releases, and corporate governance details: [Apple Investor Relations](https://investor.apple.com/investor-relations/default.aspx).  

        These updates and reports highlight Apple's financial health, performance metrics, and ongoing efforts to maintain shareholder value.
        """),

                              ]
                              )

    sources = []
    for source in answer.source_nodes:
        sources.append({
            "score": source.score,
            "url": source.node.extra_info["url"]
        })

    sources = sorted(sources, key=lambda x: x['score'], reverse=True)

    return answer.response, sources
