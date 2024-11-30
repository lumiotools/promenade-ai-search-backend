from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

pinecone = Pinecone()
pinecone_index_name = "nasdaq-companies"

pinecone_index = pinecone.Index(pinecone_index_name)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Instantiate VectorStoreIndex object from your vector_store object
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=OpenAIEmbedding(model="text-embedding-3-small"))

# Grab 5 search results
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

query_engine = vector_index.as_chat_engine(llm=OpenAI(
    model="gpt-4o-mini", system_prompt="You are a financial bot you have knowledge about the invester relations of any company listed on nasdaq, use this knowledge to answer users query, Answer users query in detailed by providing the data they need"))


def handle_chat(query):
    answer = query_engine.query(query)

    sources = []
    for source in answer.source_nodes:
        sources.append({
            "score": source.score,
            "url": source.node.extra_info["url"]
        })

    return answer.response, sources
