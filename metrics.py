from concurrent.futures import ThreadPoolExecutor, as_completed
from trulens.core import Feedback, TruSession
from trulens.providers.openai import OpenAI
from chat import handle_chat
from typing import List, Dict
import math

tru = TruSession()

# Initialize the LLM provider
openai_provider = OpenAI()

# Initialize the feedback function for context relevance
relevance_feedback = Feedback(openai_provider.qs_relevance).on_input_output()

def calculate_correctness_metrics(query: str, retrieved_nodes: List[Dict], k: int = 5):
    """
    Calculate retrieval metrics (Precision, Recall, F1 Score, MRR, and nDCG) dynamically using TruLens.

    Args:
        query (str): The input query.
        retrieved_nodes (List[Dict]): List of retrieved nodes, each as a dictionary with 'node_id' and 'content'.
        k (int): Number of top results to consider for metrics.

    Returns:
        Dict: A dictionary containing calculated metrics.
    """
    if not retrieved_nodes:
        # Handle cases where no nodes are retrieved
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_score_at_k": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
        }

    # Truncate to top-k
    top_k_nodes = retrieved_nodes[:k]
    
    # Assess relevance scores for each retrieved node using TruLens
    relevance_scores = [
        relevance_feedback(query, node["content"]) for node in top_k_nodes
    ]
    
    # Determine which nodes are "relevant" based on a threshold
    relevance_threshold = 0.5  # Define a threshold for relevance
    relevant_retrieved = [score for score in relevance_scores if score > relevance_threshold]
    
    # Calculate Precision@k
    precision_at_k = len(relevant_retrieved) / k if k > 0 else 0.0
    
    # Calculate Recall@k (requires assuming max relevant nodes as retrieved for simplicity)
    recall_at_k = len(relevant_retrieved) / len(top_k_nodes) if len(top_k_nodes) > 0 else 0.0
    
    # Calculate F1 Score@k
    if precision_at_k + recall_at_k > 0:
        f1_score_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    else:
        f1_score_at_k = 0.0
    
    # Calculate MRR (Mean Reciprocal Rank)
    reciprocal_ranks = [
        1 / (idx + 1) for idx, score in enumerate(relevance_scores) if score > relevance_threshold
    ]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # Calculate nDCG (Normalized Discounted Cumulative Gain)
    dcg = sum(
        score / math.log2(idx + 2) for idx, score in enumerate(relevance_scores)
    )
    idcg = sum(
        sorted(relevance_scores, reverse=True)[idx] / math.log2(idx + 2) for idx in range(len(relevance_scores))
    )
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Return calculated metrics
    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "f1_score_at_k": f1_score_at_k,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def evaluate_correctness_for_queries(queries: List[str], k: int = 5):
    """
    Evaluate multiple queries for correctness and compute aggregate metrics.

    Args:
        queries (List[str]): A list of queries to evaluate.
        k (int): Number of top results to consider for metrics.

    Returns:
        Dict: Aggregate metrics across all queries based on correctness.
    """
    aggregate_metrics = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "f1_score_at_k": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
    }
    num_queries = len(queries)
    
    # Run handle_chat in parallel
    with ThreadPoolExecutor() as executor:
        future_to_query = {executor.submit(handle_chat, query): query for query in queries}
        
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                response, _, _ = future.result()
                
                # Calculate metrics for the query
                metrics = calculate_correctness_metrics(query, response, k)
                
                print(f"Metrics for query '{query}'")
                print(metrics)
                
                # Aggregate metrics
                for key in aggregate_metrics:
                    aggregate_metrics[key] += metrics[key]
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
    
    # Average metrics over all queries
    for key in aggregate_metrics:
        aggregate_metrics[key] /= num_queries
    
    return aggregate_metrics


# Example queries
queries = [
    "How has Amazon's AWS performance been impacted by the growth of generative AI technologies?",
    "Has apple announced any share buyback programs recently?",
    "what are google's ai strategies?",
    "What is the succesion plan of berkshire hathaway?",
]

# Evaluate the queries in parallel and get aggregate metrics
final_metrics = evaluate_correctness_for_queries(queries, k=5)
print("Aggregate Metrics:")
print(final_metrics)
