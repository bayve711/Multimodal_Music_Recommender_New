# evaluation/metrics.py

from sklearn.metrics import ndcg_score

def precision_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Computes Precision@k.

    Parameters:
    - retrieved_ids (list): List of retrieved track IDs.
    - relevant_ids (list): List of relevant track IDs.
    - k (int): The cutoff rank.

    Returns:
    - precision (float): Precision@k value.
    """
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_at_k) & set(relevant_ids)
    return len(relevant_retrieved) / k

def recall_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Computes Recall@k.

    Parameters:
    - retrieved_ids (list): List of retrieved track IDs.
    - relevant_ids (list): List of relevant track IDs.
    - k (int): The cutoff rank.

    Returns:
    - recall (float): Recall@k value.
    """
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_at_k) & set(relevant_ids)
    total_relevant = len(relevant_ids)
    return len(relevant_retrieved) / total_relevant if total_relevant > 0 else 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Computes NDCG@k.

    Parameters:
    - retrieved_ids (list): List of retrieved track IDs.
    - relevant_ids (list): List of relevant track IDs.
    - k (int): The cutoff rank.

    Returns:
    - ndcg (float): NDCG@k value.
    """
    retrieved_at_k = retrieved_ids[:k]
    relevance = [1 if track_id in relevant_ids else 0 for track_id in retrieved_at_k]
    ideal_relevance = sorted(relevance, reverse=True)
    return ndcg_score([relevance], [ideal_relevance], k=k)

def mrr_metric(retrieved_ids, relevant_ids):
    """
    Computes Mean Reciprocal Rank (MRR).

    Parameters:
    - retrieved_ids (list): List of retrieved track IDs.
    - relevant_ids (list): List of relevant track IDs.

    Returns:
    - reciprocal_rank (float): Reciprocal rank value.
    """
    for rank, track_id in enumerate(retrieved_ids, start=1):
        if track_id in relevant_ids:
            return 1.0 / rank
    return 0.0
