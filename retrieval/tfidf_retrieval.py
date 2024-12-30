# retrieval/tfidf_retrieval.py

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.sparse import csr_matrix

def tfidf_retrieval(query_track_id, id_to_index, tfidf_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks most similar to the query track based on TF-IDF cosine similarity.

    Parameters:
    - query_track_id (str): The ID of the query track.
    - id_to_index (dict): Mapping from track ID to matrix index.
    - tfidf_matrix (csr_matrix): Sparse matrix of TF-IDF vectors.
    - track_ids (list): List of track IDs corresponding to tfidf_matrix rows.
    - catalog_df (pd.DataFrame): DataFrame containing track metadata.
    - N (int): Number of tracks to retrieve.

    Returns:
    - retrieved_tracks_df (pd.DataFrame): DataFrame of retrieved tracks sorted by similarity.
    """
    # Check if query_track_id exists
    if query_track_id not in id_to_index:
        print(f"Query track ID {query_track_id} not found.")
        return pd.DataFrame()
    
    # Get the index of the query track
    query_index = id_to_index[query_track_id]
    
    # Get the TF-IDF vector for the query track
    query_vector = tfidf_matrix[query_index]
    
    # Compute cosine similarity between the query and all tracks
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Exclude the query track itself by setting its similarity to -1
    similarities[query_index] = -1
    
    # Get the indices of the top N similar tracks
    top_indices = similarities.argsort()[-N:][::-1]
    
    # Retrieve the corresponding track IDs and similarity scores
    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]
    
    # Create a DataFrame for retrieved tracks
    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity': retrieved_scores
    })
    
    # Merge with catalog_df to get track details
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name']], on='id', how='left')
    
    return retrieved_tracks_df
