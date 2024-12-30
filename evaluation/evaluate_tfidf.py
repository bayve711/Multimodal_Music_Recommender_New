# evaluation/evaluate_tfidf.py

import pandas as pd
from scipy.sparse import csr_matrix
from retrieval.tfidf_retrieval import tfidf_retrieval
from retrieval.random_retrieval import random_retrieval
from metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr_metric
import ast
import os

def load_data_for_evaluation():
    """
    Loads and preprocesses data for evaluation.

    Returns:
    - catalog_df (pd.DataFrame)
    - tfidf_matrix (csr_matrix)
    - track_ids (list)
    - id_to_index (dict)
    """
    DATA_DIR = 'data/'  # Adjust path as necessary
    
    INFORMATION_FILE = os.path.join(DATA_DIR, 'id_information_mmsr.tsv')
    GENRES_FILE = os.path.join(DATA_DIR, 'id_genres_mmsr.tsv')
    LYRICS_TFIDF_FILE = os.path.join(DATA_DIR, 'id_lyrics_tf-idf_mmsr.tsv')
    
    # Load track information
    information_df = pd.read_csv(INFORMATION_FILE, sep='\t')
    
    # Load genres
    genres_df = pd.read_csv(GENRES_FILE, sep='\t')
    
    # Parse genres from string to list
    def parse_genres(genre_str):
        try:
            return ast.literal_eval(genre_str)
        except (ValueError, SyntaxError):
            return []
    
    genres_df['genre'] = genres_df['genre'].apply(parse_genres)
    
    # Merge information and genres to create catalog
    catalog_df = pd.merge(information_df, genres_df, on='id', how='left')
    
    # Handle missing genres by assigning an empty list
    catalog_df['genre'] = catalog_df['genre'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Extract top genre
    catalog_df['top_genre'] = catalog_df['genre'].apply(lambda x: x[0] if x else None)
    
    # Load TF-IDF vectors
    tfidf_df = pd.read_csv(LYRICS_TFIDF_FILE, sep='\t')
    if 'song' in tfidf_df.columns:
        tfidf_df = tfidf_df.rename(columns={'song': 'tfidf_song'})
    
    # Merge TF-IDF vectors
    catalog_df = pd.merge(catalog_df, tfidf_df, on='id', how='left')
    
    # Identify TF-IDF feature columns
    metadata_columns = ['artist', 'song', 'album_name', 'genre', 'top_genre']
    tfidf_feature_columns = [col for col in tfidf_df.columns if col != 'id' and col != 'tfidf_song']
    
    # Drop rows with missing TF-IDF features
    catalog_df = catalog_df.dropna(subset=tfidf_feature_columns).reset_index(drop=True)
    
    # Prepare TF-IDF matrix
    track_ids = catalog_df['id'].tolist()
    tfidf_features = catalog_df.drop(columns=['id', 'artist', 'song', 'album_name', 'genre', 'top_genre'])
    tfidf_matrix = csr_matrix(tfidf_features.values)
    
    # Create ID to index mapping
    id_to_index = {id_: index for index, id_ in enumerate(track_ids)}
    
    return catalog_df, tfidf_matrix, track_ids, id_to_index

def evaluate_retrieval_system(catalog_df, tfidf_matrix, track_ids, id_to_index, N=10):
    """
    Evaluates the TF-IDF retrieval system across all queries.

    Parameters:
    - catalog_df (pd.DataFrame): DataFrame containing all tracks with metadata.
    - tfidf_matrix (csr_matrix): Sparse matrix of TF-IDF vectors.
    - track_ids (list): List of track IDs.
    - id_to_index (dict): Mapping from track ID to matrix index.
    - N (int): Number of tracks to retrieve per query.

    Returns:
    - metrics_dict (dict): Dictionary containing average Precision@10, Recall@10, NDCG@10, and MRR.
    """
    precisions = []
    recalls = []
    ndcgs = []
    mrrs = []
    
    total_queries = len(catalog_df)
    processed_queries = 0
    
    for index, query_track in catalog_df.iterrows():
        query_id = query_track['id']
        query_genre = query_track['top_genre']
        
        # Skip if no top genre
        if not query_genre:
            continue
        
        # Perform TF-IDF retrieval
        retrieved_tracks = tfidf_retrieval(
            query_track_id=query_id,
            id_to_index=id_to_index,
            tfidf_matrix=tfidf_matrix,
            track_ids=track_ids,
            catalog_df=catalog_df,
            N=N
        )
        
        # If retrieval failed or no tracks retrieved, skip
        if retrieved_tracks.empty:
            continue
        
        retrieved_ids = retrieved_tracks['id'].tolist()
        
        # Define relevant tracks based on top genre
        relevant_ids = catalog_df[catalog_df['top_genre'] == query_genre]['id'].tolist()
        
        # Compute metrics
        p = precision_at_k(retrieved_ids, relevant_ids, k=N)
        r = recall_at_k(retrieved_ids, relevant_ids, k=N)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k=N)
        rr = mrr_metric(retrieved_ids, relevant_ids)
        
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(ndcg)
        mrrs.append(rr)
        
        processed_queries += 1
        if processed_queries % 500 == 0:
            print(f"Processed {processed_queries}/{total_queries} queries")
    
    # Aggregate metrics
    metrics_dict = {
        'Precision@10': sum(precisions) / len(precisions) if precisions else 0,
        'Recall@10': sum(recalls) / len(recalls) if recalls else 0,
        'NDCG@10': sum(ndcgs) / len(ndcgs) if ndcgs else 0,
        'MRR': sum(mrrs) / len(mrrs) if mrrs else 0
    }
    
    return metrics_dict

if __name__ == "__main__":
    # Load data
    catalog_df, tfidf_matrix, track_ids, id_to_index = load_data_for_evaluation()
    
    # Evaluate retrieval system
    metrics = evaluate_retrieval_system(catalog_df, tfidf_matrix, track_ids, id_to_index, N=10)
    
    print("\nTF-IDF Retrieval Metrics:")
    print(f"Precision@10: {metrics['Precision@10']:.4f}")
    print(f"Recall@10: {metrics['Recall@10']:.4f}")
    print(f"NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"MRR: {metrics['MRR']:.4f}")
