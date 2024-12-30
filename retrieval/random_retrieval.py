# retrieval/random_retrieval.py

import pandas as pd
import random

def random_retrieval(catalog_df, N=10):
    """
    Retrieves N random tracks from the catalog.

    Parameters:
    - catalog_df (pd.DataFrame): DataFrame containing all tracks.
    - N (int): Number of tracks to retrieve.

    Returns:
    - retrieved_tracks_df (pd.DataFrame): DataFrame of randomly retrieved tracks with a 'similarity' column.
    """
    if N > len(catalog_df):
        N = len(catalog_df)
    
    # Sample N tracks
    retrieved_tracks_df = catalog_df.sample(n=N, random_state=42).reset_index(drop=True)
    
    # Select necessary columns
    retrieved_tracks_df = retrieved_tracks_df[['id', 'artist', 'song', 'album_name']].copy()
    
    # Add a 'similarity' column with a placeholder value
    retrieved_tracks_df['similarity'] = 0.0
    
    return retrieved_tracks_df
