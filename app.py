# app.py

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from scipy.sparse import csr_matrix
from retrieval.random_retrieval import random_retrieval
from retrieval.tfidf_retrieval import tfidf_retrieval
import ast
import os

app = Flask(__name__)

# Load and prepare data on startup
def load_data():
    """
    Loads and preprocesses the data required for the application.
    
    Returns:
    - catalog_df (pd.DataFrame): DataFrame containing all tracks with metadata and TF-IDF vectors.
    - tfidf_matrix (csr_matrix): Sparse matrix of TF-IDF vectors.
    - track_ids (list): List of track IDs.
    - id_to_index (dict): Mapping from track ID to matrix index.
    """
    DATA_DIR = 'data/'  # Ensure this path is correct
    
    INFORMATION_FILE = os.path.join(DATA_DIR, 'id_information_mmsr.tsv')
    GENRES_FILE = os.path.join(DATA_DIR, 'id_genres_mmsr.tsv')
    LYRICS_TFIDF_FILE = os.path.join(DATA_DIR, 'id_lyrics_tf-idf_mmsr.tsv')
    
    # Load track information
    try:
        information_df = pd.read_csv(INFORMATION_FILE, sep='\t')
        print("Loaded information_df successfully.")
    except FileNotFoundError:
        print(f"File {INFORMATION_FILE} not found. Please check the path.")
        exit()
    
    # Load genres
    try:
        genres_df = pd.read_csv(GENRES_FILE, sep='\t')
        print("Loaded genres_df successfully.")
    except FileNotFoundError:
        print(f"File {GENRES_FILE} not found. Please check the path.")
        exit()
    
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
    
    # Extract top genre (assuming the first genre in the list is the top genre)
    def get_top_genre(genres):
        return genres[0] if genres else None
    
    catalog_df['top_genre'] = catalog_df['genre'].apply(get_top_genre)
    
    # Load precomputed TF-IDF vectors
    try:
        tfidf_df = pd.read_csv(LYRICS_TFIDF_FILE, sep='\t')
        print("Loaded tfidf_df successfully.")
    except FileNotFoundError:
        print(f"File {LYRICS_TFIDF_FILE} not found. Please check the path.")
        exit()
    
    # Check if 'song' exists in tfidf_df to avoid duplication during merge
    if 'song' in tfidf_df.columns:
        # Rename 'song' in tfidf_df to prevent duplication
        tfidf_df = tfidf_df.rename(columns={'song': 'tfidf_song'})
        print("Renamed 'song' column in tfidf_df to 'tfidf_song' to avoid duplication.")
    
    # Merge TF-IDF vectors with catalog on 'id'
    catalog_df = pd.merge(catalog_df, tfidf_df, on='id', how='left')
    
    # Identify TF-IDF feature columns (exclude 'id' and metadata columns)
    metadata_columns = ['artist', 'song', 'album_name', 'genre', 'top_genre']
    tfidf_feature_columns = [col for col in tfidf_df.columns if col != 'id' and col != 'tfidf_song']
    
    # Check if TF-IDF feature columns exist
    if not tfidf_feature_columns:
        print("\nNo TF-IDF feature columns found. Please verify the tfidf_df structure.")
        exit()
    else:
        print(f"Identified {len(tfidf_feature_columns)} TF-IDF feature columns.")
    
    # Drop rows with any NaN in TF-IDF features to ensure complete data
    initial_catalog_size = len(catalog_df)
    catalog_df = catalog_df.dropna(subset=tfidf_feature_columns).reset_index(drop=True)
    final_catalog_size = len(catalog_df)
    dropped_rows = initial_catalog_size - final_catalog_size
    print(f"Dropped {dropped_rows} tracks due to missing TF-IDF vectors.")
    print(f"Catalog size after merging TF-IDF vectors: {final_catalog_size}")
    
    # Verify essential columns are present
    essential_columns = ['id', 'artist', 'song', 'top_genre']
    missing_columns = [col for col in essential_columns if col not in catalog_df.columns]
    if missing_columns:
        print(f"Missing essential columns in catalog_df: {missing_columns}")
        exit()
    else:
        print("All essential columns are present in catalog_df.")
    
    # Display the first few rows with relevant columns
    print("\nSample of catalog_df:")
    print(catalog_df[['id', 'artist', 'song', 'top_genre']].head())
    
    # Prepare the TF-IDF matrix
    track_ids = catalog_df['id'].tolist()
    tfidf_features = catalog_df.drop(columns=['id', 'artist', 'song', 'album_name', 'genre', 'top_genre'])
    tfidf_matrix = csr_matrix(tfidf_features.values)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Create a mapping from id to index
    id_to_index = {id_: index for index, id_ in enumerate(track_ids)}
    print("Created id_to_index mapping.")
    
    return catalog_df, tfidf_matrix, track_ids, id_to_index

# Load data at startup
catalog_df, tfidf_matrix, track_ids, id_to_index = load_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renders the home page with a form to select a query song and retrieval method.
    """
    if request.method == 'POST':
        # Get selected song ID and retrieval method from the form
        selected_song_id = request.form.get('song_id')
        retrieval_method = request.form.get('retrieval_method')
        
        # Validate inputs
        if not selected_song_id or not retrieval_method:
            error_message = "Please select both a song and a retrieval method."
            song_options = catalog_df.apply(lambda row: (row['id'], f"{row['song']} by {row['artist']}"), axis=1).tolist()
            return render_template('index.html', song_options=song_options, error=error_message)
        
        # Redirect to results page with the selected parameters via GET
        return redirect(url_for('display_results', selected_song_id=selected_song_id, retrieval_method=retrieval_method))
    
    # For GET request, display the selection form
    # Create a list of tuples (id, "Song Title by Artist") for dropdown
    song_options = catalog_df.apply(lambda row: (row['id'], f"{row['song']} by {row['artist']}"), axis=1).tolist()
    
    return render_template('index.html', song_options=song_options, error=None)

@app.route('/results', methods=['GET'])
def display_results():
    """
    Processes the retrieval based on user input and displays the results.
    """
    selected_song_id = request.args.get('selected_song_id')
    retrieval_method = request.args.get('retrieval_method')
    
    if not selected_song_id or not retrieval_method:
        return "Invalid input. Please select a song and retrieval method."
    
    if retrieval_method == 'random':
        # Perform random retrieval
        retrieved_tracks = random_retrieval(catalog_df, N=10)
    elif retrieval_method == 'tfidf':
        # Perform TF-IDF retrieval
        retrieved_tracks = tfidf_retrieval(
            query_track_id=selected_song_id,
            id_to_index=id_to_index,
            tfidf_matrix=tfidf_matrix,
            track_ids=track_ids,
            catalog_df=catalog_df,
            N=10
        )
    else:
        return "Invalid retrieval method selected."
    
    # Get details of the selected song to display
    try:
        selected_song = catalog_df[catalog_df['id'] == selected_song_id].iloc[0].to_dict()
    except IndexError:
        return "Selected song not found in the catalog."
    
    return render_template('results.html', 
                           selected_song=selected_song, 
                           retrieved_tracks=retrieved_tracks, 
                           retrieval_method=retrieval_method)

if __name__ == '__main__':
    app.run(debug=True)
