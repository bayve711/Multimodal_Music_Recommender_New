# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import random
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, hstack
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# ================================
# 1. Define Data Directory and File Paths
# ================================

DATA_DIR = 'data/'

INFORMATION_FILE = os.path.join(DATA_DIR, 'id_information_mmsr.tsv')
GENRES_FILE = os.path.join(DATA_DIR, 'id_genres_mmsr.tsv')
LYRICS_TFIDF_FILE = os.path.join(DATA_DIR, 'id_lyrics_tf-idf_mmsr.tsv')
LYRICS_BERT_FILE = os.path.join(DATA_DIR, 'id_lyrics_bert_mmsr.tsv')
MFCC_BOW_FILE = os.path.join(DATA_DIR, 'id_mfcc_bow_mmsr.tsv')
SPECTRAL_CONTRAST_FILE = os.path.join(DATA_DIR, 'id_blf_spectralcontrast_mmsr.tsv')
VGG19_FILE = os.path.join(DATA_DIR, 'id_vgg19_mmsr.tsv')
RESNET_FILE = os.path.join(DATA_DIR, 'id_resnet_mmsr.tsv')
TAGS_FILE = os.path.join(DATA_DIR, 'id_tags_dict.tsv')
METADATA_FILE = os.path.join(DATA_DIR, 'id_metadata_mmsr.tsv')
ID_URL_FILE = os.path.join(DATA_DIR, 'id_url_mmsr.tsv')  # New

# ================================
# 2. Setup Database with SQLAlchemy
# ================================

Base = declarative_base()


class Retrieval(Base):
    __tablename__ = 'retrievals'
    id = Column(Integer, primary_key=True)
    query_song_id = Column(String, nullable=False)
    retrieval_method = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    results = relationship("RetrievalResult", back_populates="retrieval")


class RetrievalResult(Base):
    __tablename__ = 'retrieval_results'
    id = Column(Integer, primary_key=True)
    retrieval_id = Column(Integer, ForeignKey('retrievals.id'))
    track_id = Column(String, nullable=False)
    artist = Column(String)
    song = Column(String)
    album_name = Column(String)
    similarity = Column(Float)  # Can represent similarity or distance
    url = Column(String)
    retrieval = relationship("Retrieval", back_populates="results")


# Initialize SQLite database
engine = create_engine('sqlite:///retrievals.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session_db = Session()


# ================================
# 3. Load and Prepare Data
# ================================

def load_dataframe(file_path, sep='\t', header='infer', names=None):
    """
    Utility function to load a TSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep=sep, header=header, names=names)
        print(f"Loaded DataFrame from '{file_path}' with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)


# Load Information Dataset
information_df = load_dataframe(INFORMATION_FILE)

# Load Genres Dataset
genres_df = load_dataframe(GENRES_FILE)

# Load Metadata Dataset
metadata_df = load_dataframe(METADATA_FILE)
print("\nMetadata DataFrame Columns:")
print(metadata_df.columns.tolist())

# Load Tags Dataset
tags_df = load_dataframe(TAGS_FILE, header=None, names=['id', 'tags_str'])

# Load Lyrics TF-IDF
lyrics_tfidf_df = load_dataframe(LYRICS_TFIDF_FILE)
tfidf_cols = [col for col in lyrics_tfidf_df.columns if col != 'id']
lyrics_tfidf_df.rename(columns={col: f"tfidf_{col}" for col in tfidf_cols}, inplace=True)
print("Renamed TF-IDF feature columns to prevent conflicts.")

# Load BERT Embeddings
bert_df = load_dataframe(LYRICS_BERT_FILE)
bert_feature_columns = [col for col in bert_df.columns if col != 'id']
bert_df.rename(columns={col: f"bert_{col}" for col in bert_feature_columns}, inplace=True)
print("Renamed BERT feature columns to prevent conflicts.")

# Load MFCC Bag-of-Words
mfcc_bow_df = load_dataframe(MFCC_BOW_FILE)
mfcc_bow_columns = [col for col in mfcc_bow_df.columns if col != 'id']
mfcc_bow_df.rename(columns={col: f"mfcc_{col}" for col in mfcc_bow_columns}, inplace=True)
print("Renamed MFCC Bag-of-Words feature columns to prevent conflicts.")

# Load Spectral Contrast
spectral_contrast_df = load_dataframe(SPECTRAL_CONTRAST_FILE)
spectral_contrast_columns = [col for col in spectral_contrast_df.columns if col != 'id']
spectral_contrast_df.rename(columns={col: f"spectral_{col}" for col in spectral_contrast_columns}, inplace=True)
print("Renamed Spectral Contrast feature columns to prevent conflicts.")

# Load VGG19 Features
vgg19_df = load_dataframe(VGG19_FILE)
vgg19_feature_columns = [col for col in vgg19_df.columns if col != 'id']
vgg19_df.rename(columns={col: f"vgg19_{col}" for col in vgg19_feature_columns}, inplace=True)
print("Renamed VGG19 feature columns to prevent conflicts.")

# Load ResNet Features
resnet_df = load_dataframe(RESNET_FILE)
resnet_feature_columns = [col for col in resnet_df.columns if col != 'id']
resnet_df.rename(columns={col: f"resnet_{col}" for col in resnet_feature_columns}, inplace=True)
print("Renamed ResNet feature columns to prevent conflicts.")

# Load YouTube URLs
id_url_df = load_dataframe(ID_URL_FILE)
print("\nYouTube URLs DataFrame Columns:")
print(id_url_df.columns.tolist())

# ================================
# 4. Merge and Preprocess DataFrames
# ================================

# Merge Information and Metadata
catalog_df = pd.merge(information_df, metadata_df[['id', 'popularity']], on='id', how='left')
print(f"\nMerged catalog_df shape after adding Metadata: {catalog_df.shape}")

# Verify 'popularity' Column
if 'popularity' in catalog_df.columns:
    print("\n'popularity' column successfully added to catalog_df.")
    print(catalog_df[['id', 'artist', 'song', 'popularity']].head())
else:
    print("Error: 'popularity' column is still missing after merging Metadata.")
    exit(1)

# Handle Missing 'popularity' Values
missing_popularity = catalog_df['popularity'].isnull().sum()
print(f"\nNumber of tracks with missing 'popularity': {missing_popularity}")

if missing_popularity > 0:
    # Drop tracks with missing 'popularity'
    initial_size = len(catalog_df)
    catalog_df = catalog_df.dropna(subset=['popularity']).reset_index(drop=True)
    final_size = len(catalog_df)
    dropped = initial_size - final_size
    print(f"Dropped {dropped} tracks due to missing 'popularity' values.")
else:
    print("No missing 'popularity' values found.")


# Function to parse genres from string to list
def parse_genres(genre_str):
    if pd.isnull(genre_str):
        return []
    return [genre.strip() for genre in genre_str.split(',')]


# Apply parsing to 'genre' column
genres_df['genre'] = genres_df['genre'].apply(parse_genres)

# Merge Information and Genres to update catalog_df
catalog_df = pd.merge(catalog_df, genres_df, on='id', how='left')
print(f"Merged catalog_df shape after merging Genres: {catalog_df.shape}")

# Handle missing genres by assigning an empty list
catalog_df['genre'] = catalog_df['genre'].apply(lambda x: x if isinstance(x, list) else [])


# Function to get the top genre from the genre list
def get_top_genre(genres_list):
    if not genres_list:
        return None
    return genres_list[0]  # Modify as needed


# Apply the function to determine the top genre
catalog_df['top_genre'] = catalog_df['genre'].apply(get_top_genre)

# Display sample data
print("\nSample of catalog_df:")
print(catalog_df[['id', 'artist', 'song', 'top_genre']].head())


# Merge Tags into catalog_df
def parse_tags(tag_str):
    if pd.isnull(tag_str):
        return []
    try:
        tags_dict = ast.literal_eval(tag_str)
        if isinstance(tags_dict, dict):
            return list(tags_dict.keys())
        else:
            return []
    except (ValueError, SyntaxError):
        return []


tags_df['tags'] = tags_df['tags_str'].apply(parse_tags)

catalog_df = pd.merge(catalog_df, tags_df[['id', 'tags']], on='id', how='left')
print(f"Merged catalog_df shape after merging Tags: {catalog_df.shape}")

# Handle missing tags by assigning an empty list
catalog_df['tags'] = catalog_df['tags'].apply(lambda x: x if isinstance(x, list) else [])

# Merge YouTube URLs into catalog_df
catalog_df = pd.merge(catalog_df, id_url_df, on='id', how='left')
print(f"Merged catalog_df shape after merging YouTube URLs: {catalog_df.shape}")

# Handle missing URLs by assigning a placeholder or empty string
catalog_df['url'] = catalog_df['url'].fillna('')

# Display sample data to verify
print("\nSample of catalog_df after merging tags and URLs:")
print(catalog_df[['id', 'artist', 'song', 'top_genre', 'tags', 'url']].head())

# ================================
# 5. Prepare Feature Matrices for Retrieval Systems
# ================================

# Merge all feature DataFrames into catalog_df
catalog_df = pd.merge(catalog_df, lyrics_tfidf_df, on='id', how='left')
catalog_df = pd.merge(catalog_df, bert_df, on='id', how='left')
catalog_df = pd.merge(catalog_df, mfcc_bow_df, on='id', how='left')
catalog_df = pd.merge(catalog_df, spectral_contrast_df, on='id', how='left')
catalog_df = pd.merge(catalog_df, vgg19_df, on='id', how='left')
catalog_df = pd.merge(catalog_df, resnet_df, on='id', how='left')

print(f"\nFinal catalog_df shape after merging all features: {catalog_df.shape}")

# Handle missing values by dropping any rows with NaNs in feature columns
feature_columns = (
        [col for col in catalog_df.columns if col.startswith('tfidf_')] +
        [col for col in catalog_df.columns if col.startswith('bert_')] +
        [col for col in catalog_df.columns if col.startswith('mfcc_')] +
        [col for col in catalog_df.columns if col.startswith('spectral_')] +
        [col for col in catalog_df.columns if col.startswith('vgg19_')] +
        [col for col in catalog_df.columns if col.startswith('resnet_')]
)

missing_features = catalog_df[feature_columns].isnull().sum().sum()
print(f"Total missing feature values: {missing_features}")

if missing_features > 0:
    # Drop tracks with missing features
    initial_size = len(catalog_df)
    catalog_df = catalog_df.dropna(subset=feature_columns).reset_index(drop=True)
    final_size = len(catalog_df)
    dropped = initial_size - final_size
    print(f"Dropped {dropped} tracks due to missing feature values.")
else:
    print("No missing feature values found.")

# Extract feature matrices
tfidf_feature_columns = [col for col in catalog_df.columns if col.startswith('tfidf_')]
bert_feature_columns = [col for col in catalog_df.columns if col.startswith('bert_')]
mfcc_feature_columns = [col for col in catalog_df.columns if col.startswith('mfcc_')]
spectral_feature_columns = [col for col in catalog_df.columns if col.startswith('spectral_')]
vgg19_feature_columns = [col for col in catalog_df.columns if col.startswith('vgg19_')]
resnet_feature_columns = [col for col in catalog_df.columns if col.startswith('resnet_')]

# Normalize TF-IDF features
tfidf_features = catalog_df[tfidf_feature_columns].values
tfidf_matrix = normalize(tfidf_features, norm='l2')
tfidf_matrix = csr_matrix(tfidf_matrix)
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

# Normalize BERT features
bert_features = catalog_df[bert_feature_columns].values
bert_matrix = normalize(bert_features, norm='l2')
bert_matrix = csr_matrix(bert_matrix)
print(f"BERT Matrix Shape: {bert_matrix.shape}")

# Normalize MFCC features
mfcc_features = catalog_df[mfcc_feature_columns].values
mfcc_matrix = normalize(mfcc_features, norm='l2')
mfcc_matrix = csr_matrix(mfcc_matrix)
print(f"MFCC Matrix Shape: {mfcc_matrix.shape}")

# Normalize Spectral Contrast features
spectral_features = catalog_df[spectral_feature_columns].values
spectral_matrix = normalize(spectral_features, norm='l2')
spectral_matrix = csr_matrix(spectral_matrix)
print(f"Spectral Contrast Matrix Shape: {spectral_matrix.shape}")

# Normalize VGG19 features
vgg19_features = catalog_df[vgg19_feature_columns].values
vgg19_matrix = normalize(vgg19_features, norm='l2')
vgg19_matrix = csr_matrix(vgg19_matrix)
print(f"VGG19 Matrix Shape: {vgg19_matrix.shape}")

# Normalize ResNet features
resnet_features = catalog_df[resnet_feature_columns].values
resnet_matrix = normalize(resnet_features, norm='l2')
resnet_matrix = csr_matrix(resnet_matrix)
print(f"ResNet Matrix Shape: {resnet_matrix.shape}")


# Build Tag Vocabulary and Vectorize Tags
def build_tag_vocabulary(catalog_df):
    """
    Builds a tag vocabulary from the catalog.

    Parameters:
    - catalog_df (pd.DataFrame): DataFrame containing 'id' and 'tags'.

    Returns:
    - tag_to_index (dict): Mapping from tag to index.
    """
    all_tags = set(tag for tags in catalog_df['tags'] for tag in tags)
    tag_to_index = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    return tag_to_index


def vectorize_tags(catalog_df, tag_to_index):
    """
    Vectorizes the tags for each track.

    Parameters:
    - catalog_df (pd.DataFrame): DataFrame containing 'id' and 'tags'.
    - tag_to_index (dict): Mapping from tag to index.

    Returns:
    - tag_matrix (csr_matrix): Sparse matrix of tag vectors.
    """
    row_indices = []
    col_indices = []
    data = []

    for row, tags in enumerate(catalog_df['tags']):
        for tag in tags:
            if tag in tag_to_index:
                col = tag_to_index[tag]
                row_indices.append(row)
                col_indices.append(col)
                data.append(1)  # Binary occurrence

    tag_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(len(catalog_df), len(tag_to_index)))
    tag_matrix = normalize(tag_matrix, norm='l2')
    return tag_matrix


# Building Tag Vocabulary
tag_to_index = build_tag_vocabulary(catalog_df)
print(f"Total unique tags: {len(tag_to_index)}")

# Vectorizing Tags
tag_matrix = vectorize_tags(catalog_df, tag_to_index)
print(f"Tag Matrix Shape: {tag_matrix.shape}")

# Prepare Combined Feature Matrices for Fusion
early_fusion_matrix = hstack([tfidf_matrix, bert_matrix]).tocsr()
print(f"Early Fusion (TF-IDF + BERT) Matrix Shape: {early_fusion_matrix.shape}")

# ================================
# 6. Prepare Track IDs and Index Mapping
# ================================

track_ids = catalog_df['id'].tolist()
id_to_index = {track_id: idx for idx, track_id in enumerate(track_ids)}


# ================================
# 7. Define Retrieval Functions
# ================================

def random_retrieval(query_track_id, catalog_df, N=10):
    """
    Randomly selects N tracks from the catalog, excluding the query track.
    """
    # Exclude the query track
    candidates = catalog_df[catalog_df['id'] != query_track_id]

    # Determine the number of tracks to sample
    sample_size = min(N, len(candidates))

    # Randomly sample N tracks
    retrieved_tracks = candidates.sample(n=sample_size, replace=False, random_state=random.randint(0, 1000000))

    return retrieved_tracks.reset_index(drop=True)


def tfidf_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks most similar to the query track based on TF-IDF cosine similarity.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    # Exclude the query track
    similarities[query_index] = -1

    # Get top N indices
    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity_score': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'url']],
                                   on='id', how='left')

    return retrieved_tracks_df


def bert_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks most similar to the query track based on BERT cosine similarity.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    # Exclude the query track
    similarities[query_index] = -1

    # Get top N indices
    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity_score': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'url']],
                                   on='id', how='left')

    return retrieved_tracks_df


def mfcc_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks most similar to the query track based on MFCC Euclidean distance.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index].toarray()

    # Compute Euclidean distances
    distances = np.linalg.norm(feature_matrix - query_vector, axis=1)

    # Exclude the query track
    distances[query_index] = np.inf

    # Get top N indices with smallest distances
    top_indices = distances.argsort()[:N]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_distances = [distances[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'distance': retrieved_distances
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'url']],
                                   on='id', how='left')

    return retrieved_tracks_df


def spectral_contrast_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks most similar to the query track based on Spectral Contrast Cosine similarity.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    # Exclude the query track
    similarities[query_index] = -1

    # Get top N indices
    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity_score': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'url']],
                                   on='id', how='left')

    return retrieved_tracks_df


def vgg19_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks most similar to the query track based on VGG19 Cosine similarity.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    # Exclude the query track
    similarities[query_index] = -1

    # Get top N indices
    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity_score': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'url']],
                                   on='id', how='left')

    return retrieved_tracks_df


def resnet_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks most similar to the query track based on ResNet Euclidean distance.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index].toarray()

    # Compute Euclidean distances
    distances = np.linalg.norm(feature_matrix - query_vector, axis=1)

    # Exclude the query track
    distances[query_index] = np.inf

    # Get top N indices with smallest distances
    top_indices = distances.argsort()[:N]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_distances = [distances[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'distance': retrieved_distances
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'url']],
                                   on='id', how='left')

    return retrieved_tracks_df


def tag_based_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks most similar to the query track based on Tag similarity.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    # Exclude the query track
    similarities[query_index] = -1

    # Get top N indices
    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity_score': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'url']],
                                   on='id', how='left')

    return retrieved_tracks_df


def early_fusion_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks using Early Fusion by combining TF-IDF and BERT feature matrices.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    # Exclude the query track
    similarities[query_index] = -1

    # Get top N indices
    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity_score': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'url']],
                                   on='id', how='left')

    return retrieved_tracks_df


def late_fusion_retrieval(query_track_id, id_to_index, feature_matrix1, feature_matrix2, track_ids, catalog_df, N=10,
                          weight1=0.5, weight2=0.5):
    """
    Retrieves N tracks using Late Fusion by combining MFCC and VGG19 retrieval scores.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]

    # Compute cosine similarity for MFCC
    query_vector1 = feature_matrix1[query_index]
    similarities1 = cosine_similarity(query_vector1, feature_matrix1).flatten()
    similarities1[query_index] = -1  # Exclude query

    # Compute cosine similarity for VGG19
    query_vector2 = feature_matrix2[query_index]
    similarities2 = cosine_similarity(query_vector2, feature_matrix2).flatten()
    similarities2[query_index] = -1  # Exclude query

    # Weighted average of similarities
    aggregated_similarities = weight1 * similarities1 + weight2 * similarities2

    # Get top N indices
    top_indices = aggregated_similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [aggregated_similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'aggregated_similarity_score': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'url']],
                                   on='id', how='left')

    return retrieved_tracks_df


# ================================
# 8. Define Retrieval Systems Dictionary
# ================================

retrieval_systems = {
    'Random Retrieval': random_retrieval,
    'Tag-Based Retrieval': tag_based_retrieval,
    'Early Fusion (TF-IDF + BERT) Retrieval': early_fusion_retrieval,
    'Late Fusion (MFCC + VGG19) Retrieval': late_fusion_retrieval,
    'TF-IDF Retrieval': tfidf_retrieval,
    'BERT Retrieval': bert_retrieval,
    'MFCC Retrieval': mfcc_retrieval,
    'Spectral Contrast Retrieval': spectral_contrast_retrieval,
    'VGG19 Retrieval': vgg19_retrieval,
    'ResNet Retrieval': resnet_retrieval
}

# Define a mapping from retrieval method names to their corresponding feature matrices
method_to_matrix = {
    'TF-IDF Retrieval': tfidf_matrix,
    'BERT Retrieval': bert_matrix,
    'MFCC Retrieval': mfcc_matrix,
    'Spectral Contrast Retrieval': spectral_matrix,
    'VGG19 Retrieval': vgg19_matrix,
    'ResNet Retrieval': resnet_matrix,
    'Tag-Based Retrieval': tag_matrix,
    'Early Fusion (TF-IDF + BERT) Retrieval': early_fusion_matrix,
    # 'Late Fusion (MFCC + VGG19) Retrieval' is handled separately
    # 'Random Retrieval' is handled separately
}

# ================================
# 9. Define Routes
# ================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renders the home page with a form to select a query song and IR systems.
    """
    if request.method == 'POST':
        # Get selected song ID and IR systems from the form
        selected_song_id = request.form.get('song_id')
        selected_methods = request.form.getlist('ir_methods')  # List of selected IR systems

        # Validate inputs
        if not selected_song_id or not selected_methods:
            flash("Please select a song and at least one retrieval method.", "error")
            song_options = catalog_df.apply(lambda row: (row['id'], f"{row['song']} by {row['artist']}"),
                                            axis=1).tolist()
            ir_options = list(retrieval_systems.keys())
            return render_template('index.html', song_options=song_options, ir_options=ir_options)

        # Redirect to results page with the selected parameters via GET
        return redirect(
            url_for('display_results', selected_song_id=selected_song_id, selected_methods=selected_methods))

    # For GET request, display the selection form
    # Create a list of tuples (id, "Song Title by Artist") for dropdown
    song_options = catalog_df.apply(lambda row: (row['id'], f"{row['song']} by {row['artist']}"), axis=1).tolist()
    ir_options = list(retrieval_systems.keys())

    return render_template('index.html', song_options=song_options, ir_options=ir_options)


@app.route('/results', methods=['GET'])
def display_results():
    """
    Processes the retrieval based on user input and displays the results from selected IR systems.
    """
    selected_song_id = request.args.get('selected_song_id')
    selected_methods = request.args.getlist('selected_methods')  # List of selected IR systems

    if not selected_song_id or not selected_methods:
        flash("Invalid input. Please select a song and at least one retrieval method.", "error")
        return redirect(url_for('index'))

    # Verify that the selected_song_id exists
    if selected_song_id not in id_to_index:
        flash("Selected song not found in the catalog.", "error")
        return redirect(url_for('index'))

    # Initialize a dictionary to store retrieval results for each selected IR system
    retrieval_results = {}

    # Record the retrieval in the database
    for method in selected_methods:
        if method not in retrieval_systems:
            flash(f"Retrieval method '{method}' is not recognized.", "error")
            continue

        func = retrieval_systems[method]

        if method == 'Random Retrieval':
            retrieved_tracks = func(selected_song_id, catalog_df, N=10)
            similarity_col = None  # Random retrieval doesn't have a similarity score
        elif method == 'Early Fusion (TF-IDF + BERT) Retrieval':
            retrieved_tracks = func(
                query_track_id=selected_song_id,
                id_to_index=id_to_index,
                feature_matrix=early_fusion_matrix,
                track_ids=track_ids,
                catalog_df=catalog_df,
                N=10
            )
            similarity_col = 'similarity_score'
        elif method == 'Late Fusion (MFCC + VGG19) Retrieval':
            retrieved_tracks = func(
                query_track_id=selected_song_id,
                id_to_index=id_to_index,
                feature_matrix1=mfcc_matrix,
                feature_matrix2=vgg19_matrix,
                track_ids=track_ids,
                catalog_df=catalog_df,
                N=10,
                weight1=0.5,
                weight2=0.5
            )
            similarity_col = 'aggregated_similarity_score'
        elif method == 'Tag-Based Retrieval':
            retrieved_tracks = func(
                query_track_id=selected_song_id,
                id_to_index=id_to_index,
                feature_matrix=tag_matrix,
                track_ids=track_ids,
                catalog_df=catalog_df,
                N=10
            )
            similarity_col = 'similarity_score'
        elif method in ['TF-IDF Retrieval', 'BERT Retrieval', 'Spectral Contrast Retrieval', 'VGG19 Retrieval']:
            # Use the method_to_matrix mapping to get the appropriate feature matrix
            feature_matrix = method_to_matrix.get(method)
            if feature_matrix is None:
                flash(f"No feature matrix found for method '{method}'.", "error")
                continue
            retrieved_tracks = func(
                query_track_id=selected_song_id,
                id_to_index=id_to_index,
                feature_matrix=feature_matrix,
                track_ids=track_ids,
                catalog_df=catalog_df,
                N=10
            )
            similarity_col = 'similarity_score'
        elif method == 'MFCC Retrieval':
            retrieved_tracks = func(
                query_track_id=selected_song_id,
                id_to_index=id_to_index,
                feature_matrix=mfcc_matrix,
                track_ids=track_ids,
                catalog_df=catalog_df,
                N=10
            )
            similarity_col = 'distance'
        elif method == 'ResNet Retrieval':
            retrieved_tracks = func(
                query_track_id=selected_song_id,
                id_to_index=id_to_index,
                feature_matrix=resnet_matrix,
                track_ids=track_ids,
                catalog_df=catalog_df,
                N=10
            )
            similarity_col = 'distance'
        else:
            retrieved_tracks = pd.DataFrame()
            similarity_col = None

        # Save retrieval results to the database
        new_retrieval = Retrieval(
            query_song_id=selected_song_id,
            retrieval_method=method
        )
        session_db.add(new_retrieval)
        session_db.commit()  # Commit to get the retrieval ID

        for _, row in retrieved_tracks.iterrows():
            result = RetrievalResult(
                retrieval_id=new_retrieval.id,
                track_id=row['id'],
                artist=row['artist'],
                song=row['song'],
                album_name=row['album_name'],
                similarity=row[similarity_col] if similarity_col else None,
                url=row['url']
            )
            session_db.add(result)

        session_db.commit()

        # Store the retrieved tracks in the dictionary for rendering
        retrieval_results[method] = retrieved_tracks.to_dict(orient='records')

    # Get details of the selected song to display
    selected_song = catalog_df[catalog_df['id'] == selected_song_id].iloc[0].to_dict()

    return render_template('results.html',
                           selected_song=selected_song,
                           retrieval_results=retrieval_results,
                           selected_methods=selected_methods)

@app.route('/history', methods=['GET'])
def view_history():
    """
    Displays the history of past retrievals.
    """
    # Fetch all retrievals ordered by timestamp descending
    retrievals = session_db.query(Retrieval).order_by(Retrieval.timestamp.desc()).all()

    # Prepare data for rendering
    history = []
    for retrieval in retrievals:
        results = session_db.query(RetrievalResult).filter(RetrievalResult.retrieval_id == retrieval.id).all()
        tracks = []
        for result in results:
            track_info = {
                'id': result.track_id,
                'artist': result.artist,
                'song': result.song,
                'album_name': result.album_name,
                'similarity_score': result.similarity,  # Can be similarity or distance
                'url': result.url
            }
            tracks.append(track_info)

        # Fetch query song details
        query_song = catalog_df[catalog_df['id'] == retrieval.query_song_id].iloc[0].to_dict()

        history.append({
            'id': retrieval.id,
            'query_song': {
                'id': query_song['id'],
                'artist': query_song['artist'],
                'song': query_song['song'],
                'album_name': query_song['album_name'],
                'url': query_song['url'],
                'top_genre': query_song['top_genre']
            },
            'retrieval_method': retrieval.retrieval_method,
            'timestamp': retrieval.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'results': tracks
        })

    return render_template('history.html', history=history)


# ================================
# 10. Run the Flask Application
# ================================

if __name__ == '__main__':
    app.run(debug=True)
