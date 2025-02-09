# backend.py

import pandas as pd
import numpy as np
import random
import ast
import os
import time
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, hstack
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
import string
import warnings
from datetime import datetime
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

warnings.filterwarnings('ignore')


#1. data files directory

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
FILTERED_TAGS_FILE = os.path.join(DATA_DIR, 'filtered_id_tags_dict.tsv')
METADATA_FILE = os.path.join(DATA_DIR, 'id_metadata_mmsr.tsv')


# 2. datasets

@st.cache_data
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

information_df = load_dataframe(INFORMATION_FILE)

genres_df = load_dataframe(GENRES_FILE)

metadata_df = load_dataframe(METADATA_FILE)

tags_df = load_dataframe(TAGS_FILE, header=None, names=['id', 'tags_str'])

#lyrics TF-IDF
lyrics_tfidf_df = load_dataframe(LYRICS_TFIDF_FILE)
tfidf_cols = [col for col in lyrics_tfidf_df.columns if col != 'id']
lyrics_tfidf_df.rename(columns={col: f"tfidf_{col}" for col in tfidf_cols}, inplace=True)

#BERT embeddings
bert_df = load_dataframe(LYRICS_BERT_FILE)
bert_feature_columns = [col for col in bert_df.columns if col != 'id']
bert_df.rename(columns={col: f"bert_{col}" for col in bert_feature_columns}, inplace=True)

#MFCC Bag-of-Words
mfcc_bow_df = load_dataframe(MFCC_BOW_FILE)
mfcc_bow_columns = [col for col in mfcc_bow_df.columns if col != 'id']
mfcc_bow_df.rename(columns={col: f"mfcc_{col}" for col in mfcc_bow_columns}, inplace=True)

#spectral contrast
spectral_contrast_df = load_dataframe(SPECTRAL_CONTRAST_FILE)
spectral_contrast_columns = [col for col in spectral_contrast_df.columns if col != 'id']
spectral_contrast_df.rename(columns={col: f"spectral_{col}" for col in spectral_contrast_columns}, inplace=True)

#VGG19 features
vgg19_df = load_dataframe(VGG19_FILE)
vgg19_feature_columns = [col for col in vgg19_df.columns if col != 'id']
vgg19_df.rename(columns={col: f"vgg19_{col}" for col in vgg19_feature_columns}, inplace=True)


#resnet features
resnet_df = load_dataframe(RESNET_FILE)
resnet_feature_columns = [col for col in resnet_df.columns if col != 'id']
resnet_df.rename(columns={col: f"resnet_{col}" for col in resnet_feature_columns}, inplace=True)

# 3. dataframes merging&preprocessing

catalog_df = pd.merge(information_df, metadata_df[['id', 'popularity']], on='id', how='left')

#parse genres from string to list
def parse_genres(genre_str):
    if pd.isnull(genre_str):
        return []
    return [genre.strip().lower() for genre in genre_str.split(',')]


#parsing to 'genre' column
genres_df['genre'] = genres_df['genre'].apply(parse_genres)

#update catalog_df
catalog_df = pd.merge(catalog_df, genres_df, on='id', how='left')

#assigning an empty list
catalog_df['genre'] = catalog_df['genre'].apply(lambda x: x if isinstance(x, list) else [])


#top genre from the genre list
def get_top_genre(genres_list):
    if not genres_list:
        return None
    first_genre = genres_list[0]
    if isinstance(first_genre, list):
        return first_genre[0] if first_genre else None
    return first_genre


#determine the top genre
catalog_df['top_genre'] = catalog_df['genre'].apply(get_top_genre)

#stripping and lowercasing to normalize
catalog_df['top_genre'] = catalog_df['top_genre'].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)

#sample data
print("\nSample of catalog_df:")
print(catalog_df[['id', 'artist', 'song', 'top_genre']].head())

#4. tags with weights

catalog_df = pd.merge(catalog_df, tags_df[['id', 'tags_str']], on='id', how='left')

#empty string
catalog_df['tags_str'] = catalog_df['tags_str'].fillna('{}')


#parse tags and weights from tags_str
def parse_tags_and_weights(tag_weight_str):
    """
    Parses the 'tags_str' string into separate lists of tags and weights.
    """
    try:
        # Safely evaluate the string to a dictionary
        tag_weight_dict = ast.literal_eval(tag_weight_str)
        if isinstance(tag_weight_dict, dict):
            tags = list(tag_weight_dict.keys())
            weights = list(tag_weight_dict.values())
            return tags, weights
        else:
            print(f"Warning: Expected dict, got {type(tag_weight_dict)}")
            return [], []
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing tags: {e}")
        return [], []


#create 'tags' and 'weights' columns
catalog_df[['tags', 'weights']] = catalog_df.apply(
    lambda row: pd.Series(parse_tags_and_weights(row['tags_str'])),
    axis=1
)

print("\nSample of catalog_df after parsing tags and weights:")
print(catalog_df[['id', 'artist', 'song', 'top_genre', 'tags', 'weights']].head())


#5. tags&weights

def preprocess_tag(tag):
    """
    Preprocesses a single tag: lowercases, removes punctuation, and lemmatizes.
    """
    tag = tag.lower()
    tag = tag.translate(str.maketrans('', '', string.punctuation))
    tag = lemmatizer.lemmatize(tag)
    return tag

catalog_df['processed_tags'] = catalog_df.apply(
    lambda row: [preprocess_tag(tag) for tag in row['tags']],
    axis=1
)

#6. Exclude Genre Tags from Processed Tags

#define genre tags based on the 'genre' column
genre_tags = set()
for genres in catalog_df['genre']:
    for genre in genres:
        genre_tags.add(preprocess_tag(genre))

#add 'alternative' and 'indie' to genre_tags
genre_tags.update(['alternative', 'indie'])

print(f"\nGenre Tags to Exclude: {genre_tags}")


def exclude_genre_tags(tags, genre_tags):
    """
    Excludes any tag that matches any genre tag exactly.
    """
    return [tag for tag in tags if tag not in genre_tags]


#exclusion of genre tags
catalog_df['filtered_processed_tags'] = catalog_df.apply(
    lambda row: exclude_genre_tags(row['processed_tags'], genre_tags),
    axis=1
)

#7. filter tags with a threshold

min_weight_threshold = 30

#column 'filtered_processed_tags_final' that retains only tags with weight >= threshold
catalog_df['filtered_processed_tags_final'] = catalog_df.apply(
    lambda row: [tag for tag, weight in zip(row['filtered_processed_tags'], row['weights']) if
                 weight >= min_weight_threshold],
    axis=1
)

#filter weights
catalog_df['filtered_weights_final'] = catalog_df.apply(
    lambda row: [weight for tag, weight in zip(row['filtered_processed_tags'], row['weights']) if
                 weight >= min_weight_threshold],
    axis=1
)

sample_tracks = catalog_df[['id', 'artist', 'song', 'filtered_processed_tags_final', 'filtered_weights_final']].sample(
    5, random_state=42)


# 9.exclude tracks w/ no signficant tags

#number of tracks with no significant tags
num_no_tags = catalog_df[catalog_df['filtered_processed_tags_final'].apply(len) == 0].shape[0]
total_tracks = catalog_df.shape[0]
percentage_no_tags = (num_no_tags / total_tracks) * 100
# print(f"\nNumber of tracks with no significant tags: {num_no_tags} ({percentage_no_tags:.2f}%)")

#filter tracks with no significant tags
catalog_df_filtered = catalog_df[catalog_df['filtered_processed_tags_final'].apply(len) > 0].reset_index(drop=True)
# print(f"Number of tracks after excluding those with no significant tags: {catalog_df_filtered.shape[0]}")


#10. Tag Vectorization TF-IDF

@st.cache_data
def vectorize_tags_tfidf(catalog_df, min_df=1, ngram_range=(1, 1)):
    """
    Vectorizes tags using TF-IDF with adjustable n-gram range.

    Parameters:
    - catalog_df (pd.DataFrame): DataFrame containing 'filtered_processed_tags_final'.
    - min_df (int): Minimum document frequency for a tag to be included.
    - ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams.

    Returns:
    - tag_matrix_tfidf (csr_matrix): TF-IDF normalized tag matrix.
    - vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """
    catalog_df['tags_str_final_tfidf'] = catalog_df['filtered_processed_tags_final'].apply(lambda tags: ' '.join(tags))

    #TF-IDF Vectorizer with n-gram range
    vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)

    tag_matrix_tfidf = vectorizer.fit_transform(catalog_df['tags_str_final_tfidf'])

    unique_vectors = len(np.unique(tag_matrix_tfidf.toarray(), axis=0))

    return tag_matrix_tfidf, vectorizer


#apply vectorization on Filtered Catalog
tag_matrix_tfidf, tfidf_vectorizer = vectorize_tags_tfidf(catalog_df_filtered, min_df=1, ngram_range=(1, 2))


#11. Tag Vectorization for Tag-Based Retrieval

def vectorize_tags_binary(catalog_df, min_df=1):
    """
    Vectorizes tags using binary encoding (presence/absence).

    Parameters:
    - catalog_df (pd.DataFrame): DataFrame containing 'filtered_processed_tags_final'.
    - min_df (int): Minimum document frequency for a tag to be included.

    Returns:
    - tag_matrix_binary (csr_matrix): Binary tag matrix.
    - vectorizer (CountVectorizer): Fitted CountVectorizer with binary encoding.
    """
    catalog_df['tags_str_final_binary'] = catalog_df['filtered_processed_tags_final'].apply(lambda tags: ' '.join(tags))

    #CountVectorizer with binary=True
    vectorizer = CountVectorizer(binary=True, min_df=min_df)

    tag_matrix_binary = vectorizer.fit_transform(catalog_df['tags_str_final_binary'])
    print(f"Binary Tag Matrix Shape: {tag_matrix_binary.shape}")

    unique_vectors = len(np.unique(tag_matrix_binary.toarray(), axis=0))
    print(f"Number of unique Binary Tag vectors: {unique_vectors} out of {tag_matrix_binary.shape[0]}")

    return tag_matrix_binary, vectorizer


#apply Binary Vectorization on Filtered Catalog
tag_matrix_binary, binary_vectorizer = vectorize_tags_binary(catalog_df_filtered, min_df=1)



#12. Feature Matrices for Retrieval Systems

#merge features
def merge_features(catalog_df_filtered, feature_df, prefix):
    """
    Merges a feature DataFrame with catalog_df_filtered on 'id'.
    Renames feature columns with the given prefix.
    """
    feature_df = pd.merge(catalog_df_filtered[['id']], feature_df, on='id', how='left').sort_values('id').reset_index(
        drop=True)
    feature_cols = [col for col in feature_df.columns if col != 'id']
    feature_df.rename(columns={col: f"{prefix}_{col}" for col in feature_cols}, inplace=True)
    return feature_df


#merge and align all feature matrices
merged_bert_df = merge_features(catalog_df_filtered, bert_df, 'bert')
merged_mfcc_bow_df = merge_features(catalog_df_filtered, mfcc_bow_df, 'mfcc')
merged_spectral_contrast_df = merge_features(catalog_df_filtered, spectral_contrast_df, 'spectral')
merged_vgg19_df = merge_features(catalog_df_filtered, vgg19_df, 'vgg19')
merged_resnet_df = merge_features(catalog_df_filtered, resnet_df, 'resnet')

#convert merged feature DataFrames to numpy arrays or sparse matrices
bert_matrix = merged_bert_df.drop('id', axis=1).values
mfcc_bow_matrix = merged_mfcc_bow_df.drop('id', axis=1).values
spectral_contrast_matrix = merged_spectral_contrast_df.drop('id', axis=1).values
vgg19_matrix = merged_vgg19_df.drop('id', axis=1).values
resnet_matrix = merged_resnet_df.drop('id', axis=1).values

#convert to sparse matrices if necessary
bert_matrix = csr_matrix(bert_matrix)
mfcc_bow_matrix = csr_matrix(mfcc_bow_matrix)
spectral_contrast_matrix = csr_matrix(spectral_contrast_matrix)
vgg19_matrix = csr_matrix(vgg19_matrix)
resnet_matrix = csr_matrix(resnet_matrix)

#feature matrices for retrieval functions that require them
feature_matrices = {
    'TF-IDF Retrieval': tag_matrix_tfidf,
    'Tag-Based Retrieval': tag_matrix_binary,
    'BERT Retrieval': bert_matrix,
    'MFCC Retrieval': mfcc_bow_matrix,
    'Spectral Contrast Retrieval': spectral_contrast_matrix,
    'VGG19 Retrieval': vgg19_matrix,
    'ResNet Retrieval': resnet_matrix
}


#13. IRSs

def random_retrieval(query_track_id, catalog_df, N=10):
    """
    Randomly selects N tracks from the catalog, excluding the query track.

    Parameters:
    - query_track_id (str): The ID of the query track.
    - catalog_df (pd.DataFrame): The catalog containing all tracks.
    - N (int): Number of tracks to retrieve.

    Returns:
    - retrieved_tracks (pd.DataFrame): DataFrame of retrieved tracks.
    """
    if catalog_df is None:
        raise ValueError("catalog_df must be provided for Random Retrieval.")

    #exclude query track
    candidates = catalog_df[catalog_df['id'] != query_track_id]

    #number of tracks to sample
    sample_size = min(N, len(candidates))

    #sample N tracks
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

    #cosine similarity
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    #exclude query track
    similarities[query_index] = -1

    #top N indices
    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity': retrieved_scores
    })

    #'top_genre' in the retrieved tracks
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def tag_based_retrieval_cosine(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks most similar to the query track based on Cosine similarity of binary tags.

    Parameters:
    - query_track_id (str): The ID of the query track.
    - id_to_index (dict): Mapping from track ID to index in feature_matrix.
    - feature_matrix (csr_matrix): Binary tag feature matrix.
    - track_ids (list): List of all track IDs.
    - catalog_df (pd.DataFrame): DataFrame containing track metadata.
    - N (int): Number of tracks to retrieve.

    Returns:
    - retrieved_tracks_df (pd.DataFrame): DataFrame of retrieved tracks with similarity scores.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    #query_vector is non-zero
    if query_vector.nnz == 0:
        print(f"Query track ID {query_track_id} has no significant tags. Skipping retrieval.")
        return pd.DataFrame()

    #cosine similarity between the query and all tracks
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1

    #top N similar tracks
    top_indices = similarities.argsort()[-N:][::-1]

    #track IDs and their similarity scores
    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    #DataFrame for the retrieved tracks
    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'cosine_similarity': retrieved_scores
    })

    #'top_genre' in the retrieved tracks
    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
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

    #cosine similarity
    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
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

    #euclidean distances
    distances = np.linalg.norm(feature_matrix - query_vector, axis=1)


    distances[query_index] = np.inf

    #top N indices with smallest distances
    top_indices = distances.argsort()[:N]

    # Handle potential IndexError
    if top_indices.size < N:
        print(f"Warning: Retrieved fewer than {N} tracks for query ID {query_track_id}.")

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_distances = distances[top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'distance': retrieved_distances
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
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

    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
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

    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'similarity': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
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

    distances = np.linalg.norm(feature_matrix - query_vector, axis=1)

    distances[query_index] = np.inf

    top_indices = distances.argsort()[:N]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_distances = distances[top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'distance': retrieved_distances
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def early_fusion_retrieval(query_track_id, id_to_index, feature_matrix, track_ids, catalog_df, N=10):
    """
    Retrieves N tracks using Early Fusion by combining BERT and MFCC feature matrices.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]
    query_vector = feature_matrix[query_index]

    similarities = cosine_similarity(query_vector, feature_matrix).flatten()

    similarities[query_index] = -1

    top_indices = similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'aggregated_similarity': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


def late_fusion_retrieval(query_track_id, id_to_index, feature_matrices, track_ids, catalog_df, N=10, alpha=0.5):
    """
    Retrieves N tracks using Late Fusion by combining similarities from MFCC and VGG19 retrieval systems.

    Parameters:
    - query_track_id (str): The ID of the query track.
    - id_to_index (dict): Mapping from track ID to index in feature_matrix1 and feature_matrix2.
    - feature_matrices (dict): Dictionary containing 'MFCC Retrieval' and 'VGG19 Retrieval' feature matrices.
    - track_ids (list): List of all track IDs.
    - catalog_df (pd.DataFrame): DataFrame containing track metadata.
    - N (int): Number of tracks to retrieve.
    - alpha (float): Weight parameter for the first similarity score; beta is implicitly (1 - alpha).

    Returns:
    - retrieved_tracks_df (pd.DataFrame): DataFrame of retrieved tracks with aggregated similarity scores.
    """
    if query_track_id not in id_to_index:
        return pd.DataFrame()

    query_index = id_to_index[query_track_id]

    feature_matrix1 = feature_matrices.get('MFCC Retrieval')
    if feature_matrix1 is None:
        print("Error: 'MFCC Retrieval' feature matrix not found.")
        return pd.DataFrame()
    query_vector1 = feature_matrix1[query_index]
    similarities1 = cosine_similarity(query_vector1, feature_matrix1).flatten()
    similarities1[query_index] = -1

    feature_matrix2 = feature_matrices.get('VGG19 Retrieval')
    if feature_matrix2 is None:
        print("Error: 'VGG19 Retrieval' feature matrix not found.")
        return pd.DataFrame()
    query_vector2 = feature_matrix2[query_index]
    similarities2 = cosine_similarity(query_vector2, feature_matrix2).flatten()
    similarities2[query_index] = -1

    #weighted average of similarities
    aggregated_similarities = alpha * similarities1 + (1 - alpha) * similarities2

    top_indices = aggregated_similarities.argsort()[-N:][::-1]

    retrieved_ids = [track_ids[i] for i in top_indices]
    retrieved_scores = [aggregated_similarities[i] for i in top_indices]

    retrieved_tracks_df = pd.DataFrame({
        'id': retrieved_ids,
        'aggregated_similarity': retrieved_scores
    })

    retrieved_tracks_df = pd.merge(retrieved_tracks_df, catalog_df[['id', 'artist', 'song', 'album_name', 'top_genre']],
                                   on='id', how='left')

    return retrieved_tracks_df


#14. Define IRSs

retrieval_systems = {
    'Random Retrieval': random_retrieval,
    'Tag-Based Retrieval': tag_based_retrieval_cosine,  # Using Cosine Similarity
    'TF-IDF Retrieval': tfidf_retrieval,
    'BERT Retrieval': bert_retrieval,
    'MFCC Retrieval': mfcc_retrieval,
    'Spectral Contrast Retrieval': spectral_contrast_retrieval,
    'VGG19 Retrieval': vgg19_retrieval,
    'ResNet Retrieval': resnet_retrieval,
    'Early Fusion BERT+MFCC Retrieval': early_fusion_retrieval,
    'Late Fusion MFCC+VGG19 Retrieval': late_fusion_retrieval
}

#15. Prepare Track IDs and Index Mapping

track_ids = catalog_df_filtered['id'].tolist()
id_to_index = {track_id: idx for idx, track_id in enumerate(track_ids)}


#16. evaluation Metrics

def precision_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Computes Precision@k.
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    precision = len(retrieved_set & relevant_set) / k
    return precision


def recall_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Computes Recall@k.
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
    return recall


def ndcg_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Computes NDCG@k.
    """
    dcg = 0.0
    for i, track_id in enumerate(retrieved_ids[:k]):
        if track_id in relevant_ids:
            dcg += 1 / np.log2(i + 2)

    ideal_relevant = min(len(relevant_ids), k)
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(ideal_relevant)])

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr_metric(retrieved_ids, relevant_ids):
    """
    Computes Mean Reciprocal Rank (MRR).
    """
    for rank, track_id in enumerate(retrieved_ids, start=1):
        if track_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_cov_at_n(all_retrieved_ids, catalog_df, N=10):
    """
    Computes Coverage@N: Percentage of songs that appear in at least one retrieval list.
    """
    flattened = [track_id for sublist in all_retrieved_ids for track_id in sublist[:N]]
    unique_retrieved = set(flattened)
    total_tracks = len(catalog_df)
    coverage = (len(unique_retrieved) / total_tracks) * 100
    return coverage


def compute_div_at_n(all_retrieved_tags, N=10):
    """
    Computes Diversity@N: Average number of unique tag occurrences among retrieved songs.
    """
    diversity_scores = []
    for tags in all_retrieved_tags:
        top_n_tags = tags[:N]
        unique_tags = set(top_n_tags)
        diversity_scores.append(len(unique_tags))
    average_diversity = np.mean(diversity_scores) if diversity_scores else 0
    return average_diversity


def compute_avg_pop_at_n(all_retrieved_popularity, N=10):
    """
    Computes AvgPop@N: Average popularity of retrieved songs.
    """
    avg_popularity_scores = []
    for pops in all_retrieved_popularity:
        top_n_pops = pops[:N]
        if top_n_pops:
            avg_popularity_scores.append(np.mean(top_n_pops))
    average_popularity = np.mean(avg_popularity_scores) if avg_popularity_scores else 0
    return average_popularity


def compute_popularity_diversity(all_retrieved_popularity, N=10):
    """
    Computes Popularity Diversity@N: Variance of popularity scores among retrieved songs.
    """
    diversity_scores = []
    for pops in all_retrieved_popularity:
        top_n_pops = pops[:N]
        if len(top_n_pops) > 1:
            diversity_scores.append(np.var(top_n_pops))
        else:
            diversity_scores.append(0)
    average_diversity = np.mean(diversity_scores) if diversity_scores else 0
    return average_diversity


#17. evaluation functions

def evaluate_retrieval_system(
        catalog_df,
        track_ids,
        id_to_index,
        retrieval_function,
        feature_matrix=None,
        N=10,
        retrieval_system_name='',
        feature_matrices=None,
        alpha=0.5,
        relevance_definition='top_genre'
):
    """
    Evaluates a retrieval system, computing both accuracy and beyond-accuracy metrics based on the relevance definition.

    Parameters:
    - catalog_df (pd.DataFrame): DataFrame containing all tracks with 'id', 'tags', 'popularity', 'top_genre', and 'genre'.
    - track_ids (list): List of track IDs.
    - id_to_index (dict): Mapping from track ID to index.
    - retrieval_function (function): The specific retrieval function for the IR system.
    - feature_matrix (csr_matrix, optional): Feature matrix used by the retrieval function.
    - N (int): Number of tracks to retrieve per query.
    - retrieval_system_name (str): Name of the retrieval system (for logging).
    - feature_matrices (dict, optional): Additional feature matrices for late fusion.
    - alpha (float): Weight parameter for late fusion.
    - relevance_definition (str): 'top_genre' or 'tag_overlap'

    Returns:
    - metrics (dict): Dictionary containing all evaluation metrics.
    """
    precisions = []
    recalls = []
    ndcgs = []
    mrrs = []

    all_retrieved_ids = []
    all_retrieved_tags = []
    all_retrieved_popularity = []

    total_queries = len(catalog_df)
    processed_queries = 0

    start_time = time.time()

    for index, query_track in catalog_df.iterrows():
        query_id = query_track['id']

        #relevance based on the chosen relevance_definition
        if relevance_definition == 'top_genre':
            query_genre = query_track['top_genre']
            if not query_genre:
                continue  #skip if no genre
            relevant_ids = catalog_df[catalog_df['top_genre'] == query_genre]['id'].tolist()
        elif relevance_definition == 'tag_overlap':
            query_tags = set(query_track['filtered_processed_tags_final'])
            relevant_ids = catalog_df[catalog_df['filtered_processed_tags_final'].apply(
                lambda tags: len(query_tags.intersection(tags)) >= 3)]['id'].tolist()
        else:
            print(f"Unknown relevance_definition: {relevance_definition}")
            return {}

        #retrieval
        if retrieval_system_name == 'Late Fusion MFCC+VGG19 Retrieval' and feature_matrices:
            retrieved_tracks = retrieval_function(
                query_track_id=query_id,
                id_to_index=id_to_index,
                feature_matrices=feature_matrices,
                track_ids=track_ids,
                catalog_df=catalog_df,
                N=N,
                alpha=alpha
            )
        elif retrieval_system_name == 'Early Fusion BERT+MFCC Retrieval' and feature_matrix is not None:
            retrieved_tracks = retrieval_function(
                query_track_id=query_id,
                id_to_index=id_to_index,
                feature_matrix=feature_matrix,
                track_ids=track_ids,
                catalog_df=catalog_df,
                N=N
            )
        elif retrieval_system_name == 'Random Retrieval':
            retrieved_tracks = retrieval_function(
                query_track_id=query_id,
                catalog_df=catalog_df,
                N=N
            )
        else:
            #systems that require a feature matrix
            retrieved_tracks = retrieval_function(
                query_track_id=query_id,
                id_to_index=id_to_index,
                feature_matrix=feature_matrix,
                track_ids=track_ids,
                catalog_df=catalog_df,
                N=N
            )

        if retrieved_tracks.empty:
            continue

        retrieved_ids = retrieved_tracks['id'].tolist()

        #tags and popularity for retrieved tracks
        retrieved_subset = catalog_df[catalog_df['id'].isin(retrieved_ids)]
        retrieved_tags = retrieved_subset['filtered_processed_tags_final'].tolist()
        retrieved_popularity = retrieved_subset['popularity'].tolist()

        #evaluation metrics
        p = precision_at_k(retrieved_ids, relevant_ids, k=N)
        r = recall_at_k(retrieved_ids, relevant_ids, k=N)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k=N)
        rr = mrr_metric(retrieved_ids, relevant_ids)

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(ndcg)
        mrrs.append(rr)

        #data for beyond-accuracy metrics
        all_retrieved_ids.append(retrieved_ids)
        #flatten the list of tags for each retrieved song
        flattened_tags = [tag for sublist in retrieved_tags for tag in sublist]
        all_retrieved_tags.append(flattened_tags)
        all_retrieved_popularity.append(retrieved_popularity)

        processed_queries += 1
        if processed_queries % 500 == 0:
            print(f"Processed {processed_queries}/{total_queries} queries")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

    #aggregated accuracy metrics
    accuracy_metrics = {
        'Precision@10': np.mean(precisions) if precisions else 0,
        'Recall@10': np.mean(recalls) if recalls else 0,
        'NDCG@10': np.mean(ndcgs) if ndcgs else 0,
        'MRR': np.mean(mrrs) if mrrs else 0
    }

    #beyond-accuracy metrics
    coverage = compute_cov_at_n(all_retrieved_ids, catalog_df, N)
    tag_diversity = compute_div_at_n(all_retrieved_tags, N)
    popularity_diversity = compute_popularity_diversity(all_retrieved_popularity, N)
    avg_popularity = compute_avg_pop_at_n(all_retrieved_popularity, N)

    metrics = {
        **accuracy_metrics,
        'Coverage@10': coverage,
        'Tag Diversity@10': tag_diversity,
        'Popularity Diversity@10': popularity_diversity,
        'AvgPop@10': avg_popularity
    }

    return metrics


#18. avg number of tags and genres for baseline comparison

avg_tags_per_track = catalog_df_filtered['filtered_processed_tags_final'].apply(len).mean()
avg_genres_per_track = catalog_df_filtered['genre'].apply(len).mean()
print(f"\nAverage number of significant tags per track: {avg_tags_per_track:.2f}")
print(f"Average number of genres per track: {avg_genres_per_track:.2f}")

#19. relevant objects and functions

__all__ = [
    'catalog_df_filtered',
    'retrieval_systems',
    'feature_matrices',
    'track_ids',
    'id_to_index',
    'tfidf_vectorizer',
    'binary_vectorizer',
    'early_fusion_retrieval',
    'late_fusion_retrieval',
    'evaluate_retrieval_system',
    'precision_at_k',
    'recall_at_k',
    'ndcg_at_k',
    'mrr_metric'
]
