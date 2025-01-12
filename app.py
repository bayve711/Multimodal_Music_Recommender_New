# app.py

import streamlit as st
import pandas as pd
import backend
import time
from scipy.sparse import hstack

#1.History and Results initialization

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'results' not in st.session_state:
    st.session_state['results'] = {}

#2.Title and Description

st.title("Music Information Retrieval System")
st.markdown("""
Welcome to the Music Information Retrieval System. Select a track and one or more retrieval systems to discover similar songs.
""")

#3. Sidebar for selection of track and IRS

st.sidebar.header("Selection Panel")

#search possibility
st.sidebar.subheader("Select a Track")

#mapping from display name to track ID
track_display_to_id = {
    f"{row['artist']} - {row['song']}": row['id']
    for idx, row in backend.catalog_df_filtered[['id', 'artist', 'song']].iterrows()
}

#list of display names
track_display_names = list(track_display_to_id.keys())

#search functionality
search_query = st.sidebar.text_input("Search for a Track:")

if search_query:
    #filter tracks based on search query
    filtered_tracks = [name for name in track_display_names if search_query.lower() in name.lower()]
else:
    filtered_tracks = track_display_names

#selectbox for track selection
selected_track_display = st.sidebar.selectbox(
    "Choose a Track:",
    filtered_tracks
)

#get the corresponding track ID
selected_track_id = track_display_to_id[selected_track_display]

#multiselect IRS
st.sidebar.subheader("Select Information Retrieval Systems")
retrieval_system_options = list(backend.retrieval_systems.keys())
selected_retrieval_systems = st.sidebar.multiselect(
    "Choose IR Systems:",
    retrieval_system_options
)

#4. Main Section, Retrieve Results

st.header("Retrieval Results")

#Query Track's Genre
query_track_genre = \
backend.catalog_df_filtered[backend.catalog_df_filtered['id'] == selected_track_id]['top_genre'].values[0]
st.subheader(f"Query Track: {selected_track_display}")
st.write(f"**Top Genre:** {query_track_genre.capitalize() if isinstance(query_track_genre, str) else 'N/A'}")

#Retrieve Results Button
if st.sidebar.button("Retrieve Results"):
    if not selected_retrieval_systems:
        st.sidebar.error("Please select at least one IR system.")
    else:
        with st.spinner('Retrieving results...'):
            results = {}
            for system in selected_retrieval_systems:
                retrieval_func = backend.retrieval_systems[system]
                try:
                    if system == 'Late Fusion MFCC+VGG19 Retrieval':
                        retrieved = retrieval_func(
                            query_track_id=selected_track_id,
                            id_to_index=backend.id_to_index,
                            feature_matrices={
                                'MFCC Retrieval': backend.feature_matrices['MFCC Retrieval'],
                                'VGG19 Retrieval': backend.feature_matrices['VGG19 Retrieval']
                            },
                            track_ids=backend.track_ids,
                            catalog_df=backend.catalog_df_filtered,
                            N=10,
                            alpha=0.5
                        )
                    elif system == 'Early Fusion BERT+MFCC Retrieval':
                        #Combine BERT and MFCC using hstack
                        combined_feature_matrix = hstack([
                            backend.feature_matrices['BERT Retrieval'],
                            backend.feature_matrices['MFCC Retrieval']
                        ]).tocsr()
                        retrieved = retrieval_func(
                            query_track_id=selected_track_id,
                            id_to_index=backend.id_to_index,
                            feature_matrix=combined_feature_matrix,
                            track_ids=backend.track_ids,
                            catalog_df=backend.catalog_df_filtered,
                            N=10
                        )
                    elif system == 'Random Retrieval':
                        retrieved = retrieval_func(
                            query_track_id=selected_track_id,
                            catalog_df=backend.catalog_df_filtered,
                            N=10
                        )
                    else:
                        feature_matrix = backend.feature_matrices.get(system)
                        if feature_matrix is not None:
                            retrieved = retrieval_func(
                                query_track_id=selected_track_id,
                                id_to_index=backend.id_to_index,
                                feature_matrix=feature_matrix,
                                track_ids=backend.track_ids,
                                catalog_df=backend.catalog_df_filtered,
                                N=10
                            )
                        else:
                            st.error(f"Feature matrix for {system} not found.")
                            retrieved = pd.DataFrame()

                    #store the retrieved DataFrame
                    results[system] = retrieved
                except Exception as e:
                    st.error(f"Error retrieving results for {system}: {e}")
                    results[system] = pd.DataFrame()

            #store results for Qualitative Analysis
            st.session_state['results'] = results

            #display
            for system, df in results.items():
                st.subheader(f"Results from {system}")
                if df.empty:
                    st.write("No results found.")
                else:
                    #which columns to display based on retrieval system
                    display_columns = ['artist', 'song', 'album_name', 'top_genre']
                    if 'similarity' in df.columns:
                        display_columns.append('similarity')
                    if 'distance' in df.columns:
                        display_columns.append('distance')
                    if 'cosine_similarity' in df.columns:
                        display_columns.append('cosine_similarity')
                    if 'aggregated_similarity' in df.columns:
                        display_columns.append('aggregated_similarity')

                    st.dataframe(df[display_columns].fillna('').head(10))

        #history update
        retrieval_record = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'track': selected_track_display,
            'systems': selected_retrieval_systems,
            'results': {k: v.to_dict(orient='records') for k, v in results.items()}
        }
        st.session_state['history'].append(retrieval_record)

#5. sidebar for History Management

st.sidebar.subheader("History")

# View History Button
if st.sidebar.button("View History"):
    history = st.session_state['history']
    if not history:
        st.sidebar.info("No retrieval history available.")
    else:
        for record in reversed(history):
            with st.expander(f"{record['timestamp']} - {record['track']}"):
                st.write(f"**Selected IR Systems:** {', '.join(record['systems'])}")
                for system, retrieved_list in record['results'].items():
                    st.markdown(f"### {system}")
                    if not retrieved_list:
                        st.write("No results found.")
                    else:
                        #list of dicts back to DataFrame
                        df = pd.DataFrame(retrieved_list)
                        display_columns = ['artist', 'song', 'album_name', 'top_genre']
                        if 'similarity' in df.columns:
                            display_columns.append('similarity')
                        if 'distance' in df.columns:
                            display_columns.append('distance')
                        if 'cosine_similarity' in df.columns:
                            display_columns.append('cosine_similarity')
                        if 'aggregated_similarity' in df.columns:
                            display_columns.append('aggregated_similarity')
                        st.dataframe(df[display_columns].fillna('').head(10))

# Clear History Button
if st.sidebar.button("Clear History"):
    st.session_state['history'] = []
    st.sidebar.success("History cleared.")

#6. Main Section: Qualitative Analysis

st.header("Qualitative Analysis")

# Analyze Retrieval Results Button
if selected_retrieval_systems and st.button("Analyze Retrieval Results"):
    if not st.session_state['results']:
        st.info("No retrieval results available for analysis.")
    else:
        with st.spinner('Conducting qualitative analysis...'):
            results = st.session_state['results']
            for system in selected_retrieval_systems:
                #skip RR in qualitative analysis
                if system == 'Random Retrieval':
                    st.subheader(f"Analysis for {system}")
                    st.write("Qualitative analysis is not applicable for Random Retrieval.")
                    continue

                df = results.get(system)
                if df is not None and not df.empty:
                    st.subheader(f"Analysis for {system}")

                    #display Retrieved Tracks
                    st.write("### Retrieved Tracks:")
                    display_columns = ['artist', 'song', 'album_name', 'top_genre']
                    if 'similarity' in df.columns:
                        display_columns.append('similarity')
                    if 'distance' in df.columns:
                        display_columns.append('distance')
                    if 'cosine_similarity' in df.columns:
                        display_columns.append('cosine_similarity')
                    if 'aggregated_similarity' in df.columns:
                        display_columns.append('aggregated_similarity')

                    st.dataframe(df[display_columns].fillna('').head(10))

                    #wip
                    st.write("### Analysis:")
                    for idx, row in df.iterrows():
                        retrieved_track_id = row['id']
                        #query track's genre
                        query_genre = \
                        backend.catalog_df_filtered[backend.catalog_df_filtered['id'] == selected_track_id][
                            'top_genre'].values[0]
                        #retrieved track's genre
                        retrieved_genre = \
                        backend.catalog_df_filtered[backend.catalog_df_filtered['id'] == retrieved_track_id][
                            'top_genre'].values[0]

                        #normalize genres for comparison
                        normalized_query_genre = query_genre.strip().lower() if isinstance(query_genre, str) else ''
                        normalized_retrieved_genre = retrieved_genre.strip().lower() if isinstance(retrieved_genre,
                                                                                                   str) else ''

                        #similarity score
                        if 'similarity' in row and not pd.isnull(row['similarity']):
                            similarity_score = row['similarity']
                            similarity_metric = 'Similarity Score'
                        elif 'cosine_similarity' in row and not pd.isnull(row['cosine_similarity']):
                            similarity_score = row['cosine_similarity']
                            similarity_metric = 'Cosine Similarity'
                        elif 'aggregated_similarity' in row and not pd.isnull(row['aggregated_similarity']):
                            similarity_score = row['aggregated_similarity']
                            similarity_metric = 'Aggregated Similarity'
                        elif 'distance' in row and not pd.isnull(row['distance']):
                            similarity_score = row['distance']
                            similarity_metric = 'Distance Score'
                        else:
                            similarity_score = None
                            similarity_metric = 'N/A'

                        #genre match
                        genre_match = 'Yes' if normalized_retrieved_genre == normalized_query_genre else 'No'

                        #analysis statement only if similarity_score exists
                        if similarity_score is not None:
                            analysis_statement = f"**{row['artist']} - {row['song']}** | **Genre Match:** {genre_match} | **{similarity_metric}:** {similarity_score:.4f}"
                        else:
                            analysis_statement = f"**{row['artist']} - {row['song']}** | **Genre Match:** {genre_match} | **Similarity Metric:** N/A"

                        st.write(analysis_statement)

                    #wip: mb more detailed analysis based on other metadata / features
                else:
                    st.write(f"No results to analyze for {system}.")
