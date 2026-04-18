import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Hindi Songs Recommendation System",
    page_icon="🎵",
    layout="centered"
)


@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)


    for col in ["title", "artist", "mood", "tags", "lyrics"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""  

    
    def combine_features(row):
        return (
            str(row["title"]) + " " +
            str(row["artist"]) + " " +
            str(row["mood"]) + " " +
            str(row["tags"]) + " " +
            str(row["lyrics"])
        )

    df["combined_text"] = df.apply(combine_features, axis=1)
    return df

@st.cache_resource
def build_model(df: pd.DataFrame):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return vectorizer, tfidf_matrix, cosine_sim_matrix

def get_index_from_title(df: pd.DataFrame, title: str):
    result = df[df["title"].str.lower() == title.lower()]
    if result.empty:
        return None
    return result.index[0]

def recommend_by_song(df, cosine_sim_matrix, song_title, top_n=5):
    idx = get_index_from_title(df, song_title)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    # sort by similarity score (high -> low)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    count = 0
    for i, score in sim_scores:
        if i == idx:
            continue  
        song = df.iloc[i]
        recommendations.append({
            "Title": song["title"],
            "Artist": song["artist"],
            "Mood": song["mood"],
            "Tags": song["tags"],
            "Similarity": round(float(score), 3)
        })
        count += 1
        if count >= top_n:
            break
    return recommendations

def recommend_by_query(df, vectorizer, tfidf_matrix, query, top_n=5):
    if not query.strip():
        return []

    query_vec = vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    recommendations = []
    for idx in top_indices:
        song = df.iloc[idx]
        score = sim_scores[idx]
        recommendations.append({
            "Title": song["title"],
            "Artist": song["artist"],
            "Mood": song["mood"],
            "Tags": song["tags"],
            "Similarity": round(float(score), 3)
        })
    return recommendations

#streamlit

st.title("🎵 Hindi Songs Recommendation System")

# Load data + model
df = load_data("hindi_songs.csv")
vectorizer, tfidf_matrix, cosine_sim_matrix = build_model(df)

tab1, tab2 = st.tabs(["🔁 Recommend by Song", "📝 Recommend by Mood"])


with tab1:
    st.subheader("Recommend based on a selected song")

    
    song_list = df["title"].tolist()
    selected_song = st.selectbox(
        "Select a song from the list:",
        ["-- Select --"] + song_list
    )

    
    top_n_song = st.slider(
        "Number of recommendations",
        min_value=1, max_value=10, value=5
    )

    
    if st.button("Get Recommendations (by Song)"):
        if selected_song == "-- Select --":
            st.warning("Please select a valid song.")
        else:
            recs = recommend_by_song(
                df, cosine_sim_matrix, selected_song, top_n_song
            )

            if not recs:
                st.info("No similar songs found.")
            else:
                st.success(
                    f"Because you listened to **{selected_song}** you might also like:"
                )
                table = pd.DataFrame(recs)
                table.index = table.index + 1  
                st.table(table)


with tab2:
    st.subheader("Recommend based on mood")

    query = st.text_input("Type your mood:")
    top_n_query = st.slider(
        "Number of recommendations",
        min_value=1, max_value=10, value=5,
        key="query_slider"
    )

    if st.button("Get Recommendations (by Query)"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            recs = recommend_by_query(
                df, vectorizer, tfidf_matrix, query, top_n_query
            )

            if not recs:
                st.info("No results found. Try a different query.")
            else:
                st.success(f"Results for query: **{query}**")
                table = pd.DataFrame(recs)
                table.index = table.index + 1  # start numbering at 1
                st.table(table)
