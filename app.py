"""Interactive Movie Recommendation Dashboard.

Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import streamlit as st
import pandas as pd
from src.recommender import MovieRecommender

# Page Configuration
st.set_page_config(
    page_title="Movie Recommendation Engine | Sudhir Kumar",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7C3AED;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .rec-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #EDE9FE;
        color: #6D28D9;
        margin-right: 6px;
    }
    .explanation-tag {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🎬 CineMatch: Content-Based Movie Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Discover similar films using TF-IDF natural language processing and cosine similarity across genres, plot synopses, directors, and cast.</div>',
    unsafe_allow_html=True,
)

# Explicit Methodology Clarification
st.info(
    "💡 **ALGORITHM TRANSPARENCY**: This system utilizes **Content-Based Item-to-Item Similarity** (analyzing narrative keywords, genres, directors, and cast attributes). It is **not** a collaborative-filtering or user-history personalized engine, ensuring transparent and deterministic recommendations."
)

recommender = MovieRecommender()
all_titles = recommender.get_all_titles()
all_genres = recommender.get_all_genres()

# Sidebar: Controls
st.sidebar.header("🎯 Movie Selection")
selected_movie = st.sidebar.selectbox("Choose a Movie you love:", all_titles, index=all_titles.index("Interstellar") if "Interstellar" in all_titles else 0)
top_k = st.sidebar.slider("Number of Recommendations", min_value=3, max_value=8, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚡ Popular Quick Picks")
col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    if st.button("🚀 Inception"):
        selected_movie = "Inception"
        st.rerun()
    if st.button("🦇 The Dark Knight"):
        selected_movie = "The Dark Knight"
        st.rerun()
with col_b2:
    if st.button("🔫 Pulp Fiction"):
        selected_movie = "Pulp Fiction"
        st.rerun()
    if st.button("🌸 Spirited Away"):
        selected_movie = "Spirited Away"
        st.rerun()

tab_recommend, tab_catalogue = st.tabs(["🍿 Film Recommendations", "🔍 Browse Complete Catalogue"])

with tab_recommend:
    source_movie = recommender.get_movie_by_title(selected_movie)

    if source_movie:
        st.subheader("Selected Title Profile")
        with st.container():
            col_info1, col_info2 = st.columns([1, 2])
            with col_info1:
                st.markdown(f"### {source_movie['title']} ({source_movie['release_year']})")
                st.write(f"⭐ **Rating:** {source_movie['vote_average']} / 10 | ⏱️ **Runtime:** {source_movie['runtime']} min")
                st.write(f"🎬 **Director:** {source_movie['director']}")
                st.write(f"🎭 **Cast:** {source_movie['cast']}")
                # Render genre badges
                for g in str(source_movie["genres"]).split(","):
                    st.markdown(f'<span class="badge">{g.strip()}</span>', unsafe_allow_html=True)
            with col_info2:
                st.write("#### 📖 Synopsis")
                st.write(source_movie["overview"])
                st.write(f"**Themes / Keywords:** *{source_movie['keywords']}*")

        st.markdown("---")
        st.subheader(f"Top {top_k} Recommended Similar Films")

        try:
            recs = recommender.recommend(selected_movie, top_n=top_k)
            for i, rec in enumerate(recs, 1):
                with st.expander(f"#{i} — {rec['title']} ({rec['release_year']}) • Match: {rec['similarity_pct']}% • Rating: {rec['vote_average']}★", expanded=True):
                    r_col1, r_col2 = st.columns([2, 1])
                    with r_col1:
                        st.write(f"**Why Recommended:**")
                        st.markdown(f'<div class="explanation-tag">💡 {rec["explanation"]}</div>', unsafe_allow_html=True)
                        st.write("")
                        st.write(f"**Synopsis:** {rec['overview']}")
                    with r_col2:
                        st.write(f"**Director:** {rec['director']}")
                        st.write(f"**Cast:** {rec['cast']}")
                        for g in str(rec["genres"]).split(","):
                            st.markdown(f'<span class="badge">{g.strip()}</span>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error fetching recommendations: {e}")

with tab_catalogue:
    st.subheader("Search & Filter the Full Movie Catalogue")
    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        cat_query = st.text_input("Search Title, Director, or Plot Keywords:", placeholder="e.g. Nolan, space, mob")
    with c_col2:
        cat_genre = st.selectbox("Filter by Genre", ["All"] + all_genres)
    with c_col3:
        cat_rating = st.slider("Minimum Rating (★)", 7.0, 9.5, 7.5, 0.1)

    filtered_df = recommender.search_catalogue(query=cat_query, genre=cat_genre, min_rating=cat_rating)
    st.write(f"Found {len(filtered_df)} matching titles:")
    st.dataframe(
        filtered_df[["title", "release_year", "genres", "director", "vote_average", "runtime", "cast"]],
        use_container_width=True,
    )
