"""Content-Based Movie Recommendation Engine.

Uses TF-IDF Vectorization and Cosine Similarity across movie metadata
(genres, plot overview, keywords, director, and cast) with explicit recommendation rationales.
Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """Content-based recommendation engine for movies."""

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "movies.csv")

        self.df = pd.read_csv(data_path)
        self._build_similarity_matrix()

    def _build_similarity_matrix(self):
        """Constructs combined metadata tags and computes cosine similarity."""
        # Weighted tag combination: genres and director are amplified for strong thematic coherence
        def create_soup(row):
            genres_rep = (" " + str(row["genres"]).replace(",", " ")) * 2
            director_rep = (" " + str(row["director"])) * 2
            keywords = " " + str(row["keywords"]).replace(",", " ")
            cast = " " + str(row["cast"]).replace(",", " ")
            overview = " " + str(row["overview"])
            return (genres_rep + director_rep + keywords + cast + overview).lower()

        self.df["soup"] = self.df.apply(create_soup, axis=1)

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["soup"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Mapping for fast lookup
        self.indices = pd.Series(self.df.index, index=self.df["title"].str.lower()).drop_duplicates()

    def get_all_titles(self) -> List[str]:
        """Returns list of all available movie titles in alphabetical order."""
        return sorted(self.df["title"].tolist())

    def get_all_genres(self) -> List[str]:
        """Returns unique list of genres present in catalogue."""
        genres_set = set()
        for g_str in self.df["genres"].dropna():
            for g in str(g_str).split(","):
                genres_set.add(g.strip())
        return sorted(list(genres_set))

    def get_movie_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete metadata for a given movie title."""
        key = title.strip().lower()
        if key not in self.indices:
            return None
        idx = self.indices[key]
        return self.df.iloc[idx].to_dict()

    def _generate_explanation(self, source_movie: pd.Series, target_movie: pd.Series, score: float) -> str:
        """Constructs human-readable explanation of why target_movie was recommended."""
        reasons = []

        # 1. Director Match
        if source_movie["director"] == target_movie["director"]:
            reasons.append(f"Directed by {source_movie['director']}")

        # 2. Shared Cast Members
        src_cast = set(c.strip() for c in str(source_movie["cast"]).split(","))
        tgt_cast = set(c.strip() for c in str(target_movie["cast"]).split(","))
        shared_actors = src_cast.intersection(tgt_cast)
        if shared_actors:
            reasons.append(f"Stars {', '.join(list(shared_actors)[:2])}")

        # 3. Shared Genres
        src_genres = set(g.strip() for g in str(source_movie["genres"]).split(","))
        tgt_genres = set(g.strip() for g in str(target_movie["genres"]).split(","))
        shared_genres = src_genres.intersection(tgt_genres)
        if shared_genres:
            reasons.append(f"Shares {', '.join(list(shared_genres)[:3])} themes")

        if not reasons:
            reasons.append("High thematic narrative similarity in plot and style")

        return " • ".join(reasons)

    def recommend(self, title: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Finds top N most similar movies to the given title with rationales."""
        key = title.strip().lower()
        if key not in self.indices:
            raise KeyError(f"Movie '{title}' was not found in the catalogue.")

        idx = self.indices[key]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity score descending (excluding the movie itself at index idx)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

        source_movie = self.df.iloc[idx]
        recommendations = []

        for movie_idx, score in sim_scores:
            target_movie = self.df.iloc[movie_idx]
            explanation = self._generate_explanation(source_movie, target_movie, score)

            recommendations.append({
                "movie_id": int(target_movie["movie_id"]),
                "title": target_movie["title"],
                "genres": target_movie["genres"],
                "director": target_movie["director"],
                "cast": target_movie["cast"],
                "release_year": int(target_movie["release_year"]),
                "vote_average": float(target_movie["vote_average"]),
                "runtime": int(target_movie["runtime"]),
                "overview": target_movie["overview"],
                "similarity_score": round(float(score), 4),
                "similarity_pct": round(float(score) * 100.0, 1),
                "explanation": explanation,
            })

        return recommendations

    def search_catalogue(
        self,
        query: Optional[str] = None,
        genre: Optional[str] = None,
        min_rating: float = 0.0,
        decade: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filters catalogue by title keywords, genre, minimum rating, and decade."""
        filtered = self.df.copy()

        if query and query.strip():
            q = query.strip().lower()
            filtered = filtered[
                filtered["title"].str.lower().str.contains(q) |
                filtered["overview"].str.lower().str.contains(q) |
                filtered["director"].str.lower().str.contains(q)
            ]

        if genre and genre != "All":
            filtered = filtered[filtered["genres"].str.contains(genre, case=False, na=False)]

        if min_rating > 0.0:
            filtered = filtered[filtered["vote_average"] >= min_rating]

        if decade and decade != "All":
            start_year = int(decade.split("s")[0])
            filtered = filtered[(filtered["release_year"] >= start_year) & (filtered["release_year"] < start_year + 10)]

        return filtered.sort_values(by="vote_average", ascending=False).reset_index(drop=True)
