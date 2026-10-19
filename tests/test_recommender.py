"""Automated unit tests for Movie Recommendation System."""

import pytest
import os
import pandas as pd
from src.recommender import MovieRecommender


@pytest.fixture(scope="module")
def recommender():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "movies.csv")
    assert os.path.exists(data_path), "Movies dataset must exist"
    return MovieRecommender(data_path=data_path)


def test_catalogue_initialization(recommender):
    titles = recommender.get_all_titles()
    assert len(titles) >= 30
    assert "Interstellar" in titles
    assert "The Godfather" in titles
    assert "Pulp Fiction" in titles

    genres = recommender.get_all_genres()
    assert "Science Fiction" in genres
    assert "Crime" in genres
    assert "Drama" in genres


def test_recommendation_output_structure(recommender):
    recs = recommender.recommend("Inception", top_n=4)
    assert len(recs) == 4
    for r in recs:
        assert r["title"] != "Inception"  # Query movie excluded
        assert 0.0 <= r["similarity_pct"] <= 100.0
        assert "explanation" in r
        assert len(r["explanation"]) > 0
        assert "director" in r
        assert "vote_average" in r


def test_sci_fi_coherence(recommender):
    # Interstellar recommendations should feature related Sci-Fi / Space / Nolan movies
    recs = recommender.recommend("Interstellar", top_n=5)
    rec_titles = [r["title"] for r in recs]
    # Check that at least one of Nolan's or Villeneuve's sci-fi movies appears
    relevant = {"Inception", "Arrival", "Blade Runner 2049", "Dune", "The Prestige", "The Matrix"}
    assert len(set(rec_titles).intersection(relevant)) >= 2


def test_crime_mafia_coherence(recommender):
    # The Godfather should strongly recommend The Godfather Part II and Goodfellas
    recs = recommender.recommend("The Godfather", top_n=3)
    rec_titles = [r["title"] for r in recs]
    assert "The Godfather Part II" in rec_titles or "Goodfellas" in rec_titles


def test_unknown_title_raises_key_error(recommender):
    with pytest.raises(KeyError, match="not found in the catalogue"):
        recommender.recommend("NonExistentFakeMovie12345")


def test_catalogue_search_filters(recommender):
    # Search by director
    nolan_movies = recommender.search_catalogue(query="Christopher Nolan")
    assert len(nolan_movies) >= 4
    assert (nolan_movies["director"] == "Christopher Nolan").all()

    # Filter by genre and rating
    high_rated_crime = recommender.search_catalogue(genre="Crime", min_rating=8.5)
    assert not high_rated_crime.empty
    assert (high_rated_crime["vote_average"] >= 8.5).all()
