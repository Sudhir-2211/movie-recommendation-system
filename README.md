# CineMatch: Content-Based Movie Recommendation Engine

[![Tests](https://img.shields.io/badge/tests-6%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An explainable, content-based movie recommendation system built with **Python, Scikit-learn, Pandas, and Streamlit**. It computes cosine similarity across n-gram TF-IDF vectors of multi-dimensional metadata (genres, plot overviews, keywords, directors, and top cast), provides explicit recommendation rationales (*why* a film was suggested), and features a searchable movie catalogue with dynamic rating and genre filters.

Developed by **[Sudhir Kumar](https://github.com/Sudhir-2211)** as part of the student placement portfolio.

---

## 📌 Table of Contents
- [Algorithm Transparency & Methodology Clarification](#algorithm-transparency--methodology-clarification)
- [Project Overview](#project-overview)
- [Recommendation Architecture & TF-IDF Cosine Similarity](#recommendation-architecture--tf-idf-cosine-similarity)
- [Feature Weights & Metadata Soup](#feature-weights--metadata-soup)
- [Folder Structure](#folder-structure)
- [Installation & Setup](#installation--setup)
- [Interactive Streamlit Web Dashboard](#interactive-streamlit-web-dashboard)
- [Command Line Interface (CLI)](#command-line-interface-cli)
- [Automated Testing](#automated-testing)
- [Limitations & Future Roadmap](#limitations--future-roadmap)
- [License](#license)

---

## 💡 Algorithm Transparency & Methodology Clarification

> **IMPORTANT METHODOLOGY CLARIFICATION**: This recommendation engine is strictly a **Content-Based (Item-to-Item) Filtering system**. It evaluates structural similarities between films (e.g., narrative keywords, directors, cast members, and genre classifications).
>
> It does **not** employ collaborative filtering, matrix factorization (e.g., SVD), or user profiling based on private viewing histories or user-to-user rating patterns. Therefore, recommendations are **objective, deterministic, and metadata-driven**, rather than personalized to individual user consumption habits.

---

## 📖 Project Overview

Discovering cinema that matches one's specific tastes often fails with black-box recommendation algorithms that cannot explain their reasoning.

**CineMatch** solves this by:
1. Constructing a rich multi-feature metadata corpus ("soup") for every movie.
2. Generating TF-IDF vector representations with sublinear term-frequency scaling.
3. Calculating pairwise Cosine Similarity matrices across the entire catalogue.
4. Generating **human-readable rationales** for every recommendation (e.g., *"Directed by Christopher Nolan • Stars Michael Caine • Shares Science Fiction themes"*).
5. Providing an interactive Web UI and terminal CLI for real-time querying.

---

## 🛠️ Recommendation Architecture & TF-IDF Cosine Similarity

```text
  Movie Metadata: Genres, Overview, Keywords, Director, Cast
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 Weighted Metadata "Soup"                    │
  │   genres (2x) + director (2x) + keywords + cast + overview  │
  └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         TfidfVectorizer(stop_words='english', (1, 2))       │
  │         Transforms text soup into sparse TF-IDF vectors     │
  └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │             Cosine Similarity Matrix (N x N)                │
  │    cos_sim(A, B) = (A · B) / (||A|| * ||B||)                │
  └─────────────────────────────────────────────────────────────┘
                            │
                            ▼
  Query Title ➔ Rank Top N Matches ➔ Generate Explanations ➔ Deliver
```

---

## ⚖️ Feature Weights & Metadata Soup

To ensure recommendations capture the stylistic essence of a film:
- **Director & Auteur**: Repeated $\times 2$ in the text soup so that signature directorial style (e.g., Christopher Nolan, Quentin Tarantino, Denis Villeneuve) is prioritized.
- **Genres**: Repeated $\times 2$ to prevent cross-genre mismatches.
- **Keywords & Plot**: Provide fine-grained thematic matching (e.g., "space exploration", "organized crime", "heist").
- **Cast**: Links films sharing ensemble actors (e.g., Leonardo DiCaprio, Brad Pitt, Tom Hanks).

---

## 📁 Folder Structure

```text
movie-recommendation-system/
├── .gitignore                     # Git ignore file
├── LICENSE                        # MIT License
├── README.md                      # Comprehensive documentation
├── requirements.txt               # Dependencies
├── app.py                         # Streamlit interactive recommendation app
├── recommend.py                   # Command-line interface recommendation tool
├── data/
│   └── movies.csv                 # Curated international film metadata catalogue
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Metadata generator script
│   └── recommender.py             # TF-IDF cosine similarity recommendation engine
└── tests/
    ├── __init__.py
    └── test_recommender.py        # Automated unit tests
```

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sudhir-2211/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Interactive Streamlit Web Dashboard

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

Open `http://localhost:8501`. Features include:
- Searchable dropdown containing popular films (Interstellar, Inception, The Dark Knight, Pulp Fiction, Spirited Away, etc.).
- Detailed movie showcase card with rating, runtime, director, cast, and synopsis.
- Interactive top N recommendation cards with match percentages and explicit explanation badges.
- Searchable film catalogue with genre and rating sliders.

---

## 💻 Command Line Interface (CLI)

Run recommendations directly from PowerShell or terminal:

```bash
python recommend.py "Interstellar" --top 5
```

Sample output:
```text
======================================================================
  RECOMMENDATIONS BASED ON: INTERSTELLAR (2014)
  Director: Christopher Nolan | Genres: Science Fiction, Adventure, Drama
======================================================================

1. Inception (2010) - [Match: 15.6% | Rating: 8.8/10]
   Director   : Christopher Nolan
   Genres     : Action, Science Fiction, Adventure
   Cast       : Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy
   Why matched: Directed by Christopher Nolan • Shares Science Fiction, Adventure themes
   Plot       : A thief who steals corporate secrets through the use of dream-sharing technology...

2. The Prestige (2006) - [Match: 13.8% | Rating: 8.5/10]
   Director   : Christopher Nolan
   Genres     : Drama, Mystery, Science Fiction, Thriller
   Cast       : Hugh Jackman, Christian Bale, Michael Caine, Scarlett Johansson
   Why matched: Directed by Christopher Nolan • Stars Michael Caine • Shares Science Fiction, Drama themes
   Plot       : After a tragic accident, two stage magicians in 1890s London engage in a battle...
```

List all available titles:
```bash
python recommend.py --list
```

---

## 🧪 Automated Testing

Unit tests validate:
- Catalogue initialization and schema integrity.
- Recommendation output structure and positive similarity bounds.
- Query movie exclusion (a film is never recommended to itself).
- Thematic consistency (Sci-Fi query yields Sci-Fi recommendations; Mafia query yields Mafia recommendations).
- Case-insensitivity and graceful error handling on unknown titles.
- Catalogue search and rating threshold filters.

Run the test suite:
```bash
python -m pytest -v
```

Expected output:
```text
============================== 6 passed in 1.27s ==============================
```

---

## ⚠️ Limitations & Future Roadmap

- **Current Limitations**: Purely content-based; does not capture real-time social popularity trends or user-specific rating histories.
- **Future Roadmap**:
  - Incorporate modern Sentence-Transformers (e.g., `all-MiniLM-L6-v2`) for dense semantic embeddings.
  - Implement hybrid recommendation combining content vectors with collaborative filtering matrix factorization (Surprise / LightFM).
  - Add high-resolution TMDB movie poster API fetching.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
