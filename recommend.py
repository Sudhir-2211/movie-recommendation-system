"""Command-Line Interface for Content-Based Movie Recommender.

Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import argparse
from src.recommender import MovieRecommender


def main():
    parser = argparse.ArgumentParser(description="Content-Based Movie Recommendation Engine")
    parser.add_argument("title", nargs="?", default="", help="Movie title to base recommendations on")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--list", action="store_true", help="List all available movie titles in catalogue")

    args = parser.parse_args()
    engine = MovieRecommender()

    if args.list:
        print("\nAvailable Movies in Catalogue:")
        for t in engine.get_all_titles():
            print(f"  • {t}")
        return

    if not args.title:
        print("Please provide a movie title (e.g. python recommend.py 'Interstellar') or use --list.")
        return

    try:
        source = engine.get_movie_by_title(args.title)
        recs = engine.recommend(args.title, top_n=args.top)

        print("\n" + "=" * 70)
        print(f"  RECOMMENDATIONS BASED ON: {source['title'].upper()} ({source['release_year']})")
        print(f"  Director: {source['director']} | Genres: {source['genres']}")
        print("=" * 70)

        for i, r in enumerate(recs, 1):
            print(f"\n{i}. {r['title']} ({r['release_year']}) - [Match: {r['similarity_pct']}% | Rating: {r['vote_average']}/10]")
            print(f"   Director   : {r['director']}")
            print(f"   Genres     : {r['genres']}")
            print(f"   Cast       : {r['cast']}")
            print(f"   Why matched: {r['explanation']}")
            print(f"   Plot       : {r['overview'][:140]}...")

        print("\n" + "=" * 70)
        print("NOTE: Content-based item-item similarity derived from metadata tags.")
        print("=" * 70 + "\n")

    except KeyError as e:
        print(f"\n[ERROR] {e}")
        print("Use 'python recommend.py --list' to see all valid titles.\n")


if __name__ == "__main__":
    main()
