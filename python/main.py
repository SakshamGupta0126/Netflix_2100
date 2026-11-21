"""Run the full pipeline: clean -> analyse -> figures -> reports.

    python main.py
    python main.py --input data/raw/netflix_data.csv --reports reports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import analysis as an
from src import plots
from src.cleaning import load_and_clean
from src.recommender import ContentRecommender

SAMPLE_TITLES = ["Narcos", "Stranger Things", "Dangal", "The Crown", "Inception", "Sacred Games"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/netflix_data.csv")
    ap.add_argument("--processed", default="data/processed/netflix_clean.csv")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    reports = Path(args.reports)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / "tables").mkdir(parents=True, exist_ok=True)
    Path(args.processed).parent.mkdir(parents=True, exist_ok=True)

    print("1/5 Loading and cleaning ...")
    df = load_and_clean(args.input)
    df.to_csv(args.processed, index=False)
    print(f"    {len(df):,} rows -> {args.processed}")

    print("2/5 Computing tables and statistics ...")
    topic_df, dominant = an.topic_model(df)
    tables = {
        "missing_values": an.data_quality(df),
        "additions_by_year": an.additions_by_year(df),
        "top_countries": an.top_countries(df, 20),
        "genres": an.genre_counts(df),
        "ratings": an.rating_counts(df),
        "top_directors": an.top_people(df, "director", 20),
        "top_cast": an.top_people(df, "cast", 20),
        "movie_runtime_by_decade": an.movie_runtime_by_decade(df),
        "country_genre_share_pct": an.country_genre_matrix(df, 8, 10),
        "description_topics": topic_df.set_index("topic"),
    }
    for name, t in tables.items():
        t.to_csv(reports / "tables" / f"{name}.csv")

    metrics = an.headline_metrics(df)
    metrics["statistical_tests"] = an.statistical_tests(df)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("3/5 Drawing figures ...")
    paths = plots.make_all(df, topic_df, reports / "figures")
    print(f"    {len(paths)} figures -> {reports / 'figures'}")

    print("4/5 Building recommender examples ...")
    rec = ContentRecommender(df)
    frames = []
    for title in SAMPLE_TITLES:
        try:
            r = rec.recommend(title, 5)
            r.insert(0, "because_you_watched", title)
            frames.append(r)
        except KeyError:
            print(f"    (skipped, not in data: {title})")
    if frames:
        import pandas as pd
        pd.concat(frames).to_csv(reports / "tables" / "sample_recommendations.csv", index=False)

    print("5/5 Done. Key numbers:")
    for k in ["n_titles", "pct_movies", "peak_year_added", "median_movie_runtime_min", "pct_tv_single_season"]:
        print(f"    {k}: {metrics[k]}")


if __name__ == "__main__":
    main()
