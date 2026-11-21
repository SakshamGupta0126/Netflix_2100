import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import clean, explode_column, split_count  # noqa: E402


def _raw():
    return pd.DataFrame({
        "show_id": ["s1", "s2", "s3"],
        "type": ["Movie", "TV Show", "Movie"],
        "title": ["A", "B", "C"],
        "director": ["Dir One", None, "Dir Two"],
        "cast": [None, "X, Y", "Z"],
        "country": ["United States, India", "South Korea", None],
        "date_added": ["September 25, 2021", " January 1, 2019", None],
        "release_year": [2020, 2018, 2015],
        "rating": ["PG-13", "TV-MA", "74 min"],      # row 3: runtime typed into rating
        "duration": ["90 min", "2 Seasons", None],
        "listed_in": ["Dramas, Comedies", "Crime TV Shows", "Documentaries"],
        "description": ["d1", "d2", "d3"],
    })


def test_misplaced_runtime_is_repaired():
    df = clean(_raw())
    assert df.loc[2, "duration"] == "74 min"
    assert pd.isna(df.loc[2, "rating"])
    assert df.loc[2, "duration_min"] == 74


def test_dates_and_derived_columns():
    df = clean(_raw())
    assert df.loc[0, "year_added"] == 2021 and df.loc[0, "month_added"] == 9
    assert df.loc[1, "year_added"] == 2019          # leading space handled
    assert pd.isna(df.loc[2, "year_added"])
    assert df.loc[0, "years_to_netflix"] == 1
    assert df.loc[0, "release_decade"] == 2020


def test_duration_split_by_type():
    df = clean(_raw())
    assert df.loc[0, "duration_min"] == 90 and pd.isna(df.loc[0, "n_seasons"])
    assert df.loc[1, "n_seasons"] == 2 and pd.isna(df.loc[1, "duration_min"])


def test_rating_groups():
    df = clean(_raw())
    assert df.loc[0, "rating_group"] == "Teens"
    assert df.loc[1, "rating_group"] == "Adults"
    assert df.loc[2, "rating_group"] == "Unrated"


def test_list_helpers():
    df = clean(_raw())
    assert list(df["primary_country"].fillna("?")) == ["United States", "South Korea", "?"]
    assert list(split_count(df["country"])) == [2, 1, 0]
    ex = explode_column(df, "country", "nation")
    assert len(ex) == 3 and set(ex["nation"]) == {"United States", "India", "South Korea"}
