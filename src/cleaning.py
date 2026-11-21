"""Load and clean the Netflix titles dataset.

The raw file has a few well-known quirks that this module handles explicitly:

* ``director``, ``cast`` and ``country`` are missing for many titles.
* ``date_added`` is a free-text string (some values have leading spaces).
* ``duration`` mixes minutes (movies) and seasons (TV shows).
* Three rows have the runtime ("74 min") stored in the ``rating`` column.
* ``country``, ``cast``, ``director`` and ``listed_in`` hold comma-separated lists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Audience-maturity buckets used throughout the analysis.
RATING_GROUPS: dict[str, str] = {
    "TV-Y": "Kids",
    "TV-Y7": "Kids",
    "TV-Y7-FV": "Kids",
    "TV-G": "Kids",
    "G": "Kids",
    "TV-PG": "Family",
    "PG": "Family",
    "TV-14": "Teens",
    "PG-13": "Teens",
    "TV-MA": "Adults",
    "R": "Adults",
    "NC-17": "Adults",
    "NR": "Unrated",
    "UR": "Unrated",
}
GROUP_ORDER = ["Kids", "Family", "Teens", "Adults", "Unrated"]

TEXT_COLUMNS = [
    "show_id", "type", "title", "director", "cast", "country",
    "date_added", "rating", "duration", "listed_in", "description",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV exactly as shipped (all text, ``release_year`` numeric)."""
    df = pd.read_csv(path, dtype=str)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="raise").astype(int)
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of ``raw`` with extra derived columns."""
    df = raw.copy()

    # 1. Trim whitespace and turn empty strings into real missing values.
    for col in TEXT_COLUMNS:
        df[col] = df[col].str.strip().replace("", np.nan)

    # 2. Repair rows where the runtime was typed into the rating column.
    misplaced = df["rating"].str.contains("min", na=False)
    df.loc[misplaced, "duration"] = df.loc[misplaced, "rating"]
    df.loc[misplaced, "rating"] = np.nan

    # 3. Dates.
    df["date_added"] = pd.to_datetime(df["date_added"], format="%B %d, %Y", errors="coerce")
    df["year_added"] = df["date_added"].dt.year.astype("Int64")
    df["month_added"] = df["date_added"].dt.month.astype("Int64")
    df["weekday_added"] = df["date_added"].dt.day_name()
    df["years_to_netflix"] = (df["year_added"] - df["release_year"]).astype("Int64")
    df["release_decade"] = (df["release_year"] // 10 * 10).astype(int)

    # 4. Duration: minutes for movies, seasons for TV shows.
    number = df["duration"].str.extract(r"(\d+)")[0].astype(float)
    df["duration_min"] = number.where(df["type"] == "Movie")
    df["n_seasons"] = number.where(df["type"] == "TV Show")

    # 5. Audience maturity.
    df["rating_group"] = df["rating"].map(RATING_GROUPS).fillna("Unrated")

    # 6. Convenience columns for list-like fields.
    df["primary_country"] = df["country"].str.split(",").str[0].str.strip()
    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()
    df["n_countries"] = split_count(df["country"])
    df["n_genres"] = split_count(df["listed_in"])
    df["n_cast"] = split_count(df["cast"])
    df["description_words"] = df["description"].str.split().str.len()

    return df


def split_count(series: pd.Series, sep: str = ",") -> pd.Series:
    """Number of non-empty items in a comma-separated column (0 when missing)."""
    return series.fillna("").apply(lambda s: len([x for x in s.split(sep) if x.strip()]))


def explode_column(df: pd.DataFrame, col: str, name: str | None = None, sep: str = ",") -> pd.DataFrame:
    """One row per item of a comma-separated column.

    A title listed in two countries appears twice, once per country. Rows whose
    value is missing are dropped.
    """
    name = name or col
    out = df.dropna(subset=[col]).copy()
    out[name] = out[col].str.split(sep)
    out = out.explode(name)
    out[name] = out[name].str.strip()
    return out[out[name] != ""]


def load_and_clean(path: str | Path) -> pd.DataFrame:
    return clean(load_raw(path))
