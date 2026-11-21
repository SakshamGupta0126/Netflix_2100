"""Analysis functions. Each returns a DataFrame / dict so they are easy to test and reuse."""
from __future__ import annotations

from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .cleaning import GROUP_ORDER, explode_column


# --------------------------------------------------------------------------- overview
def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column (count and %), sorted high to low."""
    miss = df.isna().sum()
    out = pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(2)})
    return out[out["missing"] > 0].sort_values("missing", ascending=False)


def type_split(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["type"].value_counts()
    return pd.DataFrame({"titles": counts, "pct": (counts / counts.sum() * 100).round(1)})


# --------------------------------------------------------------------------- time
def additions_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Titles added per calendar year, split by type."""
    return (
        df.dropna(subset=["year_added"])
        .groupby(["year_added", "type"]).size().unstack(fill_value=0)
        .rename_axis(columns=None)
    )


def additions_by_month(df: pd.DataFrame) -> pd.Series:
    """Titles added per month, summed over 2016-2020 (the full years of heavy adding)."""
    sub = df[df["year_added"].between(2016, 2020)]
    return sub.groupby("month_added").size()


def release_vs_added(df: pd.DataFrame) -> pd.DataFrame:
    """How old is a title when it reaches Netflix? (years between release and addition)."""
    sub = df.dropna(subset=["years_to_netflix"]).copy()
    sub["years_to_netflix"] = sub["years_to_netflix"].astype(int)
    sub = sub[sub["years_to_netflix"] >= 0]
    return sub.groupby("type")["years_to_netflix"].describe()[["count", "mean", "50%", "75%", "max"]]


def catalog_age_by_added_year(df: pd.DataFrame) -> pd.DataFrame:
    """Median age of titles at the time they were added, per added year and type."""
    sub = df.dropna(subset=["years_to_netflix"]).copy()
    sub["years_to_netflix"] = sub["years_to_netflix"].astype(int)
    sub = sub[(sub["years_to_netflix"] >= 0) & (sub["year_added"] >= 2012)]
    return sub.groupby(["year_added", "type"])["years_to_netflix"].median().unstack()


# --------------------------------------------------------------------------- geography
def top_countries(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    exp = explode_column(df, "country", "nation")
    tab = exp.groupby(["nation", "type"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(axis=1)
    return tab.sort_values("total", ascending=False).head(n)


def country_genre_matrix(df: pd.DataFrame, n_countries: int = 8, n_genres: int = 10) -> pd.DataFrame:
    """Share (%) of each country's titles that carry each genre."""
    ex = explode_column(explode_column(df, "country", "nation"), "listed_in", "genre")
    top_c = ex["nation"].value_counts().head(n_countries).index
    top_g = ex["genre"].value_counts().head(n_genres).index
    n_titles = explode_column(df, "country", "nation").groupby("nation").size()
    cnt = ex[ex["nation"].isin(top_c) & ex["genre"].isin(top_g)].groupby(["nation", "genre"]).size().unstack(fill_value=0)
    share = cnt.div(n_titles.loc[cnt.index], axis=0) * 100
    return share.loc[top_c, top_g].round(1)


# --------------------------------------------------------------------------- genres
def genre_counts(df: pd.DataFrame) -> pd.DataFrame:
    exp = explode_column(df, "listed_in", "genre")
    tab = exp.groupby(["genre", "type"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(axis=1)
    return tab.sort_values("total", ascending=False)


def genre_cooccurrence(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """How often two genres are tagged on the same title (top-n genres)."""
    lists = df["listed_in"].str.split(",").apply(lambda xs: sorted({x.strip() for x in xs}))
    top = [g for g, _ in Counter(g for xs in lists for g in xs).most_common(n)]
    mat = pd.DataFrame(0, index=top, columns=top, dtype=int)
    for xs in lists:
        xs = [g for g in xs if g in top]
        for g in xs:
            mat.loc[g, g] += 1
        for a, b in combinations(xs, 2):
            mat.loc[a, b] += 1
            mat.loc[b, a] += 1
    return mat


# --------------------------------------------------------------------------- ratings
def rating_counts(df: pd.DataFrame) -> pd.DataFrame:
    tab = df.dropna(subset=["rating"]).groupby(["rating", "type"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(axis=1)
    return tab.sort_values("total", ascending=False)


def maturity_share_by_year(df: pd.DataFrame, start: int = 2013) -> pd.DataFrame:
    """% of each year's additions in each maturity group (rated titles only)."""
    sub = df[(df["year_added"] >= start) & (df["rating_group"] != "Unrated")]
    tab = sub.groupby(["year_added", "rating_group"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=[g for g in GROUP_ORDER if g in tab.columns])
    return tab.div(tab.sum(axis=1), axis=0) * 100


# --------------------------------------------------------------------------- duration
def movie_runtime_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    mv = df[(df["type"] == "Movie") & df["duration_min"].notna() & (df["release_year"] >= 1960)]
    out = mv.groupby("release_decade")["duration_min"].agg(["count", "median", "mean"]).round(1)
    return out


def tv_seasons(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["type"] == "TV Show", "n_seasons"].value_counts().sort_index()


# --------------------------------------------------------------------------- people
def top_people(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    exp = explode_column(df, col, "person")
    tab = exp.groupby(["person", "type"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(axis=1)
    return tab.sort_values("total", ascending=False).head(n)


# --------------------------------------------------------------------------- statistics
def statistical_tests(df: pd.DataFrame) -> dict:
    """A few hypothesis tests that back the descriptive findings."""
    out: dict = {}
    mv = df[(df["type"] == "Movie") & df["duration_min"].notna()]

    # Has movie runtime changed with release year?
    rho, p = stats.spearmanr(mv["release_year"], mv["duration_min"])
    out["runtime_vs_release_year_spearman"] = {"rho": round(float(rho), 3), "p": float(p), "n": int(len(mv))}

    # Are adult-rated movies longer than family/kids movies?
    adult = mv.loc[mv["rating_group"] == "Adults", "duration_min"]
    young = mv.loc[mv["rating_group"].isin(["Kids", "Family"]), "duration_min"]
    u, p = stats.mannwhitneyu(adult, young, alternative="greater")
    out["adult_vs_kids_family_runtime_mannwhitney"] = {
        "adult_median": float(adult.median()), "kids_family_median": float(young.median()),
        "p": float(p), "n_adult": int(len(adult)), "n_kids_family": int(len(young)),
    }

    # Is content type (Movie/TV) associated with maturity group?
    ct = pd.crosstab(df["type"], df["rating_group"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    out["type_vs_rating_group_chi2"] = {"chi2": round(float(chi2), 1), "dof": int(dof), "p": float(p), "cramers_v": round(cramers_v, 3)}
    return out


# --------------------------------------------------------------------------- text mining
def description_terms(df: pd.DataFrame, n: int = 15) -> dict[str, pd.Series]:
    """Most distinctive description words per content type (mean TF-IDF weight)."""
    vec = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.5, ngram_range=(1, 1))
    X = vec.fit_transform(df["description"])
    terms = np.array(vec.get_feature_names_out())
    res = {}
    for t in ["Movie", "TV Show"]:
        mask = (df["type"] == t).values
        w = np.asarray(X[mask].mean(axis=0)).ravel()
        res[t] = pd.Series(w, index=terms).sort_values(ascending=False).head(n)
    return res


def topic_model(df: pd.DataFrame, k: int = 8, n_words: int = 8, seed: int = 42):
    """NMF topic model on descriptions. Returns (topic summary, per-title dominant topic)."""
    vec = TfidfVectorizer(stop_words="english", min_df=10, max_df=0.4)
    X = vec.fit_transform(df["description"])
    nmf = NMF(n_components=k, random_state=seed, init="nndsvd", max_iter=400)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    dominant = W.argmax(axis=1)
    rows = []
    for i, comp in enumerate(nmf.components_):
        top = terms[comp.argsort()[::-1][:n_words]]
        mask = dominant == i
        rows.append({
            "topic": i,
            "top_words": ", ".join(top),
            "titles": int(mask.sum()),
            "pct_tv": round(float((df["type"].values[mask] == "TV Show").mean() * 100), 1),
        })
    return pd.DataFrame(rows).sort_values("titles", ascending=False), pd.Series(dominant, index=df.index)


# --------------------------------------------------------------------------- headline metrics
def headline_metrics(df: pd.DataFrame) -> dict:
    """Numbers quoted in the README / insights, computed once so they never go stale."""
    by_year = additions_by_year(df)
    peak_year = int(by_year.sum(axis=1).idxmax())
    mv = df[df["type"] == "Movie"]
    tv = df[df["type"] == "TV Show"]
    countries = top_countries(df, 5)
    n_with_country = df["country"].notna().sum()
    us_titles = int(explode_column(df, "country", "nation").query("nation == 'United States'").shape[0])
    return {
        "n_titles": int(len(df)),
        "n_movies": int(len(mv)),
        "n_tv_shows": int(len(tv)),
        "pct_movies": round(len(mv) / len(df) * 100, 1),
        "date_range_added": [str(df["date_added"].min().date()), str(df["date_added"].max().date())],
        "release_year_range": [int(df["release_year"].min()), int(df["release_year"].max())],
        "peak_year_added": peak_year,
        "peak_year_titles": int(by_year.sum(axis=1).max()),
        "pct_added_2016_2020": round(float(df["year_added"].between(2016, 2020).mean() * 100), 1),
        "median_movie_runtime_min": float(mv["duration_min"].median()),
        "pct_tv_single_season": round(float((tv["n_seasons"] == 1).mean() * 100), 1),
        "max_seasons": int(tv["n_seasons"].max()),
        "top5_countries": {k: int(v) for k, v in countries["total"].items()},
        "us_share_of_country_tagged_pct": round(us_titles / n_with_country * 100, 1),
        "pct_missing_country": round(float(df["country"].isna().mean() * 100), 1),
        "pct_missing_director": round(float(df["director"].isna().mean() * 100), 1),
        "pct_adult_rated_of_rated": round(float((df.loc[df["rating_group"] != "Unrated", "rating_group"] == "Adults").mean() * 100), 1),
        "median_years_to_netflix": float(df["years_to_netflix"].dropna().astype(int).median()),
    }
