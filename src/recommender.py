"""Content-based recommender: TF-IDF on description, genres, cast, director + cosine similarity."""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        text = (
            self.df["description"].fillna("") + " "
            + self.df["listed_in"].fillna("").str.replace(",", " ").str.replace(" ", "_") * 2 + " "
            + self.df["director"].fillna("").str.replace(" ", "_").str.replace(",_", " ") + " "
            + self.df["cast"].fillna("").str.replace(" ", "_").str.replace(",_", " ")
        )
        self.matrix = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5).fit_transform(text)
        self._index = pd.Series(self.df.index, index=self.df["title"].str.lower()).groupby(level=0).first()

    def recommend(self, title: str, n: int = 5) -> pd.DataFrame:
        key = title.lower()
        if key not in self._index:
            raise KeyError(f"'{title}' not found in the catalogue")
        i = self._index[key]
        sims = linear_kernel(self.matrix[i], self.matrix).ravel()
        order = sims.argsort()[::-1]
        order = [j for j in order if j != i][:n]
        out = self.df.loc[order, ["title", "type", "release_year", "listed_in"]].copy()
        out["similarity"] = sims[order].round(3)
        return out.reset_index(drop=True)
