"""All figures. One function per chart; each saves a PNG and returns its path.

Colour rules: Movie = blue, TV Show = orange (fixed for every chart), magnitude = one-hue
blue ramp, text always in neutral ink (never in a series colour).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import analysis as an
from .cleaning import GROUP_ORDER

MOVIE, TV = "#2a78d6", "#eb6834"
TYPE_COLORS = {"Movie": MOVIE, "TV Show": TV}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc2ee", "#2a78d6", "#0b2f66"])
GROUP_COLORS = {"Kids": "#1baf7a", "Family": "#eda100", "Teens": "#2a78d6", "Adults": "#6250d6"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "font.size": 10, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150,
})


def _save(fig, out_dir: Path, name: str) -> Path:
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _hbar_labels(ax, bars, fmt="{:,.0f}"):
    for b in bars:
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(b.get_width()),
                va="center", ha="left", color=INK2, fontsize=9)


# 1 ---------------------------------------------------------------------------
def type_split(df, out):
    t = an.type_split(df)
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    bars = ax.barh(t.index[::-1], t["titles"][::-1], color=[TYPE_COLORS[i] for i in t.index[::-1]], height=0.55)
    for b, (_, r) in zip(bars, t[::-1].iterrows()):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {int(r.titles):,} ({r.pct}%)", va="center", color=INK2)
    ax.set_xlim(0, t["titles"].max() * 1.3)
    ax.set_title("Movies outnumber TV shows more than two to one")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "01_type_split")


# 2 ---------------------------------------------------------------------------
def additions_by_year(df, out):
    t = an.additions_by_year(df)
    t = t[t.index >= 2008]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in ["Movie", "TV Show"]:
        ax.plot(t.index.astype(int), t[col], color=TYPE_COLORS[col], lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=col)
    ax.text(2021.15, 120, "2021 is a partial year\n(data ends 25 Sep)", ha="right", va="bottom", color=INK2, fontsize=9)
    ax.set_title("Netflix's catalogue growth peaked in 2019")
    ax.set_ylabel("Titles added per year")
    ax.set_xticks(range(2008, 2022))
    ax.legend(loc="upper left")
    return _save(fig, out, "02_additions_by_year")


# 3 ---------------------------------------------------------------------------
def release_year_hist(df, out):
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(1940, 2023, 2)
    for t in ["Movie", "TV Show"]:
        ax.hist(df.loc[(df["type"] == t) & (df["release_year"] >= 1940), "release_year"], bins=bins,
                color=TYPE_COLORS[t], alpha=0.85, label=t, edgecolor=SURFACE, linewidth=0.8)
    ax.set_title("The catalogue is overwhelmingly recent: most titles were released after 2010")
    ax.set_xlabel("Release year"); ax.set_ylabel("Titles")
    ax.legend()
    return _save(fig, out, "03_release_year_distribution")


# 4 ---------------------------------------------------------------------------
def monthly_pattern(df, out):
    m = an.additions_by_month(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(MONTHS, [m.get(i, 0) for i in range(1, 13)], color=MOVIE, width=0.65)
    peak = int(m.idxmax()); low = int(m.idxmin())
    ax.set_title(f"Additions are steady through the year: {MONTHS[peak-1]} highest, {MONTHS[low-1]} lowest")
    ax.set_ylabel("Titles added, 2016–2020 combined")
    ax.grid(axis="x", visible=False)
    return _save(fig, out, "04_monthly_pattern")


# 5 ---------------------------------------------------------------------------
def top_countries(df, out):
    t = an.top_countries(df, 12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(t.index, t["Movie"], color=MOVIE, height=0.65, label="Movie", edgecolor=SURFACE, linewidth=1)
    ax.barh(t.index, t["TV Show"], left=t["Movie"], color=TV, height=0.65, label="TV Show", edgecolor=SURFACE, linewidth=1)
    for y, (_, r) in enumerate(t.iterrows()):
        ax.text(r.total, y, f"  {int(r.total):,}", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, t["total"].max() * 1.12)
    ax.set_title("The US dominates production; India is a clear second")
    ax.set_xlabel("Titles (co-productions count once per country)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, out, "05_top_countries")


# 6 ---------------------------------------------------------------------------
def top_genres(df, out):
    g = an.genre_counts(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False)
    for ax, t in zip(axes, ["Movie", "TV Show"]):
        s = g[t].sort_values(ascending=False).head(10)[::-1]
        bars = ax.barh(s.index, s.values, color=TYPE_COLORS[t], height=0.65)
        _hbar_labels(ax, bars)
        ax.set_xlim(0, s.max() * 1.18)
        ax.set_title(f"Top 10 genres: {t}s", fontsize=11)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out, "06_top_genres_by_type")


# 7 ---------------------------------------------------------------------------
def genre_cooccurrence(df, out):
    m = an.genre_cooccurrence(df, 14)
    arr = m.to_numpy(dtype=float).copy()
    np.fill_diagonal(arr, np.nan)
    off = pd.DataFrame(arr, index=m.index, columns=m.columns)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(arr, cmap=SEQ)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(m.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index)
    ax.grid(False)
    vmax = np.nanmax(off.values)
    for i in range(len(m)):
        for j in range(len(m)):
            if i != j and off.iat[i, j] >= vmax * 0.15:
                ax.text(j, i, int(off.iat[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if off.iat[i, j] > vmax * 0.5 else INK)
    ax.set_title("Which genres appear together on the same title?")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Titles sharing both genres")
    return _save(fig, out, "07_genre_cooccurrence")


# 8 ---------------------------------------------------------------------------
def country_genre_heatmap(df, out):
    m = an.country_genre_matrix(df, 8, 10)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(m.values, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, rotation=40, ha="right")
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white" if v > m.values.max() * 0.55 else INK)
    ax.set_title("Genre fingerprint of each country (% of its titles carrying the genre)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% of country's titles")
    return _save(fig, out, "08_country_genre_heatmap")


# 9 ---------------------------------------------------------------------------
def ratings(df, out):
    r = an.rating_counts(df)
    r = r[r["total"] >= 30].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(r.index, r["Movie"], color=MOVIE, height=0.65, label="Movie", edgecolor=SURFACE, linewidth=1)
    ax.barh(r.index, r["TV Show"], left=r["Movie"], color=TV, height=0.65, label="TV Show", edgecolor=SURFACE, linewidth=1)
    ax.set_xlim(0, r["total"].max() * 1.1)
    for y, (_, row) in enumerate(r.iterrows()):
        ax.text(row.total, y, f"  {int(row.total):,}", va="center", color=INK2, fontsize=9)
    ax.set_title("TV-MA and TV-14 account for most of the catalogue")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, out, "09_ratings")


# 10 --------------------------------------------------------------------------
def maturity_trend(df, out):
    s = an.maturity_share_by_year(df, 2013)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(s))
    x = s.index.astype(int)
    for g in s.columns:
        ax.bar(x, s[g], bottom=bottom, color=GROUP_COLORS[g], width=0.7, label=g, edgecolor=SURFACE, linewidth=1.5)
        bottom += s[g].values
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_ylabel("% of titles added that year")
    ax.set_title("Mix of audience maturity among new additions")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    return _save(fig, out, "10_maturity_trend")


# 11 --------------------------------------------------------------------------
def runtime(df, out):
    mv = df[(df["type"] == "Movie") & df["duration_min"].notna()]
    dec = an.movie_runtime_by_decade(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.hist(mv["duration_min"], bins=np.arange(0, 320, 10), color=MOVIE, edgecolor=SURFACE, linewidth=0.8)
    med = mv["duration_min"].median()
    ax.axvline(med, color=INK2, lw=1, ls="--")
    ax.text(med + 4, ax.get_ylim()[1] * 0.92, f"median {med:.0f} min", color=INK2, fontsize=9)
    ax.set_title("Movie runtime distribution"); ax.set_xlabel("Minutes"); ax.set_ylabel("Movies")
    ax = axes[1]
    ax.plot(dec.index, dec["median"], color=MOVIE, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xticks(dec.index); ax.set_xticklabels([f"{d}s" for d in dec.index])
    ax.set_title("Median runtime by release decade"); ax.set_ylabel("Minutes")
    fig.tight_layout()
    return _save(fig, out, "11_movie_runtime")


# 12 --------------------------------------------------------------------------
def tv_seasons(df, out):
    s = an.tv_seasons(df)
    s.index = s.index.astype(int)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(s.index.astype(str), s.values, color=TV, width=0.65)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_title("Most TV shows on Netflix have only one season")
    ax.set_xlabel("Number of seasons"); ax.set_ylabel("TV shows")
    ax.grid(axis="x", visible=False)
    return _save(fig, out, "12_tv_seasons")


# 13 --------------------------------------------------------------------------
def people(df, out):
    d = an.top_people(df, "director", 10).iloc[::-1]
    c = an.top_people(df, "cast", 10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, data, title in zip(axes, [d, c], ["Most frequent directors", "Most frequent cast members"]):
        bars = ax.barh(data.index, data["total"], color=MOVIE, height=0.65)
        _hbar_labels(ax, bars)
        ax.set_xlim(0, data["total"].max() * 1.15)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out, "13_top_people")


# 14 --------------------------------------------------------------------------
def catalog_age(df, out):
    t = an.catalog_age_by_added_year(df)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for col in ["Movie", "TV Show"]:
        ax.plot(t.index.astype(int), t[col], color=TYPE_COLORS[col], lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=col)
    ax.set_title("Median age of a title when Netflix adds it")
    ax.set_ylabel("Years between release and addition")
    ax.set_xticks(t.index.astype(int))
    ax.legend()
    return _save(fig, out, "14_catalog_age_at_addition")


# 15 --------------------------------------------------------------------------
def missing(df, out):
    q = an.data_quality(df)
    q = q[q["pct"] > 0].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(q.index, q["pct"], color=MOVIE, height=0.6)
    for b, (_, r) in zip(bars, q.iterrows()):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {r.pct}%  ({int(r.missing):,})", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, q["pct"].max() * 1.3)
    ax.set_title("Missing values by column (% of 8,807 rows)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "15_missing_values")


# 16 --------------------------------------------------------------------------
def description_terms(df, out):
    terms = an.description_terms(df, 12)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, t in zip(axes, ["Movie", "TV Show"]):
        s = terms[t].iloc[::-1]
        ax.barh(s.index, s.values, color=TYPE_COLORS[t], height=0.65)
        ax.set_title(f"Most distinctive description words: {t}s", fontsize=11)
        ax.set_xlabel("Mean TF-IDF weight")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out, "16_description_terms")


# 17 --------------------------------------------------------------------------
def topics(df, topic_df, out):
    s = topic_df.sort_values("titles")
    labels = [", ".join(w.split(", ")[:4]) for w in s["top_words"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.barh(labels, s["titles"], color=MOVIE, height=0.65)
    _hbar_labels(ax, bars)
    ax.set_xlim(0, s["titles"].max() * 1.15)
    ax.set_title("Eight themes found in title descriptions (NMF topic model)")
    ax.set_xlabel("Titles whose strongest theme this is")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "17_description_topics")


def make_all(df, topic_df, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        type_split(df, out_dir), additions_by_year(df, out_dir), release_year_hist(df, out_dir),
        monthly_pattern(df, out_dir), top_countries(df, out_dir), top_genres(df, out_dir),
        genre_cooccurrence(df, out_dir), country_genre_heatmap(df, out_dir), ratings(df, out_dir),
        maturity_trend(df, out_dir), runtime(df, out_dir), tv_seasons(df, out_dir), people(df, out_dir),
        catalog_age(df, out_dir), missing(df, out_dir), description_terms(df, out_dir), topics(df, topic_df, out_dir),
    ]
    return paths
