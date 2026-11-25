"""Ingeniería de variables: perfiles de usuario y matriz (usuario, película) -> features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

GENRE_COLS = [f"g_{g}" for g in C.GENRES]
AFF_COLS = [f"aff_{g}" for g in C.GENRES]
MOVIE_NUM = ["year", "runtime", "vote_average", "log_votes", "log_popularity", "log_budget", "is_english"]
USER_NUM = ["user_mean", "user_log_n", "user_pref_year"]
INTERACTION = ["aff_dot", "aff_max", "aff_min", "year_gap"]
FEATURES = MOVIE_NUM + GENRE_COLS + USER_NUM + AFF_COLS + INTERACTION

SHRINK = 5.0  # regularización bayesiana de la afinidad por género


def user_profiles(ratings: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Perfil por usuario a partir de sus valoraciones.

    * user_mean: nota media del usuario
    * aff_<género>: desviación media (regularizada) de sus notas en ese género
    * user_pref_year: año medio de las películas que puntúa por encima de su media
    """
    movie = catalog.set_index("tmdb_id")[GENRE_COLS + ["year"]]
    df = ratings[["user_id", "tmdb_id", "rating"]].join(movie, on="tmdb_id")
    mean = df.groupby("user_id")["rating"].mean()
    dev = df["rating"] - df["user_id"].map(mean)

    G = df[GENRE_COLS].to_numpy(dtype=float)
    num = pd.DataFrame(G * dev.to_numpy()[:, None], columns=AFF_COLS).groupby(df["user_id"].to_numpy()).sum()
    cnt = pd.DataFrame(G, columns=AFF_COLS).groupby(df["user_id"].to_numpy()).sum()
    aff = num / (cnt + SHRINK)

    liked = df[dev > 0]
    pref_year = liked.groupby("user_id")["year"].mean().reindex(mean.index).fillna(df["year"].mean())

    prof = aff.copy()
    prof.index.name = "user_id"
    prof["user_mean"] = mean
    prof["user_log_n"] = np.log1p(df.groupby("user_id").size())
    prof["user_pref_year"] = pref_year
    return prof


def pair_features(pairs: pd.DataFrame, profiles: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de features para pares (user_id, tmdb_id)."""
    movie = catalog.set_index("tmdb_id")[MOVIE_NUM + GENRE_COLS]
    X = pairs[["user_id", "tmdb_id"]].join(movie, on="tmdb_id").join(profiles, on="user_id")
    return add_interactions(X)


def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    G = X[GENRE_COLS].to_numpy(dtype=float)
    A = X[AFF_COLS].to_numpy(dtype=float)
    prod = G * A
    n = G.sum(axis=1).clip(min=1)
    X = X.copy()
    X["aff_dot"] = prod.sum(axis=1) / n
    masked = np.where(G > 0, A, np.nan)
    X["aff_max"] = np.nan_to_num(np.nanmax(np.where(G > 0, masked, -np.inf), axis=1), neginf=0.0)
    X["aff_min"] = np.nan_to_num(np.nanmin(np.where(G > 0, masked, np.inf), axis=1), posinf=0.0)
    X["year_gap"] = (X["year"] - X["user_pref_year"]).abs()
    return X
