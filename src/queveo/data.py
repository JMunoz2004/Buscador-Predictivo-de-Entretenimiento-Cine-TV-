"""Pipeline de datos: descarga, limpieza, imputación, unión TMDb + MovieLens.

Uso:  python -m scripts.build_dataset
"""
from __future__ import annotations

import hashlib
import json
import urllib.request

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Descarga
# --------------------------------------------------------------------------- #
def download_raw(force: bool = False) -> None:
    """Descarga los CSV originales a data/raw/ si no existen."""
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, url in C.RAW_SOURCES.items():
        dest = C.DATA_RAW / name
        if dest.exists() and not force:
            continue
        print(f"Descargando {name} ...")
        urllib.request.urlretrieve(url, dest)


# --------------------------------------------------------------------------- #
# Utilidades de parseo
# --------------------------------------------------------------------------- #
def parse_names(raw: str | float, key: str = "name") -> list[str]:
    """Convierte las columnas JSON de TMDb ('[{"id":..,"name":..}]') en listas."""
    if not isinstance(raw, str) or not raw:
        return []
    try:
        return [d[key] for d in json.loads(raw) if key in d]
    except (json.JSONDecodeError, TypeError):
        return []


def _director(crew_raw: str) -> str:
    try:
        crew = json.loads(crew_raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    for person in crew:
        if person.get("job") == "Director":
            return person.get("name", "")
    return ""


def _stable_unit(text: str) -> float:
    """Número pseudoaleatorio en [0, 1) estable a partir de un texto (reproducible)."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


# --------------------------------------------------------------------------- #
# Disponibilidad en plataformas
# --------------------------------------------------------------------------- #
# TMDb 5000 no incluye disponibilidad en streaming. Se aproxima a partir de la
# productora (catálogos propios de cada grupo) y, para el resto, mediante una
# asignación determinista que imita las licencias rotativas. Es una
# aproximación documentada; la fase siguiente la sustituye por la API
# "watch/providers" de TMDb (JustWatch).
STUDIO_PLATFORM = {
    "Disney+": ["Walt Disney", "Pixar", "Marvel Studios", "Lucasfilm", "Twentieth Century Fox",
                "Fox Searchlight", "Touchstone", "Hollywood Pictures", "Blue Sky", "Fox 2000"],
    "HBO Max": ["Warner Bros", "New Line", "HBO", "DC Comics", "DC Entertainment",
                "Castle Rock", "Turner"],
    "Prime Video": ["Metro-Goldwyn-Mayer", "United Artists", "Amazon", "Orion", "Lionsgate",
                    "Lions Gate"],
    "Netflix": ["Netflix"],
}


def assign_platforms(title: str, companies: list[str]) -> list[str]:
    platforms: set[str] = set()
    joined = " | ".join(companies)
    for platform, studios in STUDIO_PLATFORM.items():
        if any(s.lower() in joined.lower() for s in studios):
            platforms.add(platform)
    # Licencias de terceros: cada plataforma "licencia" la película con cierta probabilidad
    probs = {"Netflix": 0.30, "Prime Video": 0.30, "HBO Max": 0.12, "Disney+": 0.04}
    for platform, p in probs.items():
        if _stable_unit(f"{title}::{platform}") < p:
            platforms.add(platform)
    return sorted(platforms)


# --------------------------------------------------------------------------- #
# Catálogo
# --------------------------------------------------------------------------- #
def build_catalog(min_votes: int = 20) -> pd.DataFrame:
    movies = pd.read_csv(C.DATA_RAW / "tmdb_5000_movies.csv")
    credits = pd.read_csv(C.DATA_RAW / "tmdb_5000_credits.csv")

    df = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id", how="left")

    # 1. Filtrado básico
    df = df[df["status"] == "Released"].copy()
    df = df.dropna(subset=["release_date"])
    df = df[df["vote_count"] >= min_votes]

    # 2. Parseo de columnas JSON
    df["genres"] = df["genres"].apply(parse_names)
    df["keywords"] = df["keywords"].apply(parse_names)
    df["companies"] = df["production_companies"].apply(parse_names)
    df["cast"] = df["cast"].apply(lambda s: parse_names(s)[:4])
    df["director"] = df["crew"].apply(_director)

    df["genres"] = df["genres"].apply(lambda g: [x for x in g if x in C.GENRES_ES])
    df = df[df["genres"].str.len() > 0]

    # 3. Fechas
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year.astype(int)

    # 4. Imputación de duración: 0 o nulo -> mediana del género principal
    df["runtime"] = df["runtime"].replace(0, np.nan)
    df["main_genre"] = df["genres"].str[0]
    df["runtime_imputed"] = df["runtime"].isna()
    df["runtime"] = df["runtime"].fillna(df.groupby("main_genre")["runtime"].transform("median"))
    df["runtime"] = df["runtime"].fillna(df["runtime"].median()).round().astype(int)

    # 5. Texto
    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["soup"] = (
        df["overview"] + " " + df["tagline"] + " "
        + df["keywords"].apply(" ".join) + " "
        + df["genres"].apply(lambda g: " ".join(g * 2))
    ).str.lower()

    # 6. Variables numéricas derivadas
    df["log_votes"] = np.log1p(df["vote_count"])
    df["log_popularity"] = np.log1p(df["popularity"])
    df["log_budget"] = np.log1p(df["budget"].where(df["budget"] > 1000, 0))
    df["is_english"] = (df["original_language"] == "en").astype(int)
    for g in C.GENRES:
        df[f"g_{g}"] = df["genres"].apply(lambda gs, g=g: int(g in gs))

    # 7. Plataformas
    df["platforms"] = df.apply(lambda r: assign_platforms(r["title"], r["companies"]), axis=1)

    keep = ["id", "title", "original_title", "year", "runtime", "runtime_imputed", "genres", "main_genre",
            "keywords", "director", "cast", "companies", "overview", "tagline", "soup", "vote_average",
            "vote_count", "popularity", "log_votes", "log_popularity", "log_budget", "is_english",
            "original_language", "platforms"] + [f"g_{g}" for g in C.GENRES]
    return df[keep].rename(columns={"id": "tmdb_id"}).reset_index(drop=True)


def build_ratings(catalog: pd.DataFrame) -> pd.DataFrame:
    """Valoraciones de MovieLens asociadas a películas del catálogo (vía links.csv)."""
    ratings = pd.read_csv(C.DATA_RAW / "ratings.csv")
    links = pd.read_csv(C.DATA_RAW / "links.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    r = ratings.merge(links[["movieId", "tmdbId"]], on="movieId").rename(
        columns={"userId": "user_id", "tmdbId": "tmdb_id"})
    r = r[r["tmdb_id"].isin(catalog["tmdb_id"])]
    return r[["user_id", "tmdb_id", "rating", "timestamp"]].reset_index(drop=True)


def add_rating_stats(catalog: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    stats = ratings.groupby("tmdb_id")["rating"].agg(ml_n="count", ml_mean="mean").reset_index()
    out = catalog.merge(stats, on="tmdb_id", how="left")
    out["ml_n"] = out["ml_n"].fillna(0).astype(int)
    return out


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    download_raw()
    catalog = build_catalog()
    ratings = build_ratings(catalog)
    catalog = add_rating_stats(catalog, ratings)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(C.CATALOG_PATH, index=False)
    ratings.to_parquet(C.RATINGS_PATH, index=False)
    return catalog, ratings


def load_catalog() -> pd.DataFrame:
    return pd.read_parquet(C.CATALOG_PATH)


def load_ratings() -> pd.DataFrame:
    return pd.read_parquet(C.RATINGS_PATH)
