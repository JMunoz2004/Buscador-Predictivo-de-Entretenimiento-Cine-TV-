import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from queveo import config as C  # noqa: E402
from queveo.data import assign_platforms, load_catalog, load_ratings, parse_names  # noqa: E402
from queveo.features import FEATURES, pair_features, user_profiles  # noqa: E402
from queveo.model import load_metrics, load_model, split_by_user  # noqa: E402
from queveo.recommender import Query, Recommender  # noqa: E402


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module")
def rec(catalog):
    return Recommender(catalog, load_model(), load_metrics())


# ------------------------------------------------------------------ datos
def test_parse_names():
    assert parse_names('[{"id": 1, "name": "Action"}]') == ["Action"]
    assert parse_names(np.nan) == []
    assert parse_names("no es json") == []


def test_platforms_deterministic():
    a = assign_platforms("Toy Story", ["Pixar Animation Studios"])
    assert "Disney+" in a
    assert a == assign_platforms("Toy Story", ["Pixar Animation Studios"])


def test_catalog_quality(catalog):
    assert len(catalog) > 3000
    assert catalog["tmdb_id"].is_unique
    assert catalog["runtime"].gt(0).all()
    assert catalog[[f"g_{g}" for g in C.GENRES]].sum(axis=1).gt(0).all()


# ------------------------------------------------------------------ modelo
def test_split_has_no_overlap():
    r = load_ratings()
    tr, te = split_by_user(r)
    assert len(set(tr.index) & set(te.index)) == 0
    assert abs(len(te) / len(r) - 0.2) < 0.02


def test_features_shape(catalog):
    r = load_ratings().head(2000)
    X = pair_features(r, user_profiles(r, catalog), catalog)
    assert list(X[FEATURES].columns) == FEATURES
    assert not X[FEATURES].isna().any().any()


def test_hybrid_beats_baselines():
    m = load_metrics()["modelos"]
    hybrid = m["Híbrido (GBM perfil + contenido)"]["RMSE"]
    assert all(hybrid < v["RMSE"] for k, v in m.items() if not k.startswith("Híbrido"))


# ------------------------------------------------------------------ recomendador
def test_respects_constraints(rec):
    q = Query(minutes=95, mood="😂 Quiero reírme", era="Años 2000", platforms=["Netflix"])
    out = rec.recommend(q)
    assert len(out) > 0
    assert out["runtime"].le(100).all()
    assert out["year"].between(2000, 2009).all()
    assert out["platforms"].apply(lambda p: "Netflix" in p).all()
    assert out["match"].between(0, 100).all()
    assert out["match"].is_monotonic_decreasing


def test_mood_changes_ranking(rec):
    fun = rec.recommend(Query(mood="😂 Quiero reírme", top_n=10))
    scary = rec.recommend(Query(mood="👻 Quiero pasar miedo", top_n=10))
    assert fun["genres"].apply(lambda g: "Comedy" in g).mean() >= 0.7
    assert scary["genres"].apply(lambda g: "Horror" in g or "Thriller" in g).mean() >= 0.7


def test_liked_movies_excluded(rec, catalog):
    inception = int(catalog.loc[catalog.title == "Inception", "tmdb_id"].iloc[0])
    out = rec.recommend(Query(liked_ids=[inception]))
    assert inception not in set(out["tmdb_id"])


def test_impossible_query_returns_empty(rec):
    out = rec.recommend(Query(minutes=10))
    assert isinstance(out, pd.DataFrame) and out.empty
