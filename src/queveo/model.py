"""Modelo de predicción de satisfacción (nota esperada 0,5–5).

Esquema de validación sin fuga de información:
  1. Por cada usuario se reserva un 20 % de sus valoraciones como *test*.
  2. En *train* los perfiles de usuario se calculan *out-of-fold* (5 pliegues): la fila que se
     predice nunca participa en el perfil que la describe.
  3. En *test* el perfil se calcula con todo el train del usuario.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config as C
from .features import FEATURES, MOVIE_NUM, GENRE_COLS, pair_features, user_profiles


def split_by_user(ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = C.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    u = rng.random(len(ratings))
    rank = pd.Series(u, index=ratings.index).groupby(ratings["user_id"]).rank(pct=True)
    test_mask = rank > (1 - test_frac)
    return ratings[~test_mask].copy(), ratings[test_mask].copy()


def oof_features(train: pd.DataFrame, catalog: pd.DataFrame, k: int = 5, seed: int = C.RANDOM_STATE):
    folds = np.random.default_rng(seed).integers(0, k, len(train))
    parts = []
    for f in range(k):
        prof = user_profiles(train[folds != f], catalog)
        part = pair_features(train[folds == f], prof, catalog)
        part["rating"] = train.loc[folds == f, "rating"].to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=C.RANDOM_STATE)


def _ranking_quality(test: pd.DataFrame, pred: np.ndarray) -> dict:
    """Correlación de Spearman media por usuario y precisión@5 (nota real >= 4)."""
    t = test.assign(pred=pred)
    rhos, precs = [], []
    for _, g in t.groupby("user_id"):
        if len(g) < 5 or g["rating"].nunique() < 2:
            continue
        if g["pred"].nunique() > 1:  # un predictor constante no ordena
            rhos.append(spearmanr(g["rating"], g["pred"]).statistic)
        # desempate aleatorio para que un predictor constante equivalga a azar
        order = g.assign(r=np.random.default_rng(0).random(len(g))).sort_values(["pred", "r"], ascending=False)
        precs.append((order.head(5)["rating"] >= 4).mean())
    return {"spearman_por_usuario": float(np.mean(rhos)) if rhos else None,
            "precision@5": float(np.mean(precs))}


@dataclass
class TrainResult:
    model: HistGradientBoostingRegressor
    metrics: dict
    importances: pd.Series


def train_and_evaluate(ratings: pd.DataFrame, catalog: pd.DataFrame) -> TrainResult:
    train, test = split_by_user(ratings)
    Xtr = oof_features(train, catalog)
    prof_full = user_profiles(train, catalog)
    Xte = pair_features(test, prof_full, catalog)
    y_te = test["rating"].to_numpy()

    model = make_model().fit(Xtr[FEATURES], Xtr["rating"])
    pred = np.clip(model.predict(Xte[FEATURES]), 0.5, 5.0)

    # --- Baselines ---------------------------------------------------------------
    gmean = train["rating"].mean()
    base_global = np.full(len(test), gmean)
    base_user = test["user_id"].map(train.groupby("user_id")["rating"].mean()).fillna(gmean).to_numpy()
    # Sesgos usuario + película (modelo clásico de "baseline predictors")
    bu = train.groupby("user_id")["rating"].mean() - gmean
    resid = train["rating"] - gmean - train["user_id"].map(bu)
    bi = resid.groupby(train["tmdb_id"]).agg(lambda s: s.sum() / (len(s) + 10))
    base_bias = (gmean + test["user_id"].map(bu).fillna(0) + test["tmdb_id"].map(bi).fillna(0)).to_numpy()
    # Solo contenido (sin perfil del usuario)
    content_cols = MOVIE_NUM + GENRE_COLS
    content = make_model().fit(Xtr[content_cols], Xtr["rating"])
    base_content = content.predict(Xte[content_cols])

    def m(p):
        return {"RMSE": float(np.sqrt(mean_squared_error(y_te, p))), "MAE": float(mean_absolute_error(y_te, p)),
                **_ranking_quality(test, p)}

    metrics = {
        "n_train": int(len(train)), "n_test": int(len(test)), "n_users": int(ratings.user_id.nunique()),
        "modelos": {
            "Media global": m(base_global),
            "Media del usuario": m(base_user),
            "Sesgos usuario + película": m(base_bias),
            "Solo contenido (GBM)": m(base_content),
            "Híbrido (GBM perfil + contenido)": m(pred),
        },
    }

    # Importancia por permutación (sobre una muestra del test)
    from sklearn.inspection import permutation_importance
    sample = Xte.sample(min(6000, len(Xte)), random_state=C.RANDOM_STATE)
    pi = permutation_importance(model, sample[FEATURES], test.loc[sample.index, "rating"],  # mismo índice
                                n_repeats=3, random_state=C.RANDOM_STATE, scoring="neg_root_mean_squared_error")
    importances = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)

    # Modelo final: reentrenado con TODAS las valoraciones (perfiles OOF)
    X_all = oof_features(ratings, catalog)
    final = make_model().fit(X_all[FEATURES], X_all["rating"])
    prof_all = user_profiles(ratings, catalog)
    metrics["importancia_top"] = {k: round(float(v), 4) for k, v in importances.head(12).items()}
    # Escalas usadas por el recomendador para construir el "usuario sintético" de la app
    metrics["afinidad_p95"] = float(np.nanpercentile(np.abs(prof_all.filter(like="aff_").to_numpy()), 95))
    metrics["user_log_n_mediana"] = float(prof_all["user_log_n"].median())
    metrics["media_global"] = float(ratings["rating"].mean())
    return TrainResult(final, metrics, importances)


def save(result: TrainResult) -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, C.MODEL_PATH, compress=3)
    C.METRICS_PATH.write_text(json.dumps(result.metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model() -> HistGradientBoostingRegressor:
    return joblib.load(C.MODEL_PATH)


def load_metrics() -> dict:
    return json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
