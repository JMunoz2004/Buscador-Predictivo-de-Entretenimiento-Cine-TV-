"""Motor de recomendación híbrido: contexto del usuario -> ranking con *match score*.

match = 0.45·satisfacción predicha (GBM)  +  0.25·encaje con el estado de ánimo
      + 0.20·similitud de contenido (TF-IDF) +  0.10·aprovechamiento del tiempo
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import config as C
from .features import AFF_COLS, FEATURES, GENRE_COLS, SHRINK, add_interactions


@dataclass
class Query:
    minutes: int | None = 120
    mood: str = "🎲 Sorpréndeme"
    era: str = "Me da igual"
    platforms: list[str] = field(default_factory=list)  # vacío = todas
    liked_ids: list[int] = field(default_factory=list)
    min_rating: float = 0.0  # nota TMDb mínima
    top_n: int = 10
    seed: int = 0  # semilla para el modo "Sorpréndeme"


class Recommender:
    def __init__(self, catalog: pd.DataFrame, model, metrics: dict):
        self.cat = catalog.reset_index(drop=True)
        self.model = model
        self.metrics = metrics
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                                     max_features=30000, sublinear_tf=True)
        self.M = self.tfidf.fit_transform(self.cat["soup"])
        self.idx = pd.Series(self.cat.index, index=self.cat["tmdb_id"])
        self.G = self.cat[GENRE_COLS].to_numpy(dtype=float)

    # ------------------------------------------------------------------ perfil
    def synthetic_profile(self, q: Query) -> dict:
        """Usuario sintético equivalente a los perfiles con los que se entrenó el modelo."""
        gmean = self.metrics["media_global"]
        scale = self.metrics["afinidad_p95"]
        mood_w = C.MOODS.get(q.mood, {}).get("genres", {})
        aff = np.array([mood_w.get(g, 0.0) * scale for g in C.GENRES])

        liked_rows = [self.idx[i] for i in q.liked_ids if i in self.idx]
        if liked_rows:  # películas favoritas = valoraciones de 5 estrellas
            Gl = self.G[liked_rows]
            dev = 5.0 - gmean
            aff = aff + (Gl * dev).sum(axis=0) / (Gl.sum(axis=0) + SHRINK)
        # mantener el perfil dentro del rango observado en usuarios reales (evita extrapolar)
        aff = np.clip(aff, -scale, scale)

        lo, hi = C.ERAS.get(q.era, (1900, 2100))
        if q.era != "Me da igual":
            pref_year = (lo + min(hi, int(self.cat["year"].max()))) / 2
        elif liked_rows:
            pref_year = float(self.cat.loc[liked_rows, "year"].mean())
        else:
            pref_year = float(self.cat["year"].median())

        prof = dict(zip(AFF_COLS, aff))
        prof.update(user_mean=gmean, user_log_n=self.metrics["user_log_n_mediana"], user_pref_year=pref_year)
        return prof

    # -------------------------------------------------------------- candidatos
    def candidates(self, q: Query) -> pd.Series:
        c = self.cat
        mask = pd.Series(True, index=c.index)
        if q.minutes:
            mask &= c["runtime"] <= q.minutes + 5
        lo, hi = C.ERAS.get(q.era, (1900, 2100))
        mask &= c["year"].between(lo, hi)
        if q.platforms:
            sel = set(q.platforms)
            mask &= c["platforms"].apply(lambda p: bool(sel.intersection(p)))
        if q.min_rating:
            mask &= c["vote_average"] >= q.min_rating
        if q.liked_ids:
            mask &= ~c["tmdb_id"].isin(q.liked_ids)
        return mask

    # -------------------------------------------------------------- puntuación
    def predict_satisfaction(self, rows: pd.DataFrame, profile: dict) -> np.ndarray:
        X = rows.assign(**profile)
        X = add_interactions(X)
        return np.clip(self.model.predict(X[FEATURES]), 0.5, 5.0)

    def content_scores(self, q: Query, rows_idx: np.ndarray) -> np.ndarray | None:
        text = C.MOODS.get(q.mood, {}).get("keywords", "")
        liked_rows = [self.idx[i] for i in q.liked_ids if i in self.idx]
        if not text and not liked_rows:
            return None
        parts = []
        if text:
            parts.append(self.tfidf.transform([text]))
        if liked_rows:
            parts.append(self.M[liked_rows])
        sims = np.vstack([linear_kernel(p, self.M[rows_idx]) for p in parts]).mean(axis=0)
        top = np.percentile(sims, 99) if sims.max() > 0 else 1.0
        return np.clip(sims / max(top, 1e-9), 0, 1)

    def recommend(self, q: Query) -> pd.DataFrame:
        mask = self.candidates(q)
        rows = self.cat[mask]
        if rows.empty:
            return rows.assign(match=[])
        idx = rows.index.to_numpy()
        prof = self.synthetic_profile(q)

        pred = self.predict_satisfaction(rows, prof)
        s_pred = np.clip((pred - 2.5) / 2.0, 0, 1)

        mood_w = C.MOODS.get(q.mood, {}).get("genres", {})
        if mood_w:
            w = np.array([mood_w.get(g, 0.0) for g in C.GENRES])
            G = self.G[idx]
            # media de los 2 géneros más afines (evita penalizar películas con muchos géneros)
            vals = np.where(G > 0, w, -np.inf)
            top2 = np.sort(vals, axis=1)[:, -2:]
            top2 = np.where(np.isinf(top2), np.nan, top2)
            s_mood = (np.nan_to_num(np.nanmean(top2, axis=1)) + 1) / 2
        else:
            s_mood = np.full(len(idx), np.nan)

        s_content = self.content_scores(q, idx)
        if s_content is None:
            s_content = np.full(len(idx), np.nan)

        if q.minutes:
            ratio = rows["runtime"].to_numpy() / q.minutes
            s_time = np.where(ratio > 1, 0.85, np.where(ratio >= 0.6, 1.0, ratio / 0.6))
        else:
            s_time = np.full(len(idx), np.nan)

        comps = {"satisfaccion": s_pred, "animo": s_mood, "contenido": s_content, "tiempo": s_time}
        W = C.MATCH_WEIGHTS
        active = {k: v for k, v in comps.items() if not np.isnan(v).all()}
        wsum = sum(W[k] for k in active)
        match = sum(W[k] * v for k, v in active.items()) / wsum

        if q.mood == "🎲 Sorpréndeme":  # exploración: ruido controlado entre las buenas opciones
            noise = np.random.default_rng(q.seed).random(len(idx))
            match = 0.8 * match + 0.2 * noise * (s_pred > 0.6)

        out = rows.copy()
        out["pred_rating"] = pred
        for k, v in comps.items():
            out[f"s_{k}"] = v
        out["match"] = np.round(match * 100, 1)
        out = out.sort_values(["match", "vote_count"], ascending=False).head(q.top_n)
        out["reasons"] = [self.explain(r, q) for _, r in out.iterrows()]
        return out

    # ------------------------------------------------------------ explicación
    def explain(self, r: pd.Series, q: Query) -> list[str]:
        reasons = []
        mood_w = C.MOODS.get(q.mood, {}).get("genres", {})
        good = [C.GENRES_ES[g] for g in r["genres"] if mood_w.get(g, 0) > 0.3]
        if good:
            reasons.append(f"Encaja con tu ánimo: {', '.join(good[:2]).lower()}")
        if r["pred_rating"] >= 3.8:
            reasons.append(f"Predicción de satisfacción alta ({r['pred_rating']:.1f}/5)")
        if r["vote_average"] >= 7.5:
            reasons.append(f"Muy bien valorada ({r['vote_average']:.1f}/10 en TMDb)")
        if q.minutes and r["runtime"] <= q.minutes:
            reasons.append(f"Dura {r['runtime']} min: te sobran {q.minutes - r['runtime']}")
        elif q.minutes:
            reasons.append(f"Dura {r['runtime']} min, apenas pasa de tu tiempo")
        if not np.isnan(r["s_contenido"]) and r["s_contenido"] >= 0.6 and q.liked_ids:
            reasons.append("Se parece a las películas que te gustan")
        return reasons[:3]

    def similar(self, tmdb_id: int, k: int = 6, exclude: list[int] | None = None) -> pd.DataFrame:
        i = self.idx[tmdb_id]
        sims = linear_kernel(self.M[i], self.M).ravel()
        g = self.G[i]
        jacc = (self.G @ g) / np.maximum((self.G + g).clip(max=1).sum(axis=1), 1)
        quality = ((self.cat["vote_average"] - 5) / 3).clip(0, 1).to_numpy()
        score = 0.6 * sims + 0.25 * jacc + 0.15 * quality
        score[i] = -1
        for j in exclude or []:
            if j in self.idx:
                score[self.idx[j]] = -1
        top = np.argsort(-score)[:k]
        return self.cat.iloc[top].assign(similarity=score[top])
