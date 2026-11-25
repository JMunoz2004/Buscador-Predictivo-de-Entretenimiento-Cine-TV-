"""Cliente opcional de la API de TMDb (pósters y disponibilidad real en España).

Solo se usa si existe la variable TMDB_API_KEY (o el secreto de Streamlit del mismo nombre).
Sin clave, la app funciona igual con pósters generados y disponibilidad aproximada.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

IMG_BASE = "https://image.tmdb.org/t/p/w342"
PROVIDER_MAP = {"Netflix": "Netflix", "Max": "HBO Max", "HBO Max": "HBO Max",
                "Disney Plus": "Disney+", "Amazon Prime Video": "Prime Video"}


def _key() -> str | None:
    return os.environ.get("TMDB_API_KEY") or None


def _get(path: str, **params) -> dict | None:
    key = _key()
    if not key:
        return None
    headers = {"accept": "application/json"}
    if key.startswith("ey"):  # token de lectura v4
        headers["Authorization"] = f"Bearer {key}"
    else:
        params["api_key"] = key
    url = f"https://api.themoviedb.org/3{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=6) as r:
            return json.loads(r.read())
    except Exception:  # red caída, clave inválida, etc. -> se degrada sin romper la app
        return None


def movie_details(tmdb_id: int) -> dict:
    """Devuelve {'poster': url|None, 'title_es': str|None, 'providers_es': [..]|None}."""
    out = {"poster": None, "title_es": None, "providers_es": None}
    d = _get(f"/movie/{tmdb_id}", language="es-ES", append_to_response="watch/providers")
    if not d:
        return out
    if d.get("poster_path"):
        out["poster"] = IMG_BASE + d["poster_path"]
    out["title_es"] = d.get("title")
    es = (d.get("watch/providers") or {}).get("results", {}).get("ES", {})
    names = [p.get("provider_name", "") for p in es.get("flatrate", [])]
    out["providers_es"] = sorted({PROVIDER_MAP[n] for n in names if n in PROVIDER_MAP})
    return out


def enabled() -> bool:
    return _key() is not None
