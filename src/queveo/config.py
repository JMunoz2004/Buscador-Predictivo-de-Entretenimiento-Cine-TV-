"""Configuración central del proyecto: rutas, fuentes de datos y diccionarios de dominio."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

CATALOG_PATH = DATA_PROCESSED / "catalog.parquet"
RATINGS_PATH = DATA_PROCESSED / "ratings.parquet"
MODEL_PATH = MODELS_DIR / "satisfaction_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Copias públicas en GitHub de los datasets originales (TMDb 5000 de Kaggle y
# MovieLens ml-latest-small de GroupLens). Se fija el commit para reproducibilidad.
_LKPY = "https://raw.githubusercontent.com/lenskit/lkpy/223fb7e833c3fe9800497bc9610ff5628a30a118/data/ml-latest-small"
_TMDB = "https://raw.githubusercontent.com/missvicki/TMDb-movie-data/master"
RAW_SOURCES = {
    "tmdb_5000_movies.csv": f"{_TMDB}/tmdb_5000_movies.csv",
    "tmdb_5000_credits.csv": f"{_TMDB}/tmdb_5000_credits.csv",
    "ratings.csv": f"{_LKPY}/ratings.csv",
    "links.csv": f"{_LKPY}/links.csv",
}

# Géneros TMDb -> nombre en castellano (orden fijo = orden de las columnas one-hot)
GENRES_ES = {
    "Action": "Acción",
    "Adventure": "Aventura",
    "Animation": "Animación",
    "Comedy": "Comedia",
    "Crime": "Crimen",
    "Documentary": "Documental",
    "Drama": "Drama",
    "Family": "Familiar",
    "Fantasy": "Fantasía",
    "History": "Historia",
    "Horror": "Terror",
    "Music": "Música",
    "Mystery": "Misterio",
    "Romance": "Romance",
    "Science Fiction": "Ciencia ficción",
    "Thriller": "Suspense",
    "War": "Bélica",
    "Western": "Western",
}
GENRES = list(GENRES_ES)

PLATFORMS = ["Netflix", "HBO Max", "Disney+", "Prime Video"]

# Estados de ánimo: afinidad por género (-1 a 1) y palabras clave para el
# filtrado basado en contenido (TF-IDF sobre sinopsis + keywords de TMDb).
MOODS = {
    "😂 Quiero reírme": {
        "genres": {"Comedy": 1.0, "Animation": 0.4, "Family": 0.3, "Romance": 0.2,
                   "Horror": -0.8, "War": -0.6, "Drama": -0.3},
        "keywords": "comedy funny parody friendship road trip buddy satire hilarious",
    },
    "💥 Necesito adrenalina": {
        "genres": {"Action": 1.0, "Thriller": 0.7, "Adventure": 0.6, "Crime": 0.4,
                   "Science Fiction": 0.3, "Romance": -0.5, "Documentary": -0.8},
        "keywords": "chase explosion heist fight revenge mission escape assassin car race",
    },
    "😢 Me apetece emocionarme": {
        "genres": {"Drama": 1.0, "Romance": 0.6, "Music": 0.3, "History": 0.3,
                   "Action": -0.4, "Horror": -0.7},
        "keywords": "loss family love grief friendship hope cancer redemption sacrifice",
    },
    "👻 Quiero pasar miedo": {
        "genres": {"Horror": 1.0, "Thriller": 0.6, "Mystery": 0.5,
                   "Comedy": -0.5, "Family": -1.0, "Animation": -0.8, "Romance": -0.5},
        "keywords": "haunted ghost demon serial killer supernatural possession monster survival",
    },
    "🧠 Quiero pensar": {
        "genres": {"Science Fiction": 0.8, "Mystery": 0.8, "Documentary": 0.6, "Drama": 0.4,
                   "History": 0.4, "Thriller": 0.3, "Family": -0.5, "Animation": -0.4},
        "keywords": "dystopia time travel artificial intelligence conspiracy philosophy twist investigation",
    },
    "❤️ Algo romántico": {
        "genres": {"Romance": 1.0, "Comedy": 0.4, "Drama": 0.4, "Music": 0.2,
                   "Horror": -0.9, "War": -0.6, "Action": -0.3},
        "keywords": "love wedding relationship romantic comedy first love couple date",
    },
    "🪐 Evadirme a otro mundo": {
        "genres": {"Fantasy": 1.0, "Science Fiction": 0.8, "Adventure": 0.7, "Animation": 0.4,
                   "Documentary": -0.8, "Crime": -0.3},
        "keywords": "magic space alien kingdom dragon quest wizard planet another world",
    },
    "👨‍👩‍👧 Ver en familia": {
        "genres": {"Family": 1.0, "Animation": 0.9, "Adventure": 0.5, "Comedy": 0.4, "Fantasy": 0.3,
                   "Horror": -1.0, "Crime": -0.6, "War": -0.6, "Thriller": -0.5},
        "keywords": "family friendship animals kids adventure christmas talking animal",
    },
    "🎲 Sorpréndeme": {"genres": {}, "keywords": ""},
}

# Pesos del match score (suman 1)
MATCH_WEIGHTS = {"satisfaccion": 0.45, "animo": 0.25, "contenido": 0.20, "tiempo": 0.10}

# Épocas: rango de años (inclusive)
ERAS = {
    "Me da igual": (1900, 2100),
    "Clásicos (antes de 1980)": (1900, 1979),
    "Años 80 y 90": (1980, 1999),
    "Años 2000": (2000, 2009),
    "Recientes (2010+)": (2010, 2100),
}

RANDOM_STATE = 42
