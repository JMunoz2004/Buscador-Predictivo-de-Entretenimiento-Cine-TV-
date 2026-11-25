"""Descarga los datos originales y genera data/processed/*.parquet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from queveo.data import run_pipeline  # noqa: E402

if __name__ == "__main__":
    catalog, ratings = run_pipeline()
    print(f"Catálogo: {len(catalog)} películas · Valoraciones: {len(ratings)} "
          f"de {ratings.user_id.nunique()} usuarios")
    print(f"Duraciones imputadas: {int(catalog.runtime_imputed.sum())}")
    print(f"Sin plataforma asignada: {(catalog.platforms.str.len() == 0).sum()}")
    print("Películas por plataforma:")
    print(catalog.explode("platforms")["platforms"].value_counts().to_string())
