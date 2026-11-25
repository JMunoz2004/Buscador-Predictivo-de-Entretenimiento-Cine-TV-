"""Genera y ejecuta notebooks/01_eda.ipynb (análisis exploratorio)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# 01 · Análisis exploratorio de datos\n\n"
       "Dataset unificado **TMDb 5000** (metadatos) + **MovieLens ml-latest-small** (valoraciones). "
       "Generado con `python scripts/build_dataset.py`."),
    code("import sys; sys.path.insert(0, '../src')\n"
         "import pandas as pd, numpy as np, matplotlib.pyplot as plt\n"
         "from queveo import config as C\n"
         "from queveo.data import load_catalog, load_ratings\n"
         "plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})\n"
         "IMG = C.ROOT / 'docs' / 'img'; IMG.mkdir(parents=True, exist_ok=True)\n"
         "cat, rat = load_catalog(), load_ratings()\n"
         "print(cat.shape, rat.shape)"),
    md("## 1. Calidad del dato\nNulos tras la limpieza y valores imputados."),
    code("print('Nulos por columna (solo >0):')\n"
         "print(cat.isna().sum()[lambda s: s > 0])\n"
         "print('Duraciones imputadas por mediana de género:', int(cat.runtime_imputed.sum()))\n"
         "print('Películas con valoraciones MovieLens:', int((cat.ml_n > 0).sum()))\n"
         "cat[['year', 'runtime', 'vote_average', 'vote_count', 'popularity']].describe().round(2)"),
    md("## 2. Distribución de géneros"),
    code("g = cat.explode('genres')['genres'].map(C.GENRES_ES).value_counts()\n"
         "ax = g.sort_values().plot.barh(figsize=(7, 5), color='#4C6EF5')\n"
         "ax.set_title('Películas por género'); ax.set_xlabel('nº películas')\n"
         "plt.tight_layout(); plt.savefig(IMG / 'eda_generos.png'); plt.show()"),
    md("## 3. Duración y época\nLa duración es clave porque la app filtra por *tiempo disponible*."),
    code("fig, axs = plt.subplots(1, 2, figsize=(11, 4))\n"
         "cat.runtime.clip(60, 200).plot.hist(bins=35, ax=axs[0], color='#12B886')\n"
         "axs[0].set_title('Duración (min)'); axs[0].axvline(cat.runtime.median(), ls='--', c='k')\n"
         "cat.year.plot.hist(bins=40, ax=axs[1], color='#FAB005'); axs[1].set_title('Año de estreno')\n"
         "plt.tight_layout(); plt.savefig(IMG / 'eda_duracion_anio.png'); plt.show()\n"
         "print('% películas <= 90 min:', round((cat.runtime <= 90).mean() * 100, 1))"),
    md("## 4. Valoraciones de usuarios (MovieLens)"),
    code("fig, axs = plt.subplots(1, 2, figsize=(11, 4))\n"
         "rat.rating.value_counts().sort_index().plot.bar(ax=axs[0], color='#7950F2')\n"
         "axs[0].set_title('Distribución de notas (0,5–5)')\n"
         "rat.groupby('user_id').size().clip(upper=600).plot.hist(bins=40, ax=axs[1], color='#E64980')\n"
         "axs[1].set_title('Valoraciones por usuario'); plt.tight_layout()\n"
         "plt.savefig(IMG / 'eda_ratings.png'); plt.show()\n"
         "dens = len(rat) / (rat.user_id.nunique() * rat.tmdb_id.nunique())\n"
         "print(f'Media global: {rat.rating.mean():.2f} · Densidad de la matriz: {dens:.2%}')"),
    md("## 5. ¿Qué explica la satisfacción?\nRelación entre la nota de MovieLens y variables de contenido."),
    code("m = cat[cat.ml_n >= 10]\n"
         "cols = ['ml_mean', 'vote_average', 'log_votes', 'log_popularity', 'runtime', 'year', 'log_budget']\n"
         "corr = m[cols].corr()['ml_mean'].drop('ml_mean').sort_values()\n"
         "ax = corr.plot.barh(color=['#FA5252' if v < 0 else '#40C057' for v in corr], figsize=(7, 3.5))\n"
         "ax.set_title('Correlación con la nota media de usuarios'); plt.tight_layout()\n"
         "plt.savefig(IMG / 'eda_correlaciones.png'); plt.show(); corr.round(3)"),
    code("gm = (rat.merge(cat[['tmdb_id', 'genres']], on='tmdb_id').explode('genres')\n"
         "        .groupby('genres').rating.agg(['mean', 'count']).sort_values('mean'))\n"
         "gm.index = gm.index.map(C.GENRES_ES); gm.round(2)"),
    md("## 6. Disponibilidad por plataforma (aproximada)"),
    code("p = cat.explode('platforms')['platforms'].value_counts()\n"
         "ax = p.plot.bar(color=['#E50914', '#5822B4', '#113CCF', '#00A8E1'], figsize=(6, 3.5), rot=0)\n"
         "ax.set_title('Títulos por plataforma'); plt.tight_layout()\n"
         "plt.savefig(IMG / 'eda_plataformas.png'); plt.show()\n"
         "print('Sin plataforma:', (cat.platforms.str.len() == 0).sum())"),
    md("## Conclusiones\n"
       "* La `vote_average` de TMDb y el número de votos son los mejores predictores *a priori* de la "
       "satisfacción (r≈0,83 para la nota TMDb); la duración y la antigüedad se asocian positivamente "
       "(sesgo de supervivencia de los clásicos) y el presupuesto, negativamente (blockbusters).\n"
       "* La matriz usuario-película es muy dispersa (~3 %), por lo que un filtrado colaborativo puro no "
       "basta: se justifica un **modelo híbrido** que combine perfil del usuario y contenido.\n"
       "* Las preferencias por género difieren entre usuarios, lo que permite aprender la interacción "
       "*afinidad del usuario × géneros de la película*, que es lo que la app simula con el estado de ánimo."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
path = ROOT / "notebooks" / "01_eda.ipynb"
NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(path.parent)}}).execute()
nbf.write(nb, path)
print("OK ->", path)
