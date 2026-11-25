"""Entrena y evalúa el modelo de satisfacción; guarda models/*.joblib y metrics.json."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from queveo.data import load_catalog, load_ratings  # noqa: E402
from queveo.model import save, train_and_evaluate  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    res = train_and_evaluate(load_ratings(), load_catalog())
    save(res)
    table = pd.DataFrame(res.metrics["modelos"]).T.round(4)
    print(table.to_string())
    print("\nVariables más importantes (permutación, ΔRMSE):")
    print(res.importances.head(12).round(4).to_string())
    print(f"\nTiempo: {time.time() - t0:.1f}s")

    # Figuras para la memoria / presentación
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from queveo import config as C

    img = C.ROOT / "docs" / "img"
    img.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#ADB5BD"] * (len(table) - 1) + ["#4C6EF5"]
    table["RMSE"].plot.barh(ax=axs[0], color=colors)
    axs[0].set_title("RMSE en test (menor es mejor)")
    axs[0].set_xlim(0.8, table["RMSE"].max() * 1.02)
    table["precision@5"].plot.barh(ax=axs[1], color=colors)
    axs[1].set_title("Precisión@5 (mayor es mejor)")
    axs[1].set_yticklabels([])
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(img / "modelo_comparativa.png", dpi=110)

    fig, ax = plt.subplots(figsize=(7, 4))
    res.importances.head(10).iloc[::-1].plot.barh(ax=ax, color="#12B886")
    ax.set_title("Importancia por permutación (ΔRMSE)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(img / "modelo_importancias.png", dpi=110)
