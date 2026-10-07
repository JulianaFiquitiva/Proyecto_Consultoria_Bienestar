"""
Visualization - Generacion de graficos estadisticos.

Genera:
- Histogramas de T-scores
- Graficos de barras por grupo
- Perfil de dimensiones
- Correlaciones
- Boxplots
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class Visualizer:
    """
    Genera visualizaciones del analisis de bienestar.
    """

    def __init__(self):
        self._figures: List[Any] = []

    def histogram_t_scores(
        self,
        df: pd.DataFrame,
        t_col: str = "T_Global",
        title: str = "Distribucion de T-scores de Bienestar",
    ) -> Dict[str, Any]:
        """
        Genera histograma de T-scores.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        t = df[t_col].dropna()
        
        ax.hist(t, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
        ax.axvline(t.mean(), color="red", linestyle="--", label=f"Media: {t.mean():.1f}")
        ax.axvline(40, color="orange", linestyle=":", label="Riesgo (T<40)")
        ax.axvline(60, color="green", linestyle=":", label="Alto (T>=60)")
        ax.set_xlabel("T-score")
        ax.set_ylabel("Frecuencia")
        ax.set_title(title)
        ax.legend()

        return {"figure": fig, "path": None}

    def bar_comparison(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        title: str = "Comparacion por Grupo",
    ) -> Dict[str, Any]:
        """
        Genera grafico de barras comparativo.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        group_stats = df.groupby(group_col)[value_col].agg(["mean", "std"]).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            group_stats[group_col].astype(str),
            group_stats["mean"],
            yerr=group_stats["std"],
            capsize=5,
            color="steelblue",
            edgecolor="white",
        )
        ax.set_ylabel(value_col)
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")

        return {"figure": fig, "path": None}

    def dimension_profile(
        self,
        df: pd.DataFrame,
        t_cols: List[str],
        title: str = "Perfil de Dimensiones de Bienestar",
    ) -> Dict[str, Any]:
        """
        Genera grafico de perfil por dimensiones.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dims = []
        means = []
        stds = []
        for col in t_cols:
            if col in df.columns:
                dims.append(col.replace("T_", ""))
                means.append(df[col].mean())
                stds.append(df[col].std())

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(dims))
        ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", edgecolor="white")
        ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="Promedio teorico (50)")
        ax.set_xticks(x)
        ax.set_xticklabels(dims, rotation=45, ha="right")
        ax.set_ylabel("T-score")
        ax.set_title(title)
        ax.legend()

        return {"figure": fig, "path": None}

    def correlation_heatmap(
        self,
        df: pd.DataFrame,
        cols: List[str],
        title: str = "Matriz de Correlaciones",
    ) -> Dict[str, Any]:
        """
        Genera mapa de calor de correlaciones.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        corr = df[cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels([c.replace("T_", "") for c in cols], rotation=45, ha="right")
        ax.set_yticklabels([c.replace("T_", "") for c in cols])
        plt.colorbar(im)
        ax.set_title(title)

        return {"figure": fig, "path": None}

    def boxplot_by_group(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        title: str = "Distribucion por Grupo",
    ) -> Dict[str, Any]:
        """
        Genera boxplot por grupo.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = df[group_col].unique()
        data_groups = [df[df[group_col] == g][value_col].dropna().values for g in groups]
        
        ax.boxplot(data_groups, labels=[str(g) for g in groups])
        ax.set_ylabel(value_col)
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")

        return {"figure": fig, "path": None}

    def save_figure(self, fig, path: str) -> str:
        """Guarda una figura en disco."""
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path
