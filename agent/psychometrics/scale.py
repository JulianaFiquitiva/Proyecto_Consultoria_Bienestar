"""
Análisis psicométrico de la escala de Ryff.

Valida: 29 ítems, ítems invertidos, 6 dimensiones,
alfa de Cronbach, correlaciones, distribución de respuestas,
consistencia interna.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.core.config import (
    DIMENSION_MAP,
    DIMENSION_ORDER,
    ITEM_LABELS,
    NEGATIVE_ITEMS,
    N_ITEMS,
    N_RESPONSE_CATEGORIES,
)
from agent.statistics.descriptive import DescriptiveStats


@dataclass
class PsychometricReport:
    """Reporte de análisis psicométrico."""

    n_items: int
    n_dimensions: int
    n_negative_items: int
    alpha_global: float
    alpha_dimensions: Dict[str, float]
    item_means: Dict[int, float]
    item_stds: Dict[int, float]
    item_difficulty: Dict[int, float]
    dimension_means: Dict[str, float]
    dimension_stds: Dict[str, float]
    correlation_matrix: pd.DataFrame
    response_distributions: Dict[int, Dict[int, float]]
    alpha_if_deleted: Dict[int, float]
    warnings: List[str]


class PsychometricScale:
    """
    Análisis psicométrico de la escala de bienestar de Ryff.

    Implementa análisis ponderado y no ponderado de:
    - Consistencia interna (α de Cronbach)
    - Distribución de respuestas
    - Dificultad de ítems
    - Correlaciones ítem-total
    - Análisis por dimensión
    """

    def __init__(self):
        self.desc = DescriptiveStats()

    def full_analysis(
        self,
        df: pd.DataFrame,
        item_cols: List[str],
        weights_col: Optional[str] = None,
    ) -> PsychometricReport:
        """
        Realiza el análisis psicométrico completo.

        Args:
            df: DataFrame con los datos.
            item_cols: Lista de 29 columnas de ítems.
            weights_col: Columna de pesos (opcional).

        Returns:
            PsychometricReport con todos los resultados.
        """
        weights = df[weights_col].values if weights_col else np.ones(len(df))

        # Alpha global
        alpha_global = self.desc.weighted_cronbach_alpha(
            df[item_cols], weights
        ) if weights_col else self._cronbach_alpha(df[item_cols])

        # Alpha por dimensión
        alpha_dims = {}
        for dim_name, item_numbers in DIMENSION_MAP.items():
            dim_cols = [item_cols[i - 1] for i in item_numbers]
            if weights_col:
                alpha_dims[dim_name] = self.desc.weighted_cronbach_alpha(
                    df[dim_cols], weights
                )
            else:
                alpha_dims[dim_name] = self._cronbach_alpha(df[dim_cols])

        # Medias y DEs por ítem
        item_means = {}
        item_stds = {}
        for i, col in enumerate(item_cols, 1):
            if weights_col:
                item_means[i] = self.desc.weighted_mean(df[col], weights)
                item_stds[i] = self.desc.weighted_std(df[col], weights)
            else:
                item_means[i] = df[col].mean()
                item_stds[i] = df[col].std()

        # Dificultad politómica (media / max possible)
        item_difficulty = {
            i: item_means[i] / N_RESPONSE_CATEGORIES for i in item_means
        }

        # Medias por dimensión
        dim_means = {}
        dim_stds = {}
        for dim_name, item_numbers in DIMENSION_MAP.items():
            dim_cols = [item_cols[i - 1] for i in item_numbers]
            dim_data = df[dim_cols].mean(axis=1)
            if weights_col:
                dim_means[dim_name] = self.desc.weighted_mean(dim_data, weights)
                dim_stds[dim_name] = self.desc.weighted_std(dim_data, weights)
            else:
                dim_means[dim_name] = dim_data.mean()
                dim_stds[dim_name] = dim_data.std()

        # Matriz de correlaciones
        corr_matrix = df[item_cols].corr()

        # Distribuciones de respuesta
        response_dist = {}
        for i, col in enumerate(item_cols, 1):
            vc = df[col].value_counts(normalize=True).sort_index()
            response_dist[i] = {int(k): float(v) for k, v in vc.items()}

        # Alpha if Deleted
        alpha_del = {}
        for j, col in enumerate(item_cols):
            remaining = [c for c in item_cols if c != col]
            if weights_col:
                alpha_del[j + 1] = self.desc.weighted_cronbach_alpha(
                    df[remaining], weights
                )
            else:
                alpha_del[j + 1] = self._cronbach_alpha(df[remaining])

        # Advertencias
        warnings = self._generate_warnings(
            alpha_global, alpha_dims, item_means, item_difficulty
        )

        return PsychometricReport(
            n_items=len(item_cols),
            n_dimensions=len(DIMENSION_MAP),
            n_negative_items=len(NEGATIVE_ITEMS),
            alpha_global=alpha_global,
            alpha_dimensions=alpha_dims,
            item_means=item_means,
            item_stds=item_stds,
            item_difficulty=item_difficulty,
            dimension_means=dim_means,
            dimension_stds=dim_stds,
            correlation_matrix=corr_matrix,
            response_distributions=response_dist,
            alpha_if_deleted=alpha_del,
            warnings=warnings,
        )

    def validate_item_inversion(
        self,
        df_original: pd.DataFrame,
        df_inverted: pd.DataFrame,
        item_cols: List[str],
    ) -> Dict[int, Dict[str, float]]:
        """
        Valida que la inversión de ítems negativos se realizó correctamente.

        Fórmula: valor_inv = 7 - valor_original

        Returns:
            Dict con medias originales e invertidas por ítem.
        """
        results = {}
        for item_num in NEGATIVE_ITEMS:
            col = item_cols[item_num - 1]
            orig_mean = df_original[col].mean()
            inv_mean = df_inverted[col].mean()
            expected_inv_mean = 7 - orig_mean
            results[item_num] = {
                "original_mean": float(orig_mean),
                "inverted_mean": float(inv_mean),
                "expected_mean": float(expected_inv_mean),
                "difference": float(abs(inv_mean - expected_inv_mean)),
            }
        return results

    def check_dimension_structure(
        self,
        df: pd.DataFrame,
        item_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Verifica la estructura de dimensiones de Ryff.

        Returns:
            Dict con información de la estructura.
        """
        structure = {}
        for dim_name, item_numbers in DIMENSION_MAP.items():
            dim_cols = [item_cols[i - 1] for i in item_numbers]
            dim_data = df[dim_cols]

            # Correlaciones intra-dimensión
            intra_corr = dim_data.corr().values
            n_triu = np.sum(np.triu(np.ones_like(intra_corr, dtype=bool), k=1))
            mean_intra = np.sum(intra_corr[np.triu_indices(len(intra_corr), k=1)]) / n_triu

            structure[dim_name] = {
                "items": item_numbers,
                "n_items": len(item_numbers),
                "mean_internal_correlation": float(mean_intra),
                "item_labels": [ITEM_LABELS.get(i, f"Item {i}") for i in item_numbers],
            }
        return structure

    def _cronbach_alpha(self, data: pd.DataFrame) -> float:
        """Alpha de Cronbach sin ponderar."""
        k = data.shape[1]
        total = data.sum(axis=1)
        var_total = total.var(ddof=1)
        var_items = sum(data.iloc[:, j].var(ddof=1) for j in range(k))
        return float((k / (k - 1)) * (1 - var_items / var_total))

    def _generate_warnings(
        self,
        alpha_global: float,
        alpha_dims: Dict[str, float],
        item_means: Dict[int, float],
        item_difficulty: Dict[int, float],
    ) -> List[str]:
        """Genera advertencias psicométricas."""
        warnings = []

        # Alpha global
        if alpha_global < 0.7:
            warnings.append(f"α global bajo ({alpha_global:.3f}) — Consistencia cuestionable")
        elif alpha_global < 0.8:
            warnings.append(f"α global aceptable ({alpha_global:.3f})")

        # Alpha por dimensión
        for dim, alpha in alpha_dims.items():
            if alpha < 0.6:
                warnings.append(f"α bajo en {dim} ({alpha:.3f})")

        # Dificultad extrema
        for item, diff in item_difficulty.items():
            if diff < 0.2:
                warnings.append(f"Ítem {item} muy difícil (d={diff:.3f})")
            elif diff > 0.85:
                warnings.append(f"Ítem {item} muy fácil (d={diff:.3f})")

        return warnings
