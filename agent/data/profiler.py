"""
Data Profiler — Perfil automático de datasets.

Detecta: columnas, tipos, valores faltantes, categorías, tamaños de muestra,
duplicados, rangos, variables sensibles y variables disponibles para análisis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Perfil de una columna individual."""

    name: str
    dtype: str
    n_total: int
    n_missing: int
    pct_missing: float
    n_unique: int
    is_categorical: bool
    categories: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    is_sensitive: bool = False  # PII o información personal


@dataclass
class DatasetProfile:
    """Perfil completo de un dataset."""

    n_rows: int
    n_columns: int
    n_duplicates: int
    pct_duplicates: float
    columns: List[ColumnProfile]
    missing_columns: List[str]
    sensitive_columns: List[str]
    categorical_columns: List[str]
    numeric_columns: List[str]
    total_missing_cells: int
    pct_missing_cells: float


# Patrones de detección de variables sensibles (PII)
SENSITIVE_PATTERNS = [
    "nombre", "name", "correo", "email", "teléfono", "phone",
    "cédula", "cc", "documento", "identificación", "address",
    "dirección", "ip", "password", "secret",
]


class DataProfiler:
    """
    Genera un perfil automático de un dataset.

    Detecta automáticamente:
    - Tipos de columnas
    - Valores faltantes
    - Categorías
    - Rangos
    - Variables sensibles
    - Duplicados
    """

    def __init__(self):
        self._profile: Optional[DatasetProfile] = None

    def profile(self, df: pd.DataFrame, name: str = "dataset") -> DatasetProfile:
        """
        Genera el perfil completo del dataset.

        Args:
            df: DataFrame a perfilar.
            name: Nombre descriptivo del dataset.

        Returns:
            DatasetProfile con toda la información detectada.
        """
        columns = []
        missing_cols = []
        sensitive_cols = []
        cat_cols = []
        num_cols = []

        for col in df.columns:
            col_profile = self._profile_column(col, df[col])
            columns.append(col_profile)

            if col_profile.pct_missing > 0:
                missing_cols.append(col)
            if col_profile.is_sensitive:
                sensitive_cols.append(col)
            if col_profile.is_categorical:
                cat_cols.append(col)
            else:
                num_cols.append(col)

        total_cells = df.shape[0] * df.shape[1]
        total_missing = int(df.isna().sum().sum())

        self._profile = DatasetProfile(
            n_rows=df.shape[0],
            n_columns=df.shape[1],
            n_duplicates=int(df.duplicated().sum()),
            pct_duplicates=float(df.duplicated().mean() * 100),
            columns=columns,
            missing_columns=missing_cols,
            sensitive_columns=sensitive_cols,
            categorical_columns=cat_cols,
            numeric_columns=num_cols,
            total_missing_cells=total_missing,
            pct_missing_cells=float(total_missing / total_cells * 100) if total_cells > 0 else 0,
        )
        return self._profile

    def _profile_column(self, name: str, series: pd.Series) -> ColumnProfile:
        """Perfila una columna individual."""
        n_total = len(series)
        n_missing = int(series.isna().sum())
        n_unique = series.nunique()

        # Detectar si es categórica
        is_cat = False
        categories = None
        if series.dtype == "object" or n_unique <= 20:
            is_cat = True
            categories = series.dropna().unique().tolist()
            if len(categories) > 100:
                is_cat = False
                categories = None

        # Detectar si es sensible
        name_lower = name.lower()
        is_sensitive = any(p in name_lower for p in SENSITIVE_PATTERNS)

        # Estadísticos numéricos
        min_val = max_val = mean_val = median_val = None
        if pd.api.types.is_numeric_dtype(series):
            min_val = float(series.min()) if not series.empty else None
            max_val = float(series.max()) if not series.empty else None
            mean_val = float(series.mean()) if not series.empty else None
            median_val = float(series.median()) if not series.empty else None

        return ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            n_total=n_total,
            n_missing=n_missing,
            pct_missing=float(n_missing / n_total * 100) if n_total > 0 else 0,
            n_unique=n_unique,
            is_categorical=is_cat,
            categories=categories,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            median_value=median_val,
            is_sensitive=is_sensitive,
        )

    def summary(self, profile: Optional[DatasetProfile] = None) -> str:
        """Genera un resumen en texto del perfil."""
        p = profile or self._profile
        if p is None:
            return "No hay perfil generado. Ejecute profile() primero."

        lines = [
            "=== PERFIL DEL DATASET ===",
            f"Filas: {p.n_rows}",
            f"Columnas: {p.n_columns}",
            f"Duplicados: {p.n_duplicates} ({p.pct_duplicates:.1f}%)",
            f"Celdas faltantes: {p.total_missing_cells} ({p.pct_missing_cells:.1f}%)",
            f"Columnas con faltantes: {len(p.missing_columns)}",
            f"Columnas sensibles (PII): {len(p.sensitive_columns)}",
            f"Columnas categóricas: {len(p.categorical_columns)}",
            f"Columnas numéricas: {len(p.numeric_columns)}",
            "",
            "--- Columnas ---",
        ]
        for col in p.columns:
            marker = " [SENSIBLE]" if col.is_sensitive else ""
            falt = f" ({col.pct_missing:.1f}% faltante)" if col.pct_missing > 0 else ""
            lines.append(f"  {col.name}: {col.dtype} | {col.n_unique} únicos{falt}{marker}")

        return "\n".join(lines)

    def detect_available_analyses(
        self, df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Detecta qué análisis son posibles con las variables disponibles.

        Returns:
            Diccionario con categorías de análisis y variables disponibles.
        """
        analyses = {
            "descriptive_by_group": [],
            "comparison_two_groups": [],
            "correlation": [],
            "regression": [],
            "irt": [],
        }

        # Identificar variables demográficas
        demo_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ["género", "genero", "sexo", "gender"]):
                demo_cols.append(col)
                analyses["comparison_two_groups"].append(col)
            elif any(k in col_lower for k in ["seccional", "sede", "campus"]):
                demo_cols.append(col)
                analyses["descriptive_by_group"].append(col)
                analyses["comparison_two_groups"].append(col)
            elif any(k in col_lower for k in ["modalidad", "nivel", "formación"]):
                demo_cols.append(col)
                analyses["descriptive_by_group"].append(col)
                analyses["comparison_two_groups"].append(col)
            elif any(k in col_lower for k in ["estrato", "socieconómico"]):
                demo_cols.append(col)
                analyses["descriptive_by_group"].append(col)
            elif any(k in col_lower for k in ["edad", "age"]):
                demo_cols.append(col)
                analyses["correlation"].append(col)
                analyses["regression"].append(col)

        # Detectar ítems (numéricos, rango 1-6)
        item_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].dropna()
                if len(vals) > 0 and vals.min() >= 1 and vals.max() <= 6:
                    item_cols.append(col)

        if len(item_cols) >= 5:
            analyses["irt"] = item_cols

        # Variables de resultado (T-scores, métrica global)
        outcome_cols = [c for c in df.columns if any(
            k in c.lower() for k in ["t_global", "t_", "métrica", "metrica", "theta"]
        )]
        if outcome_cols:
            analyses["regression"].extend(outcome_cols)

        return analyses
