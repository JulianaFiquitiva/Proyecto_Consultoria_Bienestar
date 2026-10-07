"""
Cargador de datasets del proyecto.

Maneja la carga de datos crudos, depurados, ponderados y de IRT.
Incluye validación básica de integridad.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.core.config import AgentConfig, DIMENSION_MAP, NEGATIVE_ITEMS


class DataLoader:
    """Carga y prepara los datasets del proyecto."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_raw(self) -> pd.DataFrame:
        """Carga el dataset crudo original (1,995 registros)."""
        path = self.config.get_data_path(self.config.raw_data_file)
        df = pd.read_excel(path)
        self._cache["raw"] = df
        return df

    def load_prepared(self) -> pd.DataFrame:
        """Carga el dataset preparado con factores de expansión (1,813 × 73)."""
        path = self.config.get_data_path(self.config.prepared_data_file)
        df = pd.read_excel(path)
        self._cache["prepared"] = df
        return df

    def load_irt(self) -> pd.DataFrame:
        """Carga el dataset con T-scores IRT (1,813 × 41)."""
        path = self.config.get_data_path(self.config.irt_data_file)
        df = pd.read_excel(path)
        self._cache["irt"] = df
        return df

    def load_grm_params(self) -> pd.DataFrame:
        """Carga los parámetros del GRM."""
        path = self.config.get_data_path(self.config.grm_params_file)
        df = pd.read_excel(path)
        self._cache["grm_params"] = df
        return df

    def load_sample_frames(self) -> Dict[str, pd.DataFrame]:
        """Carga los 6 marcos muestrales."""
        sample_files = {
            "Bogotá": "muestra_bogota_estudio_felicidad_santoto_2025.xlsx",
            "Bucaramanga_Pre": "muestra_bucaramanga_pregrado_estudio_felicidad_santoto_2025.xlsx",
            "Bucaramanga_Pos": "muestra_bucaramanga_posgrado_estudio_felicidad_santoto_2025.xlsx",
            "Tunja": "muestra_tunja_estudio_felicidad_santoto_2025.xlsx",
            "Villavicencio_Pre": "muestra_villavicencio_pregrado_estudio_felicidad_santoto_2025.xlsx",
            "Villavicencio_Pos": "muestra_villavicencio_posgrado_estudio_felicidad_santoto_2025.xlsx",
        }
        frames = {}
        for key, filename in sample_files.items():
            path = self.config.get_data_path(filename)
            frames[key] = pd.read_excel(path)
        self._cache["sample_frames"] = frames
        return frames

    def get_item_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identifica las columnas de ítems de la escala en el dataset preparado.

        Returns:
            Lista de 29 nombres de columnas de ítems.
        """
        if "Métrica Global" in df.columns:
            # Buscar delimitadores
            if "Núcleo básico de conocimiento" in df.columns:
                idx_start = list(df.columns).index("Núcleo básico de conocimiento") + 1
            elif "Programa de formación" in df.columns:
                idx_start = list(df.columns).index("Programa de formación") + 1
            else:
                idx_start = 0
            idx_end = list(df.columns).index("Métrica Global")
            item_cols = list(df.columns[idx_start:idx_end])
        else:
            # Para dataset de ejemplo o crudo: buscar columnas numéricas en rango 1-6
            candidates = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    vals = df[col].dropna()
                    if len(vals) > 0 and vals.min() >= 1 and vals.max() <= 6:
                        candidates.append(col)
            item_cols = candidates[:29]

        if len(item_cols) != 29:
            raise ValueError(f"Se esperaban 29 ítems, se encontraron {len(item_cols)}")
        return item_cols

    def get_weight_column(self, df: pd.DataFrame) -> str:
        """Identifica la columna de factores de expansión."""
        candidates = ["factor_expansion", "FExp", "peso", "weight"]
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError("No se encontró columna de factores de expansión")

    def get_demographic_columns(self, df: pd.DataFrame) -> List[str]:
        """Retorna las columnas demográficas disponibles."""
        demo_candidates = [
            "Género", "Edad", "Estrato socioeconómico",
            "Nivel de formación", "Modalidad", "Semestre",
            "seccional", "Seccional", "Programa de formación",
            "Núcleo básico de conocimiento",
        ]
        return [c for c in demo_candidates if c in df.columns]

    def invert_negative_items(
        self, df: pd.DataFrame, item_cols: List[str]
    ) -> pd.DataFrame:
        """
        Invierte los ítems con redacción negativa.

        Fórmula: valor_inv = 7 - valor_original

        Args:
            df: DataFrame con los ítems originales.
            item_cols: Lista de columnas de ítems.

        Returns:
            DataFrame con ítems invertidos (modifica in-place).
        """
        df_inv = df.copy()
        for item_num in NEGATIVE_ITEMS:
            col = item_cols[item_num - 1]
            if col in df_inv.columns:
                df_inv[col] = 7 - df_inv[col]
        return df_inv

    def compute_dimensions(
        self, df: pd.DataFrame, item_cols: List[str]
    ) -> pd.DataFrame:
        """
        Calcula las 6 dimensiones de bienestar y la métrica global.

        Args:
            df: DataFrame con ítems ya invertidos.
            item_cols: Lista de columnas de ítems.

        Returns:
            DataFrame con columnas adicionales de dimensiones.
        """
        df_out = df.copy()

        # Métrica Global (suma de 29 ítems)
        df_out["Métrica Global"] = df_out[item_cols].sum(axis=1)

        # Dimensiones (media de ítems por dimensión)
        for dim_name, item_numbers in DIMENSION_MAP.items():
            dim_cols = [item_cols[i - 1] for i in item_numbers]
            df_out[f"Dim_{dim_name}"] = df_out[dim_cols].mean(axis=1)

        return df_out

    @property
    def cached(self) -> Dict[str, pd.DataFrame]:
        """Retorna el caché de datasets cargados."""
        return self._cache
