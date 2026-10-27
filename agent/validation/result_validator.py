"""
Validador de resultados — Compara resultados reproducidos vs. reportados.

Genera tabla de comparación:
Resultado | Valor anterior | Valor reproducido | Diferencia | Estado

Estados:
- VALIDADO: coincide al redondear como se reportó (p. ej. 50.68 → 50.7)
- DIFERENCIA MENOR: no coincide al redondear, pero dentro de la tolerancia
  relativa del indicador
- DIFERENCIA IMPORTANTE
- NO REPRODUCIBLE: falta el valor reportado o no se puede calcular

Las estimaciones son POBLACIONALES: usan los factores de expansión y, para
los intervalos, la varianza del diseño estratificado (survey_design.py).
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.core.config import REPORTED_RESULTS

# Orden de presentación de los indicadores
INDICATORS = [
    "n_valid",
    "N_population",
    "T_global_mean",
    "T_global_ci_lower",
    "T_global_ci_upper",
    "alpha_cronbach",
    "risk_proportion",
    "high_proportion",
    "deff",
    "effective_sample_size",
    "pre_posgrado_T_pre",
    "pre_posgrado_T_pos",
    "pre_posgrado_gap",
]

LABELS = {
    "n_valid": "Muestra válida (n)",
    "N_population": "Población (N)",
    "T_global_mean": "T global poblacional",
    "T_global_ci_lower": "IC 95 % T global (inferior)",
    "T_global_ci_upper": "IC 95 % T global (superior)",
    "alpha_cronbach": "Alfa de Cronbach",
    "risk_proportion": "Proporción en riesgo (T < 40)",
    "high_proportion": "Proporción alta (T ≥ 60)",
    "deff": "Efecto de diseño (Kish)",
    "effective_sample_size": "Muestra efectiva",
    "pre_posgrado_T_pre": "T pregrado",
    "pre_posgrado_T_pos": "T posgrado",
    "pre_posgrado_gap": "Brecha posgrado − pregrado",
}


@dataclass
class ComparisonRow:
    """Fila de la tabla de comparación."""

    result_name: str
    reported_value: Any
    reproduced_value: Any
    difference: Optional[float]
    pct_difference: Optional[float]
    status: str  # "VALIDADO", "DIFERENCIA MENOR", "DIFERENCIA IMPORTANTE", "NO REPRODUCIBLE"
    severity: str  # "ok", "warning", "error"


class ResultValidator:
    """
    Compara resultados del pipeline con los reportados por el estudio anterior.
    """

    # Tolerancia RELATIVA para "DIFERENCIA MENOR". La coincidencia exacta
    # (VALIDADO) se decide por el redondeo del valor reportado.
    TOLERANCES = {
        "n_valid": 0.01,
        "N_population": 0.01,
        "T_global_mean": 0.01,
        "T_global_ci_lower": 0.01,
        "T_global_ci_upper": 0.01,
        "alpha_cronbach": 0.02,
        "risk_proportion": 0.10,
        "high_proportion": 0.10,
        "deff": 0.05,
        "effective_sample_size": 0.05,
        "pre_posgrado_T_pre": 0.01,
        "pre_posgrado_T_pos": 0.01,
        "pre_posgrado_gap": 0.15,
    }

    def validate_all(
        self,
        df_prepared: Optional[pd.DataFrame] = None,
        df_irt: Optional[pd.DataFrame] = None,
        reported: Optional[Dict] = None,
    ) -> List[ComparisonRow]:
        """
        Valida todos los resultados disponibles.

        Returns:
            Lista de filas de comparación.
        """
        reported = reported or REPORTED_RESULTS
        reproduced = self.reproduce(df_prepared, df_irt)
        return [
            self._compare(name, reported.get(name), reproduced[name])
            for name in INDICATORS
            if name in reproduced and name in reported
        ]

    def reproduce(
        self,
        df_prepared: Optional[pd.DataFrame] = None,
        df_irt: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Calcula los indicadores del estudio con los datos disponibles."""
        out: Dict[str, float] = {}
        base = df_prepared if df_prepared is not None else df_irt
        if base is not None:
            out["n_valid"] = len(base)

        # Efecto de diseño de Kish, con la misma convención del estudio:
        # pesos ANTES de la calibración si existen (factor_expansion_original)
        if df_prepared is not None:
            w_col = self._find_weight_col(df_prepared, prefer_original=True)
            if w_col:
                deff = self._compute_deff(df_prepared[w_col])
                out["deff"] = deff
                out["effective_sample_size"] = len(df_prepared) / deff
            alpha = self._cronbach_alpha(df_prepared)
            if alpha is not None:
                out["alpha_cronbach"] = alpha

        if df_irt is None or "T_Global" not in df_irt.columns:
            return out

        w_col = self._find_weight_col(df_irt)
        if w_col is None:
            # Sin pesos solo se puede dar la media muestral
            out["T_global_mean"] = float(df_irt["T_Global"].mean())
            return out

        t = df_irt["T_Global"]
        w = df_irt[w_col]
        out["N_population"] = float(w.sum())
        out["T_global_mean"] = float(np.average(t, weights=w))
        out["risk_proportion"] = float(np.average(t < 40, weights=w))
        out["high_proportion"] = float(np.average(t >= 60, weights=w))

        design = self._design(df_irt, w_col)
        if design is not None:
            g = design.mean("T_Global")
            out["T_global_ci_lower"] = g.ci_lower
            out["T_global_ci_upper"] = g.ci_upper

        if "modalidad_exp" in df_irt.columns:
            medias = {}
            for nivel in ("Pregrado", "Posgrado"):
                sub = df_irt[df_irt["modalidad_exp"] == nivel]
                if len(sub):
                    medias[nivel] = float(np.average(sub["T_Global"], weights=sub[w_col]))
            if len(medias) == 2:
                out["pre_posgrado_T_pre"] = medias["Pregrado"]
                out["pre_posgrado_T_pos"] = medias["Posgrado"]
                out["pre_posgrado_gap"] = medias["Posgrado"] - medias["Pregrado"]
        return out

    def compare_sources(
        self,
        sources: Dict[str, Dict],
        df_prepared: Optional[pd.DataFrame] = None,
        df_irt: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Tabla de reproducibilidad: una columna por fuente reportada, el valor
        reproducido y el estado frente a cada fuente.

        Args:
            sources: {"nombre de la fuente": diccionario de valores reportados}
        """
        reproduced = self.reproduce(df_prepared, df_irt)
        filas = []
        for name in INDICATORS:
            if name not in reproduced and not any(name in s for s in sources.values()):
                continue
            fila = {"Indicador": LABELS.get(name, name)}
            valor = reproduced.get(name)
            for fuente, valores in sources.items():
                fila[fuente] = valores.get(name, "—")
            fila["Reproducido"] = (
                round(valor, 4) if isinstance(valor, float) else valor
            )
            for fuente, valores in sources.items():
                if name in valores and valor is not None:
                    fila[f"Estado vs. {fuente}"] = self._compare(
                        name, valores[name], valor).status
                else:
                    fila[f"Estado vs. {fuente}"] = "—"
            filas.append(fila)
        return pd.DataFrame(filas)

    def generate_comparison_table(
        self,
        rows: List[ComparisonRow],
    ) -> pd.DataFrame:
        """
        Genera la tabla de comparación formateada.

        Returns:
            DataFrame con columnas:
            Resultado | Valor anterior | Valor reproducido | Diferencia | Estado
        """
        table_rows = []
        for row in rows:
            table_rows.append({
                "Resultado": row.result_name,
                "Valor anterior": row.reported_value,
                "Valor reproducido": (
                    round(row.reproduced_value, 4)
                    if isinstance(row.reproduced_value, float)
                    else row.reproduced_value
                ),
                "Diferencia": (
                    round(row.difference, 4) if row.difference is not None else "N/A"
                ),
                "% Diferencia": (
                    f"{row.pct_difference:.1f}%"
                    if row.pct_difference is not None else "N/A"
                ),
                "Estado": row.status,
            })

        return pd.DataFrame(table_rows)

    def summary(self, rows: List[ComparisonRow]) -> str:
        """Genera resumen del reporte de validación."""
        n_validated = sum(1 for r in rows if r.status == "VALIDADO")
        n_minor = sum(1 for r in rows if r.status == "DIFERENCIA MENOR")
        n_major = sum(1 for r in rows if r.status == "DIFERENCIA IMPORTANTE")
        n_not = sum(1 for r in rows if r.status == "NO REPRODUCIBLE")

        lines = [
            "=== REPORTE DE VALIDACIÓN DE RESULTADOS ===",
            f"Total comparaciones: {len(rows)}",
            f"Validados: {n_validated}",
            f"Diferencia menor: {n_minor}",
            f"Diferencia importante: {n_major}",
            f"No reproducible: {n_not}",
            "",
        ]

        for r in rows:
            icon = {
                "VALIDADO": "✓",
                "DIFERENCIA MENOR": "~",
                "DIFERENCIA IMPORTANTE": "!",
                "NO REPRODUCIBLE": "✗",
            }.get(r.status, "?")
            reproducido = (
                round(r.reproduced_value, 4)
                if isinstance(r.reproduced_value, float) else r.reproduced_value
            )
            lines.append(
                f"  {icon} {LABELS.get(r.result_name, r.result_name)}: "
                f"{r.reported_value} → {reproducido} "
                f"({r.status})"
            )

        return "\n".join(lines)

    def _compare(
        self,
        name: str,
        reported: Any,
        reproduced: Any,
    ) -> ComparisonRow:
        """Compara un resultado individual."""
        if reported is None:
            return ComparisonRow(
                result_name=name,
                reported_value="N/A",
                reproduced_value=reproduced,
                difference=None,
                pct_difference=None,
                status="NO REPRODUCIBLE",
                severity="warning",
            )

        # Calcular diferencia
        if isinstance(reported, (int, float)) and isinstance(reproduced, (int, float)):
            diff = abs(reproduced - reported)
            pct_diff = diff / abs(reported) * 100 if reported != 0 else None
        else:
            diff = None
            pct_diff = None

        # Determinar estado
        if diff is None:
            status = "NO REPRODUCIBLE"
            severity = "warning"
        elif diff <= self._rounding_tolerance(reported):
            status = "VALIDADO"
            severity = "ok"
        elif pct_diff is not None and pct_diff <= self.TOLERANCES.get(name, 0.05) * 100:
            status = "DIFERENCIA MENOR"
            severity = "warning"
        else:
            status = "DIFERENCIA IMPORTANTE"
            severity = "error"

        return ComparisonRow(
            result_name=name,
            reported_value=reported,
            reproduced_value=reproduced,
            difference=diff,
            pct_difference=pct_diff,
            status=status,
            severity=severity,
        )

    @staticmethod
    def _rounding_tolerance(reported: float) -> float:
        """
        Media unidad de la última cifra reportada: 50.7 → 0.05, 0.148 →
        0.0005, 1813 → 0.5. Un valor que redondea igual se da por validado.
        """
        # sin normalize(): 54.0 debe conservar su decimal (tolerancia 0.05)
        exponente = Decimal(str(reported)).as_tuple().exponent
        return 0.5 * 10 ** min(exponente, 0) + 1e-9

    def _find_weight_col(self, df: pd.DataFrame,
                         prefer_original: bool = False) -> Optional[str]:
        """Busca la columna de pesos."""
        candidatos = ["factor_expansion", "FExp", "peso", "weight"]
        if prefer_original:
            candidatos.insert(0, "factor_expansion_original")
        for c in candidatos:
            if c in df.columns:
                return c
        return None

    def _compute_deff(self, weights: pd.Series) -> float:
        """Efecto de diseño de Kish: DEFF = n·Σw² / (Σw)²."""
        w = weights.dropna().values
        return float(len(w) * np.sum(w ** 2) / np.sum(w) ** 2)

    @staticmethod
    def _cronbach_alpha(df_prepared: pd.DataFrame) -> Optional[float]:
        """
        Alfa sin ponderar sobre los 29 ítems, como lo reporta el estudio.
        El dataset preparado ya trae invertidos los ítems negativos.
        """
        from agent.data.loader import DataLoader
        from agent.statistics.descriptive import DescriptiveStats
        try:
            items = DataLoader().get_item_columns(df_prepared)
        except ValueError:
            return None
        datos = df_prepared[items].astype(float)
        return DescriptiveStats.weighted_cronbach_alpha(datos, np.ones(len(datos)))

    @staticmethod
    def _design(df: pd.DataFrame, w_col: str):
        """Diseño estratificado si los datos traen las columnas de estrato."""
        from agent.statistics.survey_design import SurveyDesign
        try:
            return SurveyDesign(df, weight_col=w_col)
        except ValueError:
            return None
