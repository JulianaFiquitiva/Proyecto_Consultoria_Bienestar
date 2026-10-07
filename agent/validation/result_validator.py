"""
Validador de resultados — Compara resultados reproducidos vs. reportados.

Genera tabla de comparación:
Resultado | Valor anterior | Valor reproducido | Diferencia | Estado

Estados:
- VALIDADO
- DIFERENCIA MENOR
- DIFERENCIA IMPORTANTE
- NO REPRODUCIBLE
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.core.config import REPORTED_RESULTS


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

    # Tolerancias por tipo de resultado
    TOLERANCES = {
        "n_valid": {"absolute": 5, "relative": 0.01},
        "T_global_mean": {"absolute": 0.5, "relative": 0.02},
        "alpha_cronbach": {"absolute": 0.02, "relative": 0.03},
        "risk_proportion": {"absolute": 0.02, "relative": 0.10},
        "high_proportion": {"absolute": 0.02, "relative": 0.10},
        "deff": {"absolute": 0.1, "relative": 0.10},
        "pre_posgrado_gap": {"absolute": 1.0, "relative": 0.15},
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
        rows = []

        # 1. Número de registros
        if df_prepared is not None:
            rows.append(self._compare(
                "n_valid",
                reported.get("n_valid"),
                len(df_prepared),
            ))

        # 2. T-score Global
        if df_irt is not None and "T_Global" in df_irt.columns:
            rows.append(self._compare(
                "T_global_mean",
                reported.get("T_global_mean"),
                df_irt["T_Global"].mean(),
            ))

        # 3. DEFF
        if df_prepared is not None:
            w_col = self._find_weight_col(df_prepared)
            if w_col:
                deff = self._compute_deff(df_prepared[w_col])
                rows.append(self._compare(
                    "deff",
                    reported.get("deff"),
                    deff,
                ))

        # 4. Proporciones de bienestar
        if df_irt is not None and "T_Global" in df_irt.columns:
            t = df_irt["T_Global"]
            risk = (t < 40).mean()
            high = (t >= 60).mean()

            rows.append(self._compare(
                "risk_proportion",
                reported.get("risk_proportion"),
                risk,
            ))
            rows.append(self._compare(
                "high_proportion",
                reported.get("high_proportion"),
                high,
            ))

        return rows

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
            lines.append(
                f"  {icon} {r.result_name}: "
                f"{r.reported_value} → {r.reproduced_value} "
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

        # Obtener tolerancia
        tol = self.TOLERANCES.get(name, {"absolute": 0.1, "relative": 0.10})

        # Determinar estado
        if diff is None:
            status = "NO REPRODUCIBLE"
            severity = "warning"
        elif diff <= tol["absolute"]:
            status = "VALIDADO"
            severity = "ok"
        elif pct_diff is not None and pct_diff <= tol["relative"] * 100:
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

    def _find_weight_col(self, df: pd.DataFrame) -> Optional[str]:
        """Busca la columna de pesos."""
        for c in ["factor_expansion", "FExp", "peso", "weight"]:
            if c in df.columns:
                return c
        return None

    def _compute_deff(self, weights: pd.Series) -> float:
        """Calcula DEFF."""
        w = weights.dropna().values
        return float(np.sum(w) ** 2 / np.sum(w ** 2))
