"""
Validador de factores de expansión y diseño muestral.

Valida:
- Factores de expansión
- Pesos
- Suma de factores
- Distribución por estrato
- Diferencias entre muestra y población
- Consistencia con el diseño muestral
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.core.config import REPORTED_RESULTS


@dataclass
class WeightValidationResult:
    """Resultado de validación de pesos."""

    check_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str
    severity: str = "warning"


class SurveyWeightValidator:
    """
    Valida la consistencia de los factores de expansión
    y el diseño muestral.
    """

    def validate(
        self,
        df: pd.DataFrame,
        weights_col: str = "factor_expansion",
        strata_col: str = "seccional",
    ) -> List[WeightValidationResult]:
        """
        Validación completa de factores de expansión.

        Args:
            df: DataFrame con datos y factores.
            weights_col: Columna de factores de expansión.
            strata_col: Columna de estratos.

        Returns:
            Lista de resultados de validación.
        """
        results = []

        # 1. Columna de pesos existe
        results.append(WeightValidationResult(
            check_name="Columna de factores presente",
            passed=weights_col in df.columns,
            expected=weights_col,
            actual=list(df.columns) if weights_col not in df.columns else "Presente",
            message="La columna de factores de expansión no existe",
            severity="error",
        ))

        if weights_col not in df.columns:
            return results

        weights = df[weights_col]

        # 2. Sin valores nulos
        n_null = weights.isna().sum()
        results.append(WeightValidationResult(
            check_name="Sin valores nulos en factores",
            passed=n_null == 0,
            expected=0,
            actual=int(n_null),
            message=f"{n_null} valores nulos en factores de expansión",
            severity="error",
        ))

        # 3. Todos positivos
        n_nonpositive = (weights.dropna() <= 0).sum()
        results.append(WeightValidationResult(
            check_name="Factores positivos",
            passed=n_nonpositive == 0,
            expected=0,
            actual=int(n_nonpositive),
            message=f"{n_nonpositive} factores no positivos",
            severity="error",
        ))

        # 4. Rango razonable
        w_min, w_max = weights.min(), weights.max()
        results.append(WeightValidationResult(
            check_name="Rango de factores razonable",
            passed=0.1 < w_min and w_max < 100,
            expected="0.1 < FExp < 100",
            actual=f"{w_min:.2f} – {w_max:.2f}",
            message="Los factores están fuera de un rango esperado",
            severity="warning",
        ))

        # 5. Suma de factores ≈ N poblacional
        total_w = weights.sum()
        expected_n = REPORTED_RESULTS["N_population"]
        pct_diff = abs(total_w - expected_n) / expected_n * 100
        results.append(WeightValidationResult(
            check_name="Suma de factores ≈ N",
            passed=pct_diff < 5,
            expected=f"~{expected_n:,}",
            actual=f"{total_w:,.0f} (diff: {pct_diff:.1f}%)",
            message="La suma de factores no se acerca a la población",
            severity="error",
        ))

        # 6. Distribución por estrato
        if strata_col in df.columns:
            strata_totals = df.groupby(strata_col)[weights_col].sum()
            for strata, total in strata_totals.items():
                if total <= 0:
                    results.append(WeightValidationResult(
                        check_name=f"Estrato '{strata}' con peso total > 0",
                        passed=False,
                        expected="> 0",
                        actual=float(total),
                        message=f"El estrato '{strata}' tiene peso total ≤ 0",
                        severity="error",
                    ))

            # 7. DEFF (Design Effect)
            deff = self._compute_deff(weights)
            results.append(WeightValidationResult(
                check_name="DEFF razonable",
                passed=0.5 < deff < 5,
                expected="0.5 < DEFF < 5",
                actual=f"{deff:.3f}",
                message="DEFF fuera de rango esperado",
                severity="warning",
            ))

            # 8. Muestra efectiva
            n_eff = len(df) / deff
            results.append(WeightValidationResult(
                check_name="Muestra efectiva razonable",
                passed=n_eff > len(df) * 0.5,
                expected=f"> {len(df) * 0.5:,.0f}",
                actual=f"{n_eff:,.0f}",
                message="Muestra efectiva muy baja",
                severity="warning",
            ))

        return results

    def compare_sample_population(
        self,
        sample_df: pd.DataFrame,
        pop_totals: Dict[str, float],
        strata_col: str,
        weights_col: str = "factor_expansion",
    ) -> pd.DataFrame:
        """
        Compara distribución de muestra vs. población por estrato.

        Args:
            sample_df: DataFrame de la muestra.
            pop_totals: Dict con totales poblacionales por estrato.
            strata_col: Columna de estratos.
            weights_col: Columna de factores.

        Returns:
            DataFrame con comparación.
        """
        sample_counts = sample_df[strata_col].value_counts()
        sample_weighted = sample_df.groupby(strata_col)[weights_col].sum()

        rows = []
        for strata in pop_totals:
            n_sample = sample_counts.get(strata, 0)
            weighted = sample_weighted.get(strata, 0)
            pop = pop_totals[strata]
            pct_sample = n_sample / len(sample_df) * 100
            pct_pop = pop / sum(pop_totals.values()) * 100
            diff = pct_sample - pct_pop

            rows.append({
                "Estrato": strata,
                "n_muestra": n_sample,
                "%_muestra": round(pct_sample, 1),
                "Peso_total": round(weighted, 0),
                "%_población": round(pct_pop, 1),
                "Diferencia_%": round(diff, 1),
            })

        return pd.DataFrame(rows)

    def _compute_deff(self, weights: pd.Series) -> float:
        """
        Calcula el Design Effect (DEFF).
        DEFF = (Σ w_i)² / (Σ w_i²)
        """
        w = weights.dropna().values
        return float(np.sum(w) ** 2 / np.sum(w ** 2))
