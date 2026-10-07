"""
IRT Validator - Validacion del modelo de Teoria de Respuesta al Item.

Valida:
- Parametros GRM (discriminaciones y umbrales)
- Estimacion de theta
- T-scores (T = 50 + 10*theta)
- Clasificacion de bienestar
- Consistencia con Parametros_GRM.xlsx
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.core.config import DIMENSION_MAP, DIMENSION_ORDER, N_ITEMS, REPORTED_RESULTS
from agent.psychometrics.irt import IRTAnalyzer


@dataclass
class IRTValidationResult:
    check_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str
    severity: str = "warning"


class IRTValidator:
    """
    Validador del modelo IRT (GRM) y T-scores.
    """

    def __init__(self):
        self.analyzer = IRTAnalyzer()

    def validate_all(
        self,
        df: pd.DataFrame,
        item_cols: List[str],
        grm_params: Optional[pd.DataFrame] = None,
    ) -> List[IRTValidationResult]:
        """
        Ejecuta todas las validaciones IRT.
        """
        results = []

        # 1. Theta presente
        has_theta = "theta" in df.columns
        results.append(IRTValidationResult(
            check_name="Theta presente",
            passed=has_theta,
            expected="Columna 'theta'",
            actual="Presente" if has_theta else "Ausente",
            message="Theta es la estimacion EAP del GRM",
            severity="error",
        ))

        # 2. T_Global presente
        has_t = "T_Global" in df.columns
        results.append(IRTValidationResult(
            check_name="T_Global presente",
            passed=has_t,
            expected="Columna 'T_Global'",
            actual="Presente" if has_t else "Ausente",
            message="T_Global es el T-score global de bienestar",
            severity="error",
        ))

        # 3. T_Global media ~50
        if has_t:
            t_mean = df["T_Global"].mean()
            expected_mean = REPORTED_RESULTS["T_global_mean"]
            diff = abs(t_mean - expected_mean)
            results.append(IRTValidationResult(
                check_name="T_Global media ~50",
                passed=diff < 1.0,
                expected=f"~{expected_mean}",
                actual=f"{t_mean:.2f} (diff: {diff:.2f})",
                message="La media de T_Global debe estar cerca de 50",
                severity="warning",
            ))

        # 4. T_Global DE ~10
        if has_t:
            t_std = df["T_Global"].std()
            results.append(IRTValidationResult(
                check_name="T_Global DE ~10",
                passed=8 <= t_std <= 12,
                expected="~10",
                actual=f"{t_std:.2f}",
                message="La DE de T_Global debe estar cerca de 10",
                severity="warning",
            ))

        # 5. T-scores por dimension
        t_dims = [f"T_{d}" for d in DIMENSION_ORDER]
        found_t = [c for c in t_dims if c in df.columns]
        results.append(IRTValidationResult(
            check_name="T-scores por dimension",
            passed=len(found_t) >= 5,
            expected=">= 5 dimensiones con T-score",
            actual=f"{len(found_t)} encontradas",
            message="Se esperaban T-scores para al menos 5 dimensiones",
            severity="warning",
        ))

        # 6. Parametros GRM
        if grm_params is not None:
            results.append(IRTValidationResult(
                check_name="Parametros GRM presentes",
                passed=len(grm_params) == N_ITEMS,
                expected=N_ITEMS,
                actual=len(grm_params),
                message="Debe haber parametros para cada item",
                severity="error",
            ))

            # Verificar discriminaciones positivas
            if "a (discriminacion)" in grm_params.columns:
                disc = grm_params["a (discriminacion)"]
                all_positive = bool((disc > 0).all())
                results.append(IRTValidationResult(
                    check_name="Discriminaciones positivas",
                    passed=all_positive,
                    expected="Todas > 0",
                    actual=f"Rango: {disc.min():.3f} - {disc.max():.3f}",
                    message="Las discriminaciones deben ser positivas",
                    severity="error",
                ))

            # Verificar umbrales ordenados
            threshold_cols = [c for c in grm_params.columns if c.startswith("b")]
            if threshold_cols:
                all_ordered = True
                for _, row in grm_params.iterrows():
                    thresholds = [row[c] for c in threshold_cols if pd.notna(row[c])]
                    if len(thresholds) > 1:
                        for i in range(len(thresholds) - 1):
                            if thresholds[i] >= thresholds[i + 1]:
                                all_ordered = False
                                break
                
                results.append(IRTValidationResult(
                    check_name="Umbrales ordenados",
                    passed=all_ordered,
                    expected="b1 < b2 < b3 < b4 < b5",
                    actual="OK" if all_ordered else "Desordenados",
                    message="Los umbrales del GRM deben estar ordenados",
                    severity="warning",
                ))

        # 7. Clasificacion de bienestar
        if has_t:
            t = df["T_Global"]
            risk_pct = (t < 40).mean()
            high_pct = (t >= 60).mean()
            expected_risk = REPORTED_RESULTS["risk_proportion"]
            expected_high = REPORTED_RESULTS["high_proportion"]

            results.append(IRTValidationResult(
                check_name="Proporcion en riesgo (T<40)",
                passed=abs(risk_pct - expected_risk) < 0.05,
                expected=f"~{expected_risk*100:.1f}%",
                actual=f"{risk_pct*100:.1f}%",
                message="Proporcion de estudiantes con bienestar bajo",
                severity="warning",
            ))

            results.append(IRTValidationResult(
                check_name="Proporcion alto (T>=60)",
                passed=abs(high_pct - expected_high) < 0.05,
                expected=f"~{expected_high*100:.1f}%",
                actual=f"{high_pct*100:.1f}%",
                message="Proporcion de estudiantes con bienestar alto",
                severity="warning",
            ))

        # 8. Relacion theta-T_score
        if has_theta and has_t:
            corr = df["theta"].corr(df["T_Global"])
            results.append(IRTValidationResult(
                check_name="Correlacion theta-T_Global",
                passed=corr > 0.99,
                expected="> 0.99",
                actual=f"{corr:.4f}",
                message="T_Global debe ser una transformacion lineal de theta",
                severity="error",
            ))

            # Verificar T = 50 + 10*theta
            expected_t = 50 + 10 * df["theta"]
            diff_t = (df["T_Global"] - expected_t).abs().max()
            results.append(IRTValidationResult(
                check_name="T = 50 + 10*theta",
                passed=diff_t < 0.01,
                expected="Diferencia max < 0.01",
                actual=f"{diff_t:.4f}",
                message="T_global debe ser exactamente 50 + 10*theta",
                severity="error",
            ))

        return results

    def summary(self, results: List[IRTValidationResult]) -> str:
        """Genera resumen de validacion IRT."""
        n_passed = sum(1 for r in results if r.passed)
        n_failed = sum(1 for r in results if not r.passed)

        lines = [
            "=== VALIDACION IRT ===",
            f"Total checks: {len(results)}",
            f"Pasados: {n_passed}",
            f"Fallidos: {n_failed}",
            "",
        ]

        for r in results:
            icon = "ok" if r.passed else "X"
            lines.append(f"  [{icon}] {r.check_name}: {r.actual}")
            if not r.passed:
                lines.append(f"      Esperado: {r.expected}")

        return "\n".join(lines)
