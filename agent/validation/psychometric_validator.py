"""
Psychometric Validator - Validacion completa del analisis psicometrico.

Valida:
- 29 items de la escala de Ryff
- Items invertidos (10 items negativos)
- 6 dimensiones
- Alpha de Cronbach global y por dimension
- Correlaciones
- Distribucion de respuestas
- Consistencia interna
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.core.config import (
    DIMENSION_MAP,
    DIMENSION_ORDER,
    ITEM_LABELS,
    NEGATIVE_ITEMS,
    N_ITEMS,
    N_RESPONSE_CATEGORIES,
    REPORTED_RESULTS,
)
from agent.psychometrics.scale import PsychometricScale
from agent.statistics.descriptive import DescriptiveStats


@dataclass
class PsychometricValidationResult:
    check_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str
    severity: str = "warning"


class PsychometricValidator:
    """
    Validador completo del analisis psicometrico de la escala de Ryff.
    """

    def __init__(self):
        self.scale = PsychometricScale()
        self.desc = DescriptiveStats()

    def validate_all(
        self,
        df: pd.DataFrame,
        item_cols: List[str],
        weights_col: Optional[str] = None,
    ) -> List[PsychometricValidationResult]:
        """
        Ejecuta todas las validaciones psicometricas.
        """
        results = []

        # 1. Numero de items
        results.append(PsychometricValidationResult(
            check_name="Numero de items",
            passed=len(item_cols) == N_ITEMS,
            expected=N_ITEMS,
            actual=len(item_cols),
            message=f"Se esperaban {N_ITEMS} items, se encontraron {len(item_cols)}",
            severity="error",
        ))

        # 2. Rango de items (1-6)
        all_vals = df[item_cols].values.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        in_range = bool(np.all((all_vals >= 1) & (all_vals <= 6)))
        results.append(PsychometricValidationResult(
            check_name="Rango de items (1-6)",
            passed=in_range,
            expected="Todos los valores entre 1 y 6",
            actual=f"Rango: {all_vals.min():.0f} - {all_vals.max():.0f}",
            message="Los items Likert deben estar en el rango 1-6",
            severity="error",
        ))

        # 3. Items negativos invertidos
        inversion_ok = self._check_item_inversion(df, item_cols)
        results.append(PsychometricValidationResult(
            check_name="Items negativos invertidos",
            passed=inversion_ok,
            expected=f"{len(NEGATIVE_ITEMS)} items invertidos",
            actual="OK" if inversion_ok else "ERROR",
            message="Verificar que los items negativos fueron invertidos con formula 7-valor",
            severity="error",
        ))

        # 4. Dimensiones presentes
        dim_cols_expected = [f"Dim_{d}" for d in DIMENSION_ORDER]
        dim_cols_found = [c for c in dim_cols_expected if c in df.columns]
        results.append(PsychometricValidationResult(
            check_name="6 dimensiones presentes",
            passed=len(dim_cols_found) == 6,
            expected=6,
            actual=len(dim_cols_found),
            message=f"Se encontraron {len(dim_cols_found)} de 6 dimensiones",
            severity="error",
        ))

        # 5. Alpha de Cronbach
        weights = df[weights_col].values if weights_col and weights_col in df.columns else np.ones(len(df))
        if weights_col and weights_col in df.columns:
            alpha = self.desc.weighted_cronbach_alpha(df[item_cols], weights)
        else:
            alpha = self.scale._cronbach_alpha(df[item_cols])
        
        expected_alpha = REPORTED_RESULTS["alpha_cronbach"]
        diff_alpha = abs(alpha - expected_alpha)
        results.append(PsychometricValidationResult(
            check_name="Alpha de Cronbach",
            passed=diff_alpha < 0.05,
            expected=f"~{expected_alpha:.3f}",
            actual=f"{alpha:.3f} (diff: {diff_alpha:.3f})",
            message="Alpha de Cronbach global",
            severity="warning",
        ))

        # 6. Alpha por dimension
        for dim_name, item_numbers in DIMENSION_MAP.items():
            dim_cols = [item_cols[i - 1] for i in item_numbers]
            if all(c in df.columns for c in dim_cols):
                if weights_col and weights_col in df.columns:
                    dim_alpha = self.desc.weighted_cronbach_alpha(df[dim_cols], weights)
                else:
                    dim_alpha = self.scale._cronbach_alpha(df[dim_cols])
                
                results.append(PsychometricValidationResult(
                    check_name=f"Alpha dimension: {dim_name}",
                    passed=dim_alpha >= 0.6,
                    expected=">= 0.60",
                    actual=f"{dim_alpha:.3f}",
                    message=f"Alpha de Cronbach para {dim_name}",
                    severity="warning",
                ))

        # 7. Distribucion de respuestas (no debe haber sesgo extremo)
        for i, col in enumerate(item_cols, 1):
            freq = df[col].value_counts(normalize=True)
            max_freq = freq.max()
            if max_freq > 0.7:
                results.append(PsychometricValidationResult(
                    check_name=f"Distribucion item {i}",
                    passed=False,
                    expected="< 70% en una categoria",
                    actual=f"{max_freq*100:.1f}% en una categoria",
                    message=f"Item {i} tiene sesgo extremo",
                    severity="warning",
                ))

        # 8. Correlaciones items-total
        total = df[item_cols].sum(axis=1)
        low_corr_items = []
        for i, col in enumerate(item_cols, 1):
            corr = df[col].corr(total)
            if corr < 0.2:
                low_corr_items.append(i)
        
        results.append(PsychometricValidationResult(
            check_name="Correlaciones item-total",
            passed=len(low_corr_items) == 0,
            expected="Todos los items r > 0.2",
            actual=f"{len(low_corr_items)} items con r < 0.2" if low_corr_items else "OK",
            message=f"Items con baja correlacion: {low_corr_items}" if low_corr_items else "Todos los items correlacionan bien con el total",
            severity="warning",
        ))

        return results

    def _check_item_inversion(
        self, df: pd.DataFrame, item_cols: List[str]
    ) -> bool:
        """
        Verifica si los items negativos estan invertidos.
        
        Para items invertidos, la media debe ser > 3.5 (porque la escala es 1-6
        y los items negativos originales tienen media < 3.5).
        """
        for item_num in NEGATIVE_ITEMS:
            if item_num <= len(item_cols):
                col = item_cols[item_num - 1]
                if col in df.columns:
                    mean_val = df[col].mean()
                    # Si la media es muy baja, probablemente no fue invertido
                    if mean_val < 2.5:
                        return False
        return True

    def summary(self, results: List[PsychometricValidationResult]) -> str:
        """Genera resumen de validacion psicometrica."""
        n_passed = sum(1 for r in results if r.passed)
        n_failed = sum(1 for r in results if not r.passed)
        n_warnings = sum(1 for r in results if r.severity == "warning" and not r.passed)

        lines = [
            "=== VALIDACION PSICOMETRICA ===",
            f"Total checks: {len(results)}",
            f"Pasados: {n_passed}",
            f"Fallidos: {n_failed}",
            f"Advertencias: {n_warnings}",
            "",
        ]

        for r in results:
            icon = "ok" if r.passed else ("~" if r.severity == "warning" else "X")
            lines.append(f"  [{icon}] {r.check_name}: {r.actual}")
            if not r.passed:
                lines.append(f"      Esperado: {r.expected}")
                lines.append(f"      {r.message}")

        return "\n".join(lines)
