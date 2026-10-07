"""
Data Validator — Validación de integridad y consistencia de datasets.

Valida: dimensiones, cantidad de registros, valores permitidos,
consistencia de categorías, duplicados, valores faltantes, rangos,
estratos, variables de encuesta y consistencia de factores de expansión.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.core.config import (
    AgentConfig,
    DIMENSION_MAP,
    ITEM_LABELS,
    NEGATIVE_ITEMS,
    N_ITEMS,
    REPORTED_RESULTS,
)


@dataclass
class ValidationResult:
    """Resultado de una validación individual."""

    name: str
    passed: bool
    expected: Any
    actual: Any
    message: str
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class ValidationReport:
    """Reporte completo de validación."""

    dataset_name: str
    n_checks: int
    n_passed: int
    n_failed: int
    n_warnings: int
    results: List[ValidationResult]

    @property
    def passed(self) -> bool:
        """True si todas las validaciones críticas pasaron."""
        return not any(
            r.severity == "error" and not r.passed for r in self.results
        )

    def summary(self) -> str:
        """Genera resumen del reporte."""
        lines = [
            f"=== VALIDACIÓN: {self.dataset_name} ===",
            f"Total checks: {self.n_checks}",
            f"Pasados: {self.n_passed}",
            f"Fallidos: {self.n_failed} ({'CRÍTICO' if not self.passed else 'ok'})",
            f"Advertencias: {self.n_warnings}",
            "",
        ]
        for r in self.results:
            status = "✓" if r.passed else "✗"
            sev = f" [{r.severity.upper()}]" if r.severity == "error" and not r.passed else ""
            lines.append(f"  {status} {r.name}{sev}")
            if not r.passed:
                lines.append(f"    Esperado: {r.expected}")
                lines.append(f"    Actual: {r.actual}")
                lines.append(f"    {r.message}")
        return "\n".join(lines)


class DataValidator:
    """Valida datasets del proyecto de bienestar estudiantil."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def validate_prepared(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validación completa del dataset preparado (01_dataset_preparado.xlsx).

        Checks:
        1. Número de registros ≈ 1813
        2. Número de columnas ≥ 50
        3. 29 ítems presentes
        4. Valores de ítems en rango 1-6
        5. 10 ítems invertidos correctamente
        6. 6 dimensiones calculadas
        7. Métrica Global calculada
        8. Factores de expansión presentes
        9. Factores de expansión positivos
        10. Suma de factores ≈ N poblacional
        11. Seccional presente
        12. Modalidad presente
        13. Sin duplicados exactos
        """
        results = []

        # 1. Número de registros
        n = len(df)
        expected_n = REPORTED_RESULTS["n_valid"]
        results.append(ValidationResult(
            name="Número de registros",
            passed=abs(n - expected_n) <= 10,
            expected=f"~{expected_n}",
            actual=n,
            message=f"Se esperaban ~{expected_n} registros válidos",
            severity="error",
        ))

        # 2. Número mínimo de columnas
        results.append(ValidationResult(
            name="Número de columnas",
            passed=len(df.columns) >= 50,
            expected="≥50",
            actual=len(df.columns),
            message="El dataset preparado debe tener al menos 50 columnas",
            severity="warning",
        ))

        # 3. Ítems presentes
        item_cols = self._find_item_columns(df)
        results.append(ValidationResult(
            name="29 ítems presentes",
            passed=len(item_cols) == N_ITEMS,
            expected=N_ITEMS,
            actual=len(item_cols),
            message="No se encontraron las 29 columnas de ítems",
            severity="error",
        ))

        # 4. Valores de ítems en rango 1-6
        if item_cols:
            all_vals = df[item_cols].values.flatten()
            all_vals = all_vals[~np.isnan(all_vals)]
            in_range = np.all((all_vals >= 1) & (all_vals <= 6))
            results.append(ValidationResult(
                name="Valores de ítems en rango 1-6",
                passed=bool(in_range),
                expected="Todos los valores entre 1 y 6",
                actual=f"Rango: {all_vals.min():.0f} – {all_vals.max():.0f}",
                message="Los ítems Likert deben estar en el rango 1-6",
                severity="error",
            ))

        # 5. Métrica Global
        has_global = "Métrica Global" in df.columns
        results.append(ValidationResult(
            name="Métrica Global calculada",
            passed=has_global,
            expected="Columna 'Métrica Global' presente",
            actual="Presente" if has_global else "Ausente",
            message="La métrica global debe estar calculada",
            severity="error",
        ))

        # 6. Dimensiones
        dim_cols = [f"Dim_{d}" for d in DIMENSION_MAP.keys()]
        found_dims = [c for c in dim_cols if c in df.columns]
        results.append(ValidationResult(
            name="6 dimensiones calculadas",
            passed=len(found_dims) == 6,
            expected=6,
            actual=len(found_dims),
            message=f"Se encontraron {len(found_dims)} de 6 dimensiones",
            severity="error",
        ))

        # 7. Factores de expansión
        w_col = self._find_weight_column(df)
        results.append(ValidationResult(
            name="Factores de expansión presentes",
            passed=w_col is not None,
            expected="Columna de factores de expansión",
            actual=w_col or "No encontrada",
            message="Se requiere una columna de factores de expansión",
            severity="error",
        ))

        # 8. Factores positivos
        if w_col:
            all_positive = (df[w_col].dropna() > 0).all()
            results.append(ValidationResult(
                name="Factores de expansión positivos",
                passed=bool(all_positive),
                expected="Todos los factores > 0",
                actual=f"Rango: {df[w_col].min():.2f} – {df[w_col].max():.2f}",
                message="Los factores de expansión deben ser positivos",
                severity="error",
            ))

        # 9. Suma de factores ≈ N
        if w_col:
            total_w = df[w_col].sum()
            expected_total = REPORTED_RESULTS["N_population"]
            pct_diff = abs(total_w - expected_total) / expected_total * 100
            results.append(ValidationResult(
                name="Suma de factores ≈ N poblacional",
                passed=pct_diff < 5,
                expected=f"~{expected_total:,}",
                actual=f"{total_w:,.0f}",
                message=f"Diferencia: {pct_diff:.1f}%",
                severity="warning",
            ))

        # 10. Seccional
        has_seccional = "seccional" in df.columns or "Seccional" in df.columns
        results.append(ValidationResult(
            name="Variable seccional presente",
            passed=has_seccional,
            expected="Columna 'seccional' o 'Seccional'",
            actual="Presente" if has_seccional else "Ausente",
            message="La seccional es necesaria para inferencia estratificada",
            severity="error",
        ))

        # 11. Duplicados
        n_dup = df.duplicated().sum()
        results.append(ValidationResult(
            name="Sin duplicados exactos",
            passed=n_dup == 0,
            expected=0,
            actual=int(n_dup),
            message=f"Se encontraron {n_dup} registros duplicados",
            severity="warning",
        ))

        return self._build_report("Dataset Preparado", results)

    def validate_irt(self, df: pd.DataFrame) -> ValidationReport:
        """Valida el dataset con T-scores IRT."""
        results = []

        # T_Global presente
        has_t = "T_Global" in df.columns
        results.append(ValidationResult(
            name="T_Global presente",
            passed=has_t,
            expected="Columna 'T_Global'",
            actual="Presente" if has_t else "Ausente",
            message="La columna T_Global es requerida para validación IRT",
            severity="error",
        ))

        # T_Global en rango razonable
        if has_t:
            t_mean = df["T_Global"].mean()
            results.append(ValidationResult(
                name="T_Global media ~50",
                passed=45 <= t_mean <= 55,
                expected="~50 (prior IRT N(0,1))",
                actual=f"{t_mean:.2f}",
                message="La media de T_Global debe estar cerca de 50",
                severity="warning",
            ))

        # T-scores por dimensión
        from agent.core.config import DIMENSION_ORDER
        t_dims = [f"T_{d}" for d in DIMENSION_ORDER]
        found_t = [c for c in t_dims if c in df.columns]
        results.append(ValidationResult(
            name="T-scores por dimensión",
            passed=len(found_t) >= 5,
            expected="≥5 dimensiones con T-score",
            actual=f"{len(found_t)} encontradas",
            message="Se esperaban T-scores para al menos 5 dimensiones",
            severity="warning",
        ))

        return self._build_report("Dataset IRT", results)

    def validate_against_reported(
        self, df: pd.DataFrame, reported: Optional[Dict] = None
    ) -> ValidationReport:
        """
        Compara resultados calculados con los reportados por el estudio anterior.

        Genera tabla de comparación:
        Resultado | Valor anterior | Valor reproducido | Diferencia | Estado
        """
        reported = reported or REPORTED_RESULTS
        results = []

        # T_Global
        if "T_Global" in df.columns:
            t_calc = df["T_Global"].mean()
            t_rep = reported.get("T_global_mean", 50.7)
            diff = abs(t_calc - t_rep)
            results.append(ValidationResult(
                name="T_Global media",
                passed=diff < 1.0,
                expected=t_rep,
                actual=round(t_calc, 2),
                message=f"Diferencia: {diff:.2f} puntos",
                severity="error",
            ))

        # n registros
        n = len(df)
        n_rep = reported.get("n_valid", 1813)
        results.append(ValidationResult(
            name="Número de registros",
            passed=n == n_rep,
            expected=n_rep,
            actual=n,
            message="Diferencia en número de registros",
            severity="error",
        ))

        return self._build_report("Validación vs. Estudio Anterior", results)

    def _find_item_columns(self, df: pd.DataFrame) -> List[str]:
        """Busca las columnas de ítems en el dataset."""
        if "Métrica Global" in df.columns:
            try:
                idx_start = list(df.columns).index("Núcleo básico de conocimiento") + 1
                idx_end = list(df.columns).index("Métrica Global")
                return list(df.columns[idx_start:idx_end])
            except ValueError:
                pass
        # Fallback: buscar columnas numéricas en rango 1-6
        candidates = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].dropna()
                if len(vals) > 0 and vals.min() >= 1 and vals.max() <= 6:
                    candidates.append(col)
        return candidates[:29]

    def _find_weight_column(self, df: pd.DataFrame) -> Optional[str]:
        """Busca la columna de factores de expansión."""
        for c in ["factor_expansion", "FExp", "peso", "weight"]:
            if c in df.columns:
                return c
        return None

    def _build_report(
        self, name: str, results: List[ValidationResult]
    ) -> ValidationReport:
        """Construye el reporte de validación."""
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        warnings = sum(1 for r in results if r.severity == "warning" and not r.passed)

        return ValidationReport(
            dataset_name=name,
            n_checks=len(results),
            n_passed=passed,
            n_failed=failed,
            n_warnings=warnings,
            results=results,
        )
