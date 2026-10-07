"""
Statistical Engine - Motor de seleccion automatica de metodos estadisticos.

Selecciona el metodo apropiado evaluando:
- Tipo de variable (numerica, categorica, ordinal)
- Escala de medicion
- Independencia de observaciones
- Normalidad
- Heterocedasticidad
- Tamano de muestra
- Diseno muestral
- Presencia de pesos

NO selecciona pruebas por popularidad.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from agent.statistics.descriptive import DescriptiveStats
from agent.statistics.inference import InferenceEngine, TestResult
from agent.statistics.regression import RegressionEngine, RegressionResult


@dataclass
class AssumptionCheck:
    name: str
    test_name: str
    statistic: float
    p_value: float
    passed: bool
    interpretation: str


@dataclass
class AnalysisPlan:
    method: str
    test_name: str
    reason: str
    assumptions_checked: List[AssumptionCheck]
    alternative: str
    warnings: List[str]


class StatisticalEngine:
    """
    Motor estadistico con seleccion automatica de metodos.
    
    Evalua supuestos antes de seleccionar la prueba.
    """
    
    def __init__(self):
        self.desc = DescriptiveStats()
        self.inference = InferenceEngine()
        self.regression = RegressionEngine()
    
    def select_comparison_method(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        weights_col: Optional[str] = None,
        alpha: float = 0.05,
    ) -> AnalysisPlan:
        """
        Selecciona el metodo de comparacion apropiado.
        
        Flujo:
        1. Verificar tipo de variable
        2. Verificar tamano de muestra por grupo
        3. Verificar normalidad (Shapiro-Wilk)
        4. Verificar homogeneidad de varianzas (Levene)
        5. Seleccionar prueba
        """
        warnings = []
        assumptions = []
        
        data = df[[value_col, group_col]].dropna()
        groups = data[group_col].unique()
        
        if len(groups) < 2:
            return AnalysisPlan(
                method="none",
                test_name="N/A",
                reason="Se necesitan al menos 2 grupos",
                assumptions_checked=[],
                alternative="N/A",
                warnings=[f"Solo {len(groups)} grupo(s) encontrado(s)"],
            )
        
        group_sizes = {g: len(data[data[group_col] == g]) for g in groups}
        
        # Verificar tamano minimo
        min_size = min(group_sizes.values())
        if min_size < 5:
            warnings.append(f"Grupo con muy pocos casos (n={min_size}), resultados poco confiables")
        
        # Verificar si hay pesos
        has_weights = weights_col is not None and weights_col in df.columns
        
        if has_weights:
            warnings.append("Se detectaron pesos de diseno muestral. Las pruebas estandar no los incorporan directamente.")
        
        # Verificar normalidad por grupo
        normality_results = []
        all_normal = True
        for g in groups:
            g_data = data[data[group_col] == g][value_col].values
            if len(g_data) >= 3 and len(g_data) <= 5000:
                stat, p = sp_stats.shapiro(g_data)
                is_normal = p > alpha
                normality_results.append(AssumptionCheck(
                    name=f"Normalidad en {g}",
                    test_name="Shapiro-Wilk",
                    statistic=round(stat, 4),
                    p_value=round(p, 4),
                    passed=is_normal,
                    interpretation="Normal" if is_normal else "No normal",
                ))
                if not is_normal:
                    all_normal = False
            else:
                warnings.append(f"Grupo '{g}': n={len(g_data)} fuera de rango para Shapiro-Wilk")
                all_normal = False
        
        # Verificar homogeneidad de varianzas (Levene)
        group_data_list = [data[data[group_col] == g][value_col].values for g in groups]
        if all(len(g) >= 2 for g in group_data_list):
            stat_levene, p_levene = sp_stats.levene(*group_data_list)
            homoscedastic = p_levene > alpha
            assumptions.append(AssumptionCheck(
                name="Homogeneidad de varianzas",
                test_name="Levene",
                statistic=round(stat_levene, 4),
                p_value=round(p_levene, 4),
                passed=homoscedastic,
                interpretation="Varianzas iguales" if homoscedastic else "Varianzas diferentes",
            ))
        else:
            homoscedastic = False
            warnings.append("Grupos muy pequenos para evaluar homogeneidad de varianzas")
        
        all_assumptions = normality_results + assumptions
        
        # Seleccionar prueba
        if len(groups) == 2:
            if all_normal and homoscedastic:
                method = "t-test_student"
                test_name = "t-test de Student"
                reason = "Normalidad y homogeneidad de varianzas cumplidas"
                alternative = "Mann-Whitney U (si no se cumplen supuestos)"
            elif all_normal and not homoscedastic:
                method = "t-test_welch"
                test_name = "t-test de Welch"
                reason = "Normalidad cumplida pero varianzas diferentes"
                alternative = "Mann-Whitney U"
            else:
                method = "mann_whitney"
                test_name = "Mann-Whitney U"
                reason = "Normalidad no cumplida (prueba no parametrica)"
                alternative = "t-test de Welch (si se transforman los datos)"
        else:
            if all_normal and homoscedastic:
                method = "anova"
                test_name = "ANOVA de una via"
                reason = "Normalidad y homogeneidad de varianzas cumplidas"
                alternative = "Kruskal-Wallis (si no se cumplen supuestos)"
            else:
                method = "kruskal_wallis"
                test_name = "Kruskal-Wallis"
                reason = "Normalidad o homogeneidad no cumplidas (prueba no parametrica)"
                alternative = "ANOVA de una via (si se transforman los datos)"
        
        return AnalysisPlan(
            method=method,
            test_name=test_name,
            reason=reason,
            assumptions_checked=all_assumptions,
            alternative=alternative,
            warnings=warnings,
        )
    
    def run_comparison(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        weights_col: Optional[str] = None,
        alpha: float = 0.05,
    ) -> Tuple[AnalysisPlan, TestResult]:
        """
        Ejecuta la comparacion apropiada entre grupos.
        """
        plan = self.select_comparison_method(df, value_col, group_col, weights_col, alpha)
        
        if plan.method == "none":
            return plan, None
        
        if plan.method in ("t-test_student", "t-test_welch"):
            result = self.inference.compare_two_groups(df, value_col, group_col, weights_col)
        elif plan.method == "mann_whitney":
            result = self.inference.compare_two_groups(df, value_col, group_col, weights_col)
        elif plan.method in ("anova", "kruskal_wallis"):
            result = self.inference.compare_multiple_groups(df, value_col, group_col, weights_col)
        else:
            result = None
        
        return plan, result
    
    def select_regression_method(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_cols: List[str],
        weights_col: Optional[str] = None,
    ) -> AnalysisPlan:
        """
        Selecciona el metodo de regresion apropiado.
        """
        warnings = []
        assumptions = []
        
        y = df[y_col].dropna()
        
        # Verificar si Y es categorica
        if y.dtype == "object" or (y.nunique() <= 5 and y.dtype != "float64"):
            if y.nunique() == 2:
                return AnalysisPlan(
                    method="logistic",
                    test_name="Regresion logistica binaria",
                    reason="Variable dependiente categorica binaria",
                    assumptions_checked=[],
                    alternative="Chi-cuadrado de independencia",
                    warnings=[],
                )
            else:
                return AnalysisPlan(
                    method="ordinal",
                    test_name="Regresion ordinal",
                    reason=f"Variable dependiente categorica ordinal ({y.nunique()} categorias)",
                    assumptions_checked=[],
                    alternative="Chi-cuadrado o Kruskal-Wallis",
                    warnings=[],
                )
        
        # Y es continua
        has_weights = weights_col is not None and weights_col in df.columns
        
        # Verificar normalidad de Y
        if len(y) <= 5000:
            _, p_normal = sp_stats.shapiro(y.values)
            is_normal = p_normal > 0.05
        else:
            is_normal = False
        
        assumptions.append(AssumptionCheck(
            name="Normalidad de Y",
            test_name="Shapiro-Wilk",
            statistic=0,
            p_value=round(p_normal, 4) if len(y) <= 5000 else 0,
            passed=is_normal,
            interpretation="Normal" if is_normal else "No normal",
        ))
        
        # Verificar colinealidad
        X = df[x_cols].dropna()
        if len(x_cols) > 1:
            corr_matrix = X.corr().abs()
            high_corr = []
            for i in range(len(x_cols)):
                for j in range(i + 1, len(x_cols)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr.append((x_cols[i], x_cols[j], corr_matrix.iloc[i, j]))
            
            if high_corr:
                warnings.append(f"Colinealidad alta detectada: {high_corr}")
        
        if has_weights:
            return AnalysisPlan(
                method="wls",
                test_name="Regresion WLS (Weighted Least Squares)",
                reason="Se detectaron pesos de diseno muestral",
                assumptions_checked=assumptions,
                alternative="OLS sin pesos (no recomendado para inferencia poblacional)",
                warnings=warnings,
            )
        else:
            return AnalysisPlan(
                method="ols",
                test_name="Regresion lineal OLS",
                reason="Variable dependiente continua, sin pesos",
                assumptions_checked=assumptions,
                alternative="WLS (si se agregan pesos de diseno)",
                warnings=warnings,
            )
    
    def run_regression(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_cols: List[str],
        weights_col: Optional[str] = None,
    ) -> Tuple[AnalysisPlan, RegressionResult]:
        """
        Ejecuta la regresion apropiada.
        """
        plan = self.select_regression_method(df, y_col, x_cols, weights_col)
        
        if plan.method == "wls":
            result = self.regression.wls(df, y_col, x_cols, weights_col)
        elif plan.method == "logistic":
            result = self.regression.logistic(df, y_col, x_cols)
        elif plan.method == "ols":
            result = self.regression.ols(df, y_col, x_cols)
        else:
            result = None
        
        return plan, result
    
    def descriptive_summary(
        self,
        df: pd.DataFrame,
        value_col: str,
        weights_col: Optional[str] = None,
        group_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Genera resumen descriptivo completo.
        """
        has_weights = weights_col is not None and weights_col in df.columns
        
        if group_col:
            groups = df[group_col].unique()
            summary = {}
            for g in groups:
                subset = df[df[group_col] == g]
                vals = subset[value_col].dropna()
                if has_weights:
                    w = subset.loc[vals.index, weights_col]
                    mean = self.desc.weighted_mean(vals, w)
                    std = self.desc.weighted_std(vals, w)
                    median = self.desc.weighted_median(vals, w)
                    se = self.desc.stratified_se(subset, value_col, weights_col, group_col) if group_col else 0
                else:
                    mean = float(vals.mean())
                    std = float(vals.std())
                    median = float(vals.median())
                    se = std / np.sqrt(len(vals)) if len(vals) > 0 else 0
                
                ci = self.desc.compute_ci(mean, se) if se > 0 else (mean, mean)
                summary[str(g)] = {
                    "n": len(vals),
                    "media": round(mean, 2),
                    "desv_std": round(std, 2),
                    "mediana": round(median, 2),
                    "min": round(float(vals.min()), 2),
                    "max": round(float(vals.max()), 2),
                    "se": round(se, 4),
                    "ci_95_lower": round(ci[0], 2),
                    "ci_95_upper": round(ci[1], 2),
                }
            return {"por_grupo": summary}
        else:
            vals = df[value_col].dropna()
            if has_weights:
                w = df.loc[vals.index, weights_col]
                mean = self.desc.weighted_mean(vals, w)
                std = self.desc.weighted_std(vals, w)
                median = self.desc.weighted_median(vals, w)
            else:
                mean = float(vals.mean())
                std = float(vals.std())
                median = float(vals.median())
            
            percentiles = self.desc.weighted_percentile(vals, w if has_weights else np.ones(len(vals)), [25, 50, 75]) if has_weights else [float(vals.quantile(p)) for p in [0.25, 0.5, 0.75]]
            
            return {
                "n": len(vals),
                "media": round(mean, 2),
                "desv_std": round(std, 2),
                "mediana": round(median, 2),
                "min": round(float(vals.min()), 2),
                "max": round(float(vals.max()), 2),
                "percentiles": {
                    "p25": round(percentiles[0], 2),
                    "p50": round(percentiles[1], 2),
                    "p75": round(percentiles[2], 2),
                },
            }
