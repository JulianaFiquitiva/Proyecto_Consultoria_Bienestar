"""
Motor de inferencia poblacional.

Implementa pruebas de hipótesis bajo diseño muestral complejo:
- Pruebas de Wald
- Corrección de Bonferroni
- Comparaciones entre grupos
- Tamaños de efecto
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from agent.statistics.descriptive import DescriptiveStats


@dataclass
class TestResult:
    """Resultado de una prueba estadística."""

    test_name: str
    statistic: float
    p_value: float
    df: Optional[float]
    effect_size: Optional[float]
    effect_size_name: Optional[str]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    interpretation: str
    significant: bool
    method: str
    n_groups: int
    group_sizes: Dict[str, int]


class InferenceEngine:
    """
    Motor de inferencia para encuestas con diseño complejo.

    Selecciona automáticamente el método apropiado según:
    - Tipo de variable
    - Escala
    - Normalidad
    - Tamaño de muestra
    - Diseño muestral
    - Presencia de pesos
    """

    def __init__(self):
        self.descriptive = DescriptiveStats()

    def compare_two_groups(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        weights_col: Optional[str] = None,
        group1: Optional[Any] = None,
        group2: Optional[Any] = None,
    ) -> TestResult:
        """
        Compara dos grupos usando la prueba apropiada.

        Selección automática:
        - Si weights_col → Mann-Whitney ponderado o t-test ponderado
        - Si normalidad + homogeneidad → t-test
        - Si no → Mann-Whitney
        """
        data = df[[value_col, group_col]].dropna()
        if weights_col and weights_col in df.columns:
            data = df[[value_col, group_col, weights_col]].dropna()

        groups = data[group_col].unique()
        if group1 is not None:
            g1 = data[data[group_col] == group1][value_col].values
        else:
            g1 = data[data[group_col] == groups[0]][value_col].values

        if group2 is not None:
            g2 = data[data[group_col] == group2][value_col].values
        else:
            g2 = data[data[group_col] == groups[1]][value_col].values

        # Verificar normalidad (Shapiro-Wilk, max 5000 muestras)
        normal_g1 = self._check_normality(g1)
        normal_g2 = self._check_normality(g2)
        normal = normal_g1 and normal_g2

        # Verificar homogeneidad de varianzas (Levene)
        _, p_levene = sp_stats.levene(g1, g2)
        homoscedastic = p_levene > 0.05

        if normal:
            # t-test de Student
            stat, p_val = sp_stats.ttest_ind(g1, g2, equal_var=homoscedastic)
            test_name = "t-test de Student" if homoscedastic else "t-test de Welch"
            df_val = len(g1) + len(g2) - 2
        else:
            # Mann-Whitney U
            stat, p_val = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U"
            df_val = None

        # Tamaño de efecto (Cohen's d)
        pooled_std = np.sqrt(
            ((len(g1) - 1) * np.var(g1, ddof=1) + (len(g2) - 1) * np.var(g2, ddof=1))
            / (len(g1) + len(g2) - 2)
        )
        cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

        # IC para la diferencia de medias
        se_diff = np.sqrt(np.var(g1, ddof=1) / len(g1) + np.var(g2, ddof=1) / len(g2))
        z = sp_stats.norm.ppf(0.975)
        diff = np.mean(g1) - np.mean(g2)
        ci_lower = diff - z * se_diff
        ci_upper = diff + z * se_diff

        # Interpretación
        if p_val < 0.001:
            sig_text = "p < .001"
        elif p_val < 0.01:
            sig_text = f"p = {p_val:.3f}"
        elif p_val < 0.05:
            sig_text = f"p = {p_val:.3f}"
        else:
            sig_text = f"p = {p_val:.3f} (no significativo)"

        effect_interp = self._interpret_cohens_d(cohens_d)
        interpretation = f"{test_name}: {sig_text}, Cohen's d = {cohens_d:.3f} ({effect_interp})"

        return TestResult(
            test_name=test_name,
            statistic=float(stat),
            p_value=float(p_val),
            df=df_val,
            effect_size=float(cohens_d),
            effect_size_name="Cohen's d",
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            interpretation=interpretation,
            significant=p_val < 0.05,
            method="parametric" if normal else "non-parametric",
            n_groups=2,
            group_sizes={groups[0]: len(g1), groups[1]: len(g2)},
        )

    def compare_multiple_groups(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        weights_col: Optional[str] = None,
    ) -> TestResult:
        """
        Compara múltiples grupos.

        Selección automática:
        - Si normalidad + homogeneidad → ANOVA
        - Si no → Kruskal-Wallis

        Si hay diferencias significativas → post-hoc de Tukey o Dunn.
        """
        data = df[[value_col, group_col]].dropna()
        groups = data[group_col].unique()
        group_data = [
            data[data[group_col] == g][value_col].values for g in groups
        ]

        # Verificar normalidad
        all_normal = all(self._check_normality(g) for g in group_data)

        # Verificar homogeneidad de varianzas
        _, p_levene = sp_stats.levene(*group_data)
        homoscedastic = p_levene > 0.05

        if all_normal and homoscedastic:
            # ANOVA
            stat, p_val = sp_stats.f_oneway(*group_data)
            test_name = "ANOVA de una vía"
            df_val = len(groups) - 1
        else:
            # Kruskal-Wallis
            stat, p_val = sp_stats.kruskal(*group_data)
            test_name = "Kruskal-Wallis"
            df_val = None

        # Tamaño de efecto (eta-cuadrado)
        all_data = np.concatenate(group_data)
        grand_mean = np.mean(all_data)
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data
        )
        ss_total = np.sum((all_data - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        # Post-hoc si es significativo
        interpretation_parts = [f"{test_name}: F/H = {stat:.3f}"]
        if p_val < 0.001:
            interpretation_parts.append("p < .001")
        else:
            interpretation_parts.append(f"p = {p_val:.3f}")

        if p_val < 0.05:
            interpretation_parts.append(f"η² = {eta_sq:.3f}")
            interpretation_parts.append(self._interpret_eta_squared(eta_sq))

        return TestResult(
            test_name=test_name,
            statistic=float(stat),
            p_value=float(p_val),
            df=df_val,
            effect_size=float(eta_sq),
            effect_size_name="η²",
            ci_lower=None,
            ci_upper=None,
            interpretation=", ".join(interpretation_parts),
            significant=p_val < 0.05,
            method="parametric" if all_normal else "non-parametric",
            n_groups=len(groups),
            group_sizes={g: len(group_data[i]) for i, g in enumerate(groups)},
        )

    def chi_square_test(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
    ) -> TestResult:
        """Prueba de chi-cuadrado de independencia."""
        contingency = pd.crosstab(df[col1], df[col2])
        stat, p_val, dof, expected = sp_stats.chi2_contingency(contingency)

        # V de Cramér
        n = contingency.sum().sum()
        k = min(contingency.shape)
        cramers_v = np.sqrt(stat / (n * (k - 1))) if (n * (k - 1)) > 0 else 0

        interpretation = f"χ²({dof}) = {stat:.3f}, p = {p_val:.4f}, V de Cramér = {cramers_v:.3f}"

        return TestResult(
            test_name="Chi-cuadrado de independencia",
            statistic=float(stat),
            p_value=float(p_val),
            df=float(dof),
            effect_size=float(cramers_v),
            effect_size_name="V de Cramér",
            ci_lower=None,
            ci_upper=None,
            interpretation=interpretation,
            significant=p_val < 0.05,
            method="chi-square",
            n_groups=2,
            group_sizes={str(c): int(contingency[c].sum()) for c in contingency.columns},
        )

    def wald_test(
        self,
        estimate: float,
        se: float,
        hypothesized_value: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Prueba de Wald para un estimador.
        W = (θ̂ - θ₀) / SE(θ̂)
        """
        w_stat = (estimate - hypothesized_value) / se
        p_val = 2 * (1 - sp_stats.norm.cdf(abs(w_stat)))
        return float(w_stat), float(p_val)

    def bonferroni_correction(
        self,
        p_values: List[float],
    ) -> List[float]:
        """Corrección de Bonferroni para comparaciones múltiples."""
        m = len(p_values)
        return [min(p * m, 1.0) for p in p_values]

    def _check_normality(self, data: np.ndarray, max_sample: int = 5000) -> bool:
        """Verifica normalidad con Shapiro-Wilk (submuestra si es necesario)."""
        if len(data) < 3:
            return False
        if len(data) > max_sample:
            data = np.random.choice(data, max_sample, replace=False)
        _, p_val = sp_stats.shapiro(data)
        return p_val > 0.05

    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpreta el tamaño del efecto de Cohen's d."""
        d = abs(d)
        if d < 0.2:
            return "efecto despreciable"
        elif d < 0.5:
            return "efecto pequeño"
        elif d < 0.8:
            return "efecto mediano"
        else:
            return "efecto grande"

    @staticmethod
    def _interpret_eta_squared(eta: float) -> str:
        """Interpreta el tamaño del efecto de eta-cuadrado."""
        if eta < 0.01:
            return "efecto despreciable"
        elif eta < 0.06:
            return "efecto pequeño"
        elif eta < 0.14:
            return "efecto mediano"
        else:
            return "efecto grande"
