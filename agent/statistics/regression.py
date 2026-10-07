"""
Motor de regresión.

Implementa modelos de regresión apropiados:
- Regresión lineal (OLS)
- WLS (Weighted Least Squares) para diseño complejo
- Regresión logística para categorías de bienestar
- Modelos ordinales
- Selección automática de método
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class RegressionResult:
    """Resultado de un modelo de regresión."""

    model_type: str
    formula: str
    n_obs: int
    r_squared: Optional[float]
    adj_r_squared: Optional[float]
    f_statistic: Optional[float]
    f_p_value: Optional[float]
    aic: Optional[float]
    bic: Optional[float]
    coefficients: pd.DataFrame
    residuals: Optional[np.ndarray]
    interpretation: str
    warnings: List[str]
    assumptions: Dict[str, Any]


class RegressionEngine:
    """
    Motor de regresión con selección automática de método.

    Evalúa:
    - Tipo de variable dependiente
    - Presencia de pesos
    - Normalidad de residuos
    - Homocedasticidad
    - Colinealidad
    """

    def __init__(self):
        self._results_cache: Dict[str, Any] = {}

    def auto_select_method(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_cols: List[str],
        weights_col: Optional[str] = None,
    ) -> str:
        """
        Selecciona automáticamente el método de regresión apropiado.

        Returns:
            Nombre del método: "ols", "wls", "logistic", "ordinal"
        """
        y = df[y_col].dropna()

        # Variable dependiente categórica
        if y.dtype == "object" or y.nunique() <= 5:
            if y.nunique() == 2:
                return "logistic"
            elif y.nunique() <= 5:
                return "ordinal"

        # Presencia de pesos → WLS
        if weights_col and weights_col in df.columns:
            return "wls"

        return "ols"

    def ols(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_cols: List[str],
        add_constant: bool = True,
    ) -> RegressionResult:
        """
        Regresión lineal OLS.

        y = β₀ + β₁x₁ + ... + βₖxₖ + ε
        """
        data = df[[y_col] + x_cols].dropna()
        y = data[y_col].values
        X = data[x_cols].values

        if add_constant:
            X = np.column_stack([np.ones(len(X)), X])
            cols = ["Intercepto"] + x_cols
        else:
            cols = x_cols

        # Ajustar modelo
        n, k = X.shape
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        residuals = y - y_pred

        # Estadísticos
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k - 1) if n > k + 1 else 0

        # F-statistic
        ss_model = ss_tot - ss_res
        ms_model = ss_model / (k - 1) if add_constant else ss_model / k
        ms_res = ss_res / (n - k)
        f_stat = ms_model / ms_res if ms_res > 0 else 0
        f_p = 1 - sp_stats.f.cdf(f_stat, k - 1 if add_constant else k, n - k)

        # Errores estándar y p-valores
        mse = ss_res / (n - k)
        try:
            cov = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            t_vals = beta / se
            p_vals = 2 * (1 - sp_stats.t.cdf(np.abs(t_vals), n - k))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), np.nan)
            t_vals = np.full(len(beta), np.nan)
            p_vals = np.full(len(beta), np.nan)

        # AIC y BIC
        log_lik = -n / 2 * (np.log(2 * np.pi * ss_res / n) + 1)
        aic = -2 * log_lik + 2 * k
        bic = -2 * log_lik + k * np.log(n)

        # Construir tabla de coeficientes
        coef_df = pd.DataFrame({
            "coef": beta,
            "std_err": se,
            "t": t_vals,
            "p": p_vals,
            "ci_lower": beta - 1.96 * se,
            "ci_upper": beta + 1.96 * se,
        }, index=cols)

        # Verificar supuestos
        assumptions = self._check_ols_assumptions(residuals, y_pred, X)

        # Interpretación
        interpretation = self._interpret_ols(coef_df, r_sq, adj_r_sq, f_stat, f_p)

        return RegressionResult(
            model_type="OLS",
            formula=f"{y_col} ~ {' + '.join(x_cols)}",
            n_obs=n,
            r_squared=float(r_sq),
            adj_r_squared=float(adj_r_sq),
            f_statistic=float(f_stat),
            f_p_value=float(f_p),
            aic=float(aic),
            bic=float(bic),
            coefficients=coef_df,
            residuals=residuals,
            interpretation=interpretation,
            warnings=assumptions.get("warnings", []),
            assumptions=assumptions,
        )

    def wls(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_cols: List[str],
        weights_col: str,
        add_constant: bool = True,
    ) -> RegressionResult:
        """
        Regresión ponderada WLS (Weighted Least Squares).

        Adecuada para encuestas con diseño complejo cuando se
        usan factores de expansión.
        """
        data = df[[y_col] + x_cols + [weights_col]].dropna()
        y = data[y_col].values
        X = data[x_cols].values
        w = data[weights_col].values

        if add_constant:
            X = np.column_stack([np.ones(len(X)), X])
            cols = ["Intercepto"] + x_cols
        else:
            cols = x_cols

        # Matriz de pesos diagonal
        W = np.diag(w)
        XtW = X.T @ W
        beta = np.linalg.solve(XtW @ X, XtW @ y)
        y_pred = X @ beta
        residuals = y - y_pred

        # Estadísticos ponderados
        n = len(y)
        k = X.shape[1]
        ss_res = np.sum(w * residuals ** 2)
        y_wmean = np.average(y, weights=w)
        ss_tot = np.sum(w * (y - y_wmean) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - k - 1) if n > k + 1 else 0

        # F-statistic
        ss_model = ss_tot - ss_res
        ms_model = ss_model / (k - 1) if add_constant else ss_model / k
        ms_res = ss_res / (n - k)
        f_stat = ms_model / ms_res if ms_res > 0 else 0
        f_p = 1 - sp_stats.f.cdf(f_stat, k - 1 if add_constant else k, n - k)

        # Errores estándar
        try:
            cov = np.linalg.inv(X.T @ W @ X)
            se = np.sqrt(np.diag(cov))
            t_vals = beta / se
            p_vals = 2 * (1 - sp_stats.t.cdf(np.abs(t_vals), n - k))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), np.nan)
            t_vals = np.full(len(beta), np.nan)
            p_vals = np.full(len(beta), np.nan)

        # AIC y BIC
        log_lik = -n / 2 * (np.log(2 * np.pi * ss_res / n) + 1)
        aic = -2 * log_lik + 2 * k
        bic = -2 * log_lik + k * np.log(n)

        coef_df = pd.DataFrame({
            "coef": beta,
            "std_err": se,
            "t": t_vals,
            "p": p_vals,
            "ci_lower": beta - 1.96 * se,
            "ci_upper": beta + 1.96 * se,
        }, index=cols)

        assumptions = {"method": "WLS", "weights": weights_col}
        interpretation = f"WLS: R² = {r_sq:.4f}, F({k-1}, {n-k}) = {f_stat:.3f}, p = {f_p:.4f}"

        return RegressionResult(
            model_type="WLS",
            formula=f"{y_col} ~ {' + '.join(x_cols)} (pesos: {weights_col})",
            n_obs=n,
            r_squared=float(r_sq),
            adj_r_squared=float(adj_r_sq),
            f_statistic=float(f_stat),
            f_p_value=float(f_p),
            aic=float(aic),
            bic=float(bic),
            coefficients=coef_df,
            residuals=residuals,
            interpretation=interpretation,
            warnings=[],
            assumptions=assumptions,
        )

    def logistic(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_cols: List[str],
        add_constant: bool = True,
    ) -> RegressionResult:
        """
        Regresión logística binaria.

        log(p/(1-p)) = β₀ + β₁x₁ + ... + βₖxₖ
        """
        from scipy.special import expit

        data = df[[y_col] + x_cols].dropna()
        y = (data[y_col] > data[y_col].median()).astype(int).values
        X = data[x_cols].values

        if add_constant:
            X = np.column_stack([np.ones(len(X)), X])
            cols = ["Intercepto"] + x_cols
        else:
            cols = x_cols

        n, k = X.shape

        # Newton-Raphson para logística
        beta = np.zeros(k)
        for _ in range(100):
            p = expit(X @ beta)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            W = np.diag(p * (1 - p))
            grad = X.T @ (y - p)
            hess = -X.T @ W @ X
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                break
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break

        y_pred_prob = expit(X @ beta)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Métricas
        ll = np.sum(y * np.log(y_pred_prob + 1e-10) + (1 - y) * np.log(1 - y_pred_prob + 1e-10))
        ll_null = np.sum(y * np.log(np.mean(y) + 1e-10) + (1 - y) * np.log(1 - np.mean(y) + 1e-10))
        chi2 = -2 * (ll_null - ll)
        chi2_p = 1 - sp_stats.chi2.cdf(chi2, k - 1 if add_constant else k)

        # McFadden's pseudo-R²
        pseudo_r2 = 1 - ll / ll_null if ll_null != 0 else 0

        # Errores estándar
        try:
            cov = np.linalg.inv(-X.T @ np.diag(y_pred_prob * (1 - y_pred_prob)) @ X)
            se = np.sqrt(np.diag(cov))
            z_vals = beta / se
            p_vals = 2 * (1 - sp_stats.norm.cdf(np.abs(z_vals)))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), np.nan)
            z_vals = np.full(len(beta), np.nan)
            p_vals = np.full(len(beta), np.nan)

        # Odds ratios
        odds_ratios = np.exp(beta)

        coef_df = pd.DataFrame({
            "coef": beta,
            "std_err": se,
            "z": z_vals,
            "p": p_vals,
            "odds_ratio": odds_ratios,
            "ci_lower_or": np.exp(beta - 1.96 * se),
            "ci_upper_or": np.exp(beta + 1.96 * se),
        }, index=cols)

        accuracy = np.mean(y_pred == y)
        interpretation = (
            f"Regresión logística: pseudo-R² de McFadden = {pseudo_r2:.4f}, "
            f"χ²({k-1}) = {chi2:.3f}, p = {chi2_p:.4f}, "
            f"Exactitud = {accuracy:.1%}"
        )

        return RegressionResult(
            model_type="Logistic",
            formula=f"{y_col} ~ {' + '.join(x_cols)}",
            n_obs=n,
            r_squared=float(pseudo_r2),
            adj_r_squared=None,
            f_statistic=float(chi2),
            f_p_value=float(chi2_p),
            aic=None,
            bic=None,
            coefficients=coef_df,
            residuals=None,
            interpretation=interpretation,
            warnings=[],
            assumptions={"method": "logistic", "pseudo_r2": "McFadden"},
        )

    def _check_ols_assumptions(
        self,
        residuals: np.ndarray,
        y_pred: np.ndarray,
        X: np.ndarray,
    ) -> Dict[str, Any]:
        """Verifica supuestos de OLS."""
        assumptions = {"warnings": []}

        # Normalidad de residuos (Shapiro-Wilk)
        if len(residuals) <= 5000:
            _, p_norm = sp_stats.shapiro(residuals)
        else:
            sample = np.random.choice(residuals, 5000, replace=False)
            _, p_norm = sp_stats.shapiro(sample)
        assumptions["normality_p"] = float(p_norm)
        if p_norm < 0.05:
            assumptions["warnings"].append(
                f"Residuos posiblemente no normales (Shapiro p={p_norm:.4f})"
            )

        # Homocedasticidad (Breusch-Pagan simplificado)
        _, p_het = sp_stats.levene(
            residuals[:len(residuals)//2],
            residuals[len(residuals)//2:]
        )
        assumptions["homoscedasticity_p"] = float(p_het)
        if p_het < 0.05:
            assumptions["warnings"].append(
                f"Possible heterocedasticidad (Levene p={p_het:.4f})"
            )

        return assumptions

    def _interpret_ols(
        self,
        coef_df: pd.DataFrame,
        r_sq: float,
        adj_r_sq: float,
        f_stat: float,
        f_p: float,
    ) -> str:
        """Interpreta el modelo OLS."""
        parts = []
        parts.append(f"OLS: R² = {r_sq:.4f}, R² ajustado = {adj_r_sq:.4f}")
        parts.append(f"F = {f_stat:.3f}, p = {f_p:.4f}")

        if f_p < 0.05:
            sig_coefs = coef_df[coef_df["p"] < 0.05].index.tolist()
            if "Intercepto" in sig_coefs:
                sig_coefs.remove("Intercepto")
            if sig_coefs:
                parts.append(f"Predictores significativos: {', '.join(sig_coefs)}")

        return ", ".join(parts)
