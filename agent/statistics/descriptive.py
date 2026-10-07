"""
Estadísticos descriptivos ponderados.

Implementa funciones de estadística descriptiva que respetan
el diseño muestral estratificado con factores de expansión.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


class DescriptiveStats:
    """
    Estadísticos descriptivos para encuestas con diseño complejo.

    Todos los métodos aceptan un parámetro `weights` para
    cálculos ponderados.
    """

    @staticmethod
    def weighted_mean(
        x: Union[pd.Series, np.ndarray],
        weights: Union[pd.Series, np.ndarray],
    ) -> float:
        """Media ponderada: x̄_w = Σ(w_i × x_i) / Σ(w_i)."""
        mask = ~(np.isnan(x) | np.isnan(weights))
        return float(np.average(x[mask], weights=weights[mask]))

    @staticmethod
    def weighted_variance(
        x: Union[pd.Series, np.ndarray],
        weights: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Varianza ponderada (análoga a ddof=1).
        S²_w = Σ w_i(x_i - x̄_w)² / (Σ w_i - 1)
        """
        mask = ~(np.isnan(x) | np.isnan(weights))
        xm, wm = np.asarray(x[mask]), np.asarray(weights[mask])
        mu = np.average(xm, weights=wm)
        return float(np.sum(wm * (xm - mu) ** 2) / (np.sum(wm) - 1))

    @staticmethod
    def weighted_std(
        x: Union[pd.Series, np.ndarray],
        weights: Union[pd.Series, np.ndarray],
    ) -> float:
        """Desviación estándar ponderada."""
        return float(np.sqrt(DescriptiveStats.weighted_variance(x, weights)))

    @staticmethod
    def weighted_median(
        x: Union[pd.Series, np.ndarray],
        weights: Union[pd.Series, np.ndarray],
    ) -> float:
        """Mediana ponderada."""
        mask = ~(np.isnan(x) | np.isnan(weights))
        xm, wm = np.asarray(x[mask]), np.asarray(weights[mask])
        idx = np.argsort(xm)
        xm, wm = xm[idx], wm[idx]
        cum = np.cumsum(wm)
        mid = cum[-1] / 2
        return float(xm[np.searchsorted(cum, mid)])

    @staticmethod
    def weighted_percentile(
        x: Union[pd.Series, np.ndarray],
        weights: Union[pd.Series, np.ndarray],
        percentiles: List[float],
    ) -> List[float]:
        """Percentiles ponderados."""
        mask = ~(np.isnan(x) | np.isnan(weights))
        xm, wm = np.asarray(x[mask]), np.asarray(weights[mask])
        idx = np.argsort(xm)
        xm, wm = xm[idx], wm[idx]
        cum = np.cumsum(wm)
        cum_norm = cum / cum[-1]
        result = []
        for p in percentiles:
            target = p / 100.0
            i = np.searchsorted(cum_norm, target)
            i = min(i, len(xm) - 1)
            result.append(float(xm[i]))
        return result

    @staticmethod
    def weighted_cov(
        x: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        weights: Union[pd.Series, np.ndarray],
    ) -> float:
        """Covarianza ponderada."""
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(weights))
        xm, ym, wm = np.asarray(x[mask]), np.asarray(y[mask]), np.asarray(weights[mask])
        mx = np.average(xm, weights=wm)
        my = np.average(ym, weights=wm)
        return float(np.sum(wm * (xm - mx) * (ym - my)) / (np.sum(wm) - 1))

    @staticmethod
    def weighted_correlation(
        x: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        weights: Union[pd.Series, np.ndarray],
    ) -> float:
        """Correlación de Pearson ponderada."""
        cxy = DescriptiveStats.weighted_cov(x, y, weights)
        sx = DescriptiveStats.weighted_std(x, weights)
        sy = DescriptiveStats.weighted_std(y, weights)
        return float(cxy / (sx * sy)) if sx > 0 and sy > 0 else 0.0

    @staticmethod
    def weighted_cronbach_alpha(
        data: pd.DataFrame,
        weights: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Alpha de Cronbach ponderado.
        α_w = (k/(k-1)) × (1 − Σ S²_w(Xj) / S²_w(X_total))
        """
        k = data.shape[1]
        total = data.sum(axis=1)
        var_total = DescriptiveStats.weighted_variance(total, weights)
        var_items = sum(
            DescriptiveStats.weighted_variance(data.iloc[:, j], weights)
            for j in range(k)
        )
        return float((k / (k - 1)) * (1 - var_items / var_total))

    @staticmethod
    def weighted_frequency(
        x: pd.Series,
        weights: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Frecuencias absolutas y relativas ponderadas."""
        if weights is None:
            weights = pd.Series(1, index=x.index)

        freq = pd.DataFrame({"value": x, "weight": weights})
        agg = freq.groupby("value")["weight"].agg(["sum", "count"]).reset_index()
        agg.columns = ["value", "freq_abs", "freq_rel"]
        total = agg["freq_abs"].sum()
        agg["freq_rel_pct"] = agg["freq_abs"] / total * 100
        return agg

    @staticmethod
    def group_comparison(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        weights_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compara estadísticos descriptivos por grupo.

        Returns:
            DataFrame con media, DE, n, etc. por grupo.
        """
        results = []
        for group_name, group_df in df.groupby(group_col):
            y = group_df[value_col].dropna()
            if weights_col and weights_col in group_df.columns:
                w = group_df.loc[y.index, weights_col]
                media = DescriptiveStats.weighted_mean(y, w)
                de = DescriptiveStats.weighted_std(y, w)
            else:
                media = y.mean()
                de = y.std()

            results.append({
                "grupo": group_name,
                "n": len(y),
                "media": round(media, 2),
                "de": round(de, 2),
                "min": round(y.min(), 2),
                "max": round(y.max(), 2),
            })

        return pd.DataFrame(results)

    @staticmethod
    def compute_ci(
        mean: float,
        se: float,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calcula intervalo de confianza."""
        z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
        return (float(mean - z * se), float(mean + z * se))

    @staticmethod
    def stratified_se(
        data: pd.DataFrame,
        y_col: str,
        w_col: str,
        strata_col: str,
    ) -> float:
        """
        Error estándar de la media ponderada bajo muestreo estratificado.
        V(ȳ_st) = Σ_h (N_h/N)² × (1 − f_h) × S²_h / r_h
        """
        N_total = data[w_col].sum()
        var_sum = 0.0

        for _, stratum in data.groupby(strata_col):
            y_h = stratum[y_col].values
            r_h = len(stratum)
            N_h = stratum[w_col].sum()

            if r_h < 2:
                continue

            S2_h = np.var(y_h, ddof=1)
            fpc_h = max(0, 1 - r_h / N_h)

            var_sum += (N_h / N_total) ** 2 * fpc_h * S2_h / r_h

        return float(np.sqrt(var_sum))

    @staticmethod
    def weighted_proportion(
        data: pd.DataFrame,
        cat_col: str,
        cat_val: Any,
        w_col: str,
    ) -> float:
        """Proporción ponderada de una categoría."""
        w = data[w_col]
        mask = data[cat_col] == cat_val
        return float(w[mask].sum() / w.sum())
