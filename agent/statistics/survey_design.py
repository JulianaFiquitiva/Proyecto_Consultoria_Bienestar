"""
Inferencia basada en el diseño muestral (entrega 2, pregunta 2.1).

El estudio es un muestreo estratificado: 8 estratos (seccional × pregrado/
posgrado) con tamaño poblacional N_h conocido, y factores de expansión
calibrados. Este módulo estima medias, proporciones y diferencias entre
dominios con su error estándar de DISEÑO, por linealización de Taylor:

    ȳ_d = Σ w_i d_i y_i / Σ w_i d_i
    z_i = w_i d_i (y_i − ȳ_d) / N̂_d                (variable linealizada)
    V(ȳ_d) = Σ_h (1 − f_h) · n_h/(n_h − 1) · Σ_i (z_hi − z̄_h)²
    f_h = n_h / N_h

El deff compara esa varianza con la de un muestreo aleatorio simple del
mismo tamaño: deff = V_diseño / V_MAS, con V_MAS = (1 − n/N̂) · S²_w / n.

No usa scipy (los cuantiles normales vienen de statistics.NormalDist), así
que funciona aunque Smart App Control bloquee scipy.

Limitación declarada: los pesos calibrados se tratan como fijos. Ignorar la
calibración suele sobrestimar un poco la varianza (resultado conservador).
"""
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

WEIGHT_COL = "factor_expansion"
STRATA_COLS = ("seccional", "modalidad_exp")
POP_COL = "N_poblacion_estrato"


@dataclass
class DesignEstimate:
    """Estimación con su error estándar de diseño."""

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    deff: Optional[float]
    n: int                      # casos de la muestra en el dominio
    population: float           # N̂ del dominio (suma de pesos)
    domain: Optional[str] = None

    @property
    def n_eff(self) -> Optional[float]:
        """Tamaño efectivo de muestra: n / deff."""
        if not self.deff:
            return None
        return self.n / self.deff

    def as_dict(self, decimals: int = 4) -> Dict[str, Any]:
        r = lambda v: None if v is None else round(float(v), decimals)
        return {
            "estimacion": r(self.estimate),
            "error_estandar": r(self.se),
            "ic_inferior": r(self.ci_lower),
            "ic_superior": r(self.ci_upper),
            "deff": r(self.deff),
            "n": self.n,
            "n_efectivo": r(self.n_eff),
            "N_estimado": r(self.population),
        }


@dataclass
class DesignDifference:
    """Diferencia entre dos dominios con prueba de Wald de diseño."""

    group1: str
    group2: str
    estimate1: DesignEstimate
    estimate2: DesignEstimate
    difference: float           # grupo1 − grupo2
    se: float
    ci_lower: float
    ci_upper: float
    z: float
    p_value: float
    p_adjusted: Optional[float] = None   # Bonferroni, si hay varias comparaciones

    @property
    def significant(self) -> bool:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return p < 0.05

    def as_dict(self, decimals: int = 4) -> Dict[str, Any]:
        r = lambda v: None if v is None else round(float(v), decimals)
        return {
            "grupo_1": self.group1,
            "grupo_2": self.group2,
            "media_1": r(self.estimate1.estimate),
            "media_2": r(self.estimate2.estimate),
            "diferencia": r(self.difference),
            "error_estandar": r(self.se),
            "ic_inferior": r(self.ci_lower),
            "ic_superior": r(self.ci_upper),
            "z": r(self.z),
            "p_valor": r(self.p_value),
            "p_ajustado_bonferroni": r(self.p_adjusted),
            "significativo": self.significant,
        }


@dataclass
class DesignRegression:
    """Regresión ponderada con errores estándar de diseño."""

    coefficients: pd.DataFrame   # coef, se, ic_inferior, ic_superior, z, p_valor
    n: int
    r_squared: float             # R² ponderado
    y_col: str

    def coef(self, name: str) -> float:
        return float(self.coefficients.loc[name, "coef"])

    def as_dict(self, decimals: int = 4) -> Dict[str, Any]:
        return {
            "n": self.n,
            "r_cuadrado": round(self.r_squared, decimals),
            "coeficientes": {
                var: {k: round(float(v), decimals) for k, v in fila.items()}
                for var, fila in self.coefficients.iterrows()
            },
        }


class SurveyDesign:
    """
    Diseño estratificado con factores de expansión.

    Args:
        df: datos de la muestra (una fila por estudiante).
        weight_col: factores de expansión.
        strata_cols: columnas cuya combinación define el estrato.
        pop_col: tamaño poblacional del estrato (N_h), para la corrección
            por población finita. Si falta, se usa la suma de pesos.
        confidence: nivel de confianza de los intervalos.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        weight_col: str = WEIGHT_COL,
        strata_cols: Sequence[str] = STRATA_COLS,
        pop_col: Optional[str] = POP_COL,
        confidence: float = 0.95,
    ):
        faltan = [c for c in (weight_col, *strata_cols) if c not in df.columns]
        if faltan:
            raise ValueError(f"Faltan columnas del diseño: {faltan}")
        self.df = df
        self.weight_col = weight_col
        self.strata_cols = list(strata_cols)
        self.confidence = confidence
        self.z_crit = NormalDist().inv_cdf(1 - (1 - confidence) / 2)

        self._w = df[weight_col].to_numpy(dtype=float)
        self._strata = (
            df[self.strata_cols].astype(str).agg(" | ".join, axis=1).to_numpy()
        )
        # N_h por estrato: columna del diseño o, si falta, suma de pesos
        if pop_col and pop_col in df.columns:
            n_pop = df.groupby(self._strata)[pop_col].first()
        else:
            n_pop = pd.Series(self._w).groupby(self._strata).sum()
        self._N_h = n_pop.to_dict()

    # ── estimadores ────────────────────────────────────────────────────

    def mean(
        self,
        y_col: str,
        domain_col: Optional[str] = None,
        domain_value: Any = None,
    ) -> DesignEstimate:
        """Media ponderada (de toda la población o de un dominio)."""
        y = self.df[y_col].to_numpy(dtype=float)
        return self._mean_of(y, self._domain_mask(domain_col, domain_value),
                             None if domain_col is None else str(domain_value))

    def proportion(
        self,
        y_col: str,
        condition: Callable[[pd.Series], pd.Series],
        domain_col: Optional[str] = None,
        domain_value: Any = None,
    ) -> DesignEstimate:
        """
        Proporción ponderada de casos que cumplen `condition`.

        Ejemplo: diseño.proportion("T_Global", lambda t: t < 40)
        Los valores faltantes de `y_col` quedan fuera (no cuentan como 0).
        """
        serie = self.df[y_col]
        y = np.array(condition(serie), dtype=float)
        y[serie.isna().to_numpy()] = np.nan
        return self._mean_of(y, self._domain_mask(domain_col, domain_value),
                             None if domain_col is None else str(domain_value))

    def domain_means(
        self,
        y_col: str,
        group_col: str,
        min_n: int = 0,
    ) -> Dict[str, DesignEstimate]:
        """Media por cada categoría de `group_col` (solo las de n ≥ min_n)."""
        resultado = {}
        for g in self.df[group_col].dropna().unique():
            est = self.mean(y_col, group_col, g)
            if est.n >= min_n:
                resultado[str(g)] = est
        return resultado

    def difference(
        self,
        y_col: str,
        group_col: str,
        group1: Any,
        group2: Any,
    ) -> DesignDifference:
        """
        Diferencia de medias entre dos dominios (grupo1 − grupo2).

        La varianza se calcula sobre la diferencia de las variables
        linealizadas, así que incluye la covarianza entre los dominios
        dentro de cada estrato.
        """
        y = self.df[y_col].to_numpy(dtype=float)
        m1 = self._domain_mask(group_col, group1)
        m2 = self._domain_mask(group_col, group2)
        e1 = self._mean_of(y, m1, str(group1))
        e2 = self._mean_of(y, m2, str(group2))

        z1 = self._linearized(y, m1, e1.estimate, e1.population)
        z2 = self._linearized(y, m2, e2.estimate, e2.population)
        se = float(np.sqrt(self._stratified_variance(z1 - z2)))

        diff = e1.estimate - e2.estimate
        z = diff / se if se > 0 else float("nan")
        p = 2 * (1 - NormalDist().cdf(abs(z))) if se > 0 else float("nan")
        return DesignDifference(
            group1=str(group1), group2=str(group2),
            estimate1=e1, estimate2=e2,
            difference=float(diff), se=se,
            ci_lower=float(diff - self.z_crit * se),
            ci_upper=float(diff + self.z_crit * se),
            z=float(z), p_value=float(p),
        )

    def pairwise(
        self,
        y_col: str,
        group_col: str,
        min_n: int = 30,
    ) -> List[DesignDifference]:
        """
        Todas las comparaciones por pares entre categorías con n ≥ min_n,
        con p-valores ajustados por Bonferroni.
        """
        # valores ORIGINALES (no su texto): un estrato numerico 3.0 no es "3.0"
        conteo = self.df.loc[self.df[y_col].notna(), group_col].value_counts()
        grupos = sorted((g for g, n in conteo.items() if n >= min_n), key=str)
        pares = [
            self.difference(y_col, group_col, a, b)
            for i, a in enumerate(grupos) for b in grupos[i + 1:]
        ]
        m = len(pares)
        for par in pares:
            par.p_adjusted = min(par.p_value * m, 1.0)
        return pares

    def regression(self, y_col: str, X: pd.DataFrame) -> DesignRegression:
        """
        Mínimos cuadrados ponderados con varianza de diseño (Binder, 1983):

            β̂ = (X'WX)⁻¹ X'Wy
            V(β̂) = (X'WX)⁻¹ · Σ_h (1 − f_h) n_h/(n_h − 1) Σ_i (u_hi − ū_h)(u_hi − ū_h)' · (X'WX)⁻¹
            u_i = w_i · x_i · e_i

        A diferencia de statsmodels WLS (que supone errores homocedásticos e
        independientes), este error estándar refleja los estratos y los pesos.

        Args:
            y_col: variable dependiente (columna de self.df).
            X: predictores con el mismo índice que self.df, SIN constante
               (se añade). Las filas con faltantes en y o X se excluyen.
        """
        y_all = self.df[y_col].to_numpy(dtype=float)
        X_all = X.to_numpy(dtype=float)
        validos = (~np.isnan(y_all) & ~np.isnan(self._w)
                   & ~np.isnan(X_all).any(axis=1))
        Xv = np.column_stack([np.ones(validos.sum()), X_all[validos]])
        yv, wv = y_all[validos], self._w[validos]
        nombres = ["Intercepto"] + list(X.columns)

        XtWX = Xv.T @ (Xv * wv[:, None])
        beta = np.linalg.solve(XtWX, Xv.T @ (wv * yv))
        e = yv - Xv @ beta

        # Contribuciones linealizadas; fuera de la submuestra valen 0
        u = np.zeros((len(y_all), Xv.shape[1]))
        u[validos] = Xv * (wv * e)[:, None]
        meat = np.zeros((Xv.shape[1], Xv.shape[1]))
        for h in np.unique(self._strata):
            uh = u[self._strata == h]
            n_h = len(uh)
            if n_h < 2:
                continue
            f_h = min(n_h / self._N_h[h], 1.0) if self._N_h.get(h) else 0.0
            dh = uh - uh.mean(axis=0)
            meat += (1 - f_h) * n_h / (n_h - 1) * dh.T @ dh
        bread = np.linalg.inv(XtWX)
        se = np.sqrt(np.diag(bread @ meat @ bread))

        z = beta / se
        p = np.array([2 * (1 - NormalDist().cdf(abs(v))) for v in z])
        tabla = pd.DataFrame({
            "coef": beta, "se": se,
            "ic_inferior": beta - self.z_crit * se,
            "ic_superior": beta + self.z_crit * se,
            "z": z, "p_valor": p,
        }, index=nombres)

        media = np.average(yv, weights=wv)
        r2 = 1 - np.sum(wv * e ** 2) / np.sum(wv * (yv - media) ** 2)
        return DesignRegression(coefficients=tabla, n=int(validos.sum()),
                                r_squared=float(r2), y_col=y_col)

    def kish_deff(self) -> float:
        """Efecto de diseño por variabilidad de pesos: n·Σw² / (Σw)²."""
        w = self._w[~np.isnan(self._w)]
        return float(len(w) * np.sum(w ** 2) / np.sum(w) ** 2)

    # ── internos ───────────────────────────────────────────────────────

    def _domain_mask(self, domain_col: Optional[str], value: Any) -> np.ndarray:
        if domain_col is None:
            return np.ones(len(self.df), dtype=bool)
        return (self.df[domain_col] == value).to_numpy()

    def _mean_of(self, y: np.ndarray, mask: np.ndarray,
                 domain: Optional[str]) -> DesignEstimate:
        validos = mask & ~np.isnan(y) & ~np.isnan(self._w)
        n = int(validos.sum())
        if n == 0:
            raise ValueError(f"El dominio {domain!r} no tiene casos válidos")
        w, yv = self._w[validos], y[validos]
        N_d = float(w.sum())
        media = float(np.sum(w * yv) / N_d)

        z = self._linearized(y, mask, media, N_d)
        var = self._stratified_variance(z)
        se = float(np.sqrt(var))

        # deff: varianza de diseño frente a la de un MAS del mismo tamaño
        deff = None
        if n > 1:
            s2 = float(np.sum(w * (yv - media) ** 2) / (N_d - N_d / n))
            var_mas = (1 - n / N_d) * s2 / n if N_d > n else s2 / n
            deff = float(var / var_mas) if var_mas > 0 else None

        return DesignEstimate(
            estimate=media, se=se,
            ci_lower=media - self.z_crit * se,
            ci_upper=media + self.z_crit * se,
            deff=deff, n=n, population=N_d, domain=domain,
        )

    def _linearized(self, y: np.ndarray, mask: np.ndarray,
                    media: float, N_d: float) -> np.ndarray:
        """z_i = w_i d_i (y_i − ȳ_d) / N̂_d; vale 0 fuera del dominio."""
        validos = mask & ~np.isnan(y) & ~np.isnan(self._w)
        z = np.zeros(len(y))
        z[validos] = self._w[validos] * (y[validos] - media) / N_d
        return z

    def _stratified_variance(self, z: np.ndarray) -> float:
        """Σ_h (1 − f_h) · n_h/(n_h − 1) · Σ_i (z_hi − z̄_h)²."""
        total = 0.0
        for h in np.unique(self._strata):
            zh = z[self._strata == h]
            n_h = len(zh)
            if n_h < 2:
                continue
            f_h = min(n_h / self._N_h[h], 1.0) if self._N_h.get(h) else 0.0
            total += (1 - f_h) * n_h / (n_h - 1) * np.sum((zh - zh.mean()) ** 2)
        return float(total)
