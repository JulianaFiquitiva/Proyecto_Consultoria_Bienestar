"""
Modelo multinivel (entrega 2, pregunta 2.4).

¿Cuánto del bienestar depende del programa académico (o del núcleo básico
de conocimiento) en que estudia cada estudiante?

Modelo de intercepto aleatorio (statsmodels MixedLM, REML):

    T_ij = β₀ + [covariables] + u_j + e_ij,   u_j ~ N(0, τ²),  e_ij ~ N(0, σ²)
    ICC = τ² / (τ² + σ²)

- ICC nulo (sin covariables): parte de la varianza total que está entre
  programas.
- ICC condicional (con nivel y edad): la parte que queda después de
  descontar que unos programas tienen estudiantes mayores o de posgrado.
- Prueba de τ² = 0: razón de verosimilitudes (ML) frente al modelo sin
  efecto aleatorio. Como τ² está en el borde del espacio de parámetros, el
  p-valor es la mitad del de una χ² con 1 gl (Self y Liang, 1987).

Limitación declarada: MixedLM no admite factores de expansión, así que
estas estimaciones son muestrales (no poblacionales).
"""
import warnings
from statistics import NormalDist
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _p_borde(lr: float) -> float:
    """p = 0.5 · P(χ²₁ > LR), sin scipy."""
    if lr <= 0:
        return 1.0
    return float(0.5 * 2 * (1 - NormalDist().cdf(np.sqrt(lr))))


def icc(
    df: pd.DataFrame,
    y_col: str,
    group_col: str,
    covariables: Optional[List[str]] = None,
    min_n_ranking: int = 30,
) -> Dict[str, Any]:
    """
    ICC del agrupamiento `group_col` con un modelo de intercepto aleatorio.

    Args:
        covariables: columnas numéricas o 0/1 a controlar (ICC condicional).
        min_n_ranking: los grupos con menos casos no aparecen en el ranking
            (su estimación es muy incierta y podría identificar personas).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    covariables = covariables or []
    datos = df[[y_col, group_col] + covariables].dropna().copy()
    datos.columns = ["y", "g"] + [f"x{i}" for i in range(len(covariables))]
    fijos = " + ".join(["1"] + [f"x{i}" for i in range(len(covariables))])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reml = smf.mixedlm(f"y ~ {fijos}", datos, groups=datos["g"]).fit(reml=True)
        ml = smf.mixedlm(f"y ~ {fijos}", datos, groups=datos["g"]).fit(reml=False)
        ols = smf.ols(f"y ~ {fijos}", datos).fit()

    tau2 = float(reml.cov_re.iloc[0, 0])
    sigma2 = float(reml.scale)
    lr = float(max(2 * (ml.llf - ols.llf), 0.0))

    tamanos = datos["g"].value_counts()
    efectos = pd.Series({g: float(v.iloc[0]) for g, v in reml.random_effects.items()})
    ranking = (
        pd.DataFrame({"efecto": efectos, "n": tamanos})
        .query("n >= @min_n_ranking")
        .sort_values("efecto", ascending=False)
    )
    fila = lambda g, r: {"grupo": str(g), "n": int(r["n"]),
                         "efecto_estimado": round(float(r["efecto"]), 2)}

    return {
        "agrupamiento": group_col,
        "covariables": covariables,
        "n": int(len(datos)),
        "n_grupos": int(datos["g"].nunique()),
        "tamano_mediano_grupo": float(tamanos.median()),
        "grupos_con_menos_de_10": int((tamanos < 10).sum()),
        "tau2_entre_grupos": round(tau2, 3),
        "sigma2_dentro": round(sigma2, 3),
        "icc": round(tau2 / (tau2 + sigma2), 4),
        "lr_tau2_cero": round(lr, 2),
        "p_valor_tau2_cero": round(_p_borde(lr), 4),
        "ranking_n_minimo": min_n_ranking,
        "grupos_mas_altos": [fila(g, r) for g, r in ranking.head(5).iterrows()],
        "grupos_mas_bajos": [fila(g, r) for g, r in ranking.tail(5)[::-1].iterrows()],
    }


def icc_programa(df: pd.DataFrame, y_col: str = "T_Global") -> Dict[str, Any]:
    """ICC nulo y condicional por programa y por NBC."""
    datos = df.copy()
    datos["_posgrado"] = (datos["modalidad_exp"] == "Posgrado").astype(float)
    controles = ["_posgrado", "Edad"]
    resultado = {}
    for nombre, col in (("programa", "Programa de formación"),
                        ("nbc", "Núcleo básico de conocimiento")):
        if col not in datos.columns:
            continue
        nulo = icc(datos, y_col, col)
        condicional = icc(datos, y_col, col, covariables=controles)
        resultado[nombre] = {
            "nulo": nulo,
            "condicional_nivel_edad": condicional,
        }
    resultado["metodo"] = ("Modelo multinivel de intercepto aleatorio (MixedLM, "
                           "REML); prueba de tau2 = 0 por razon de verosimilitudes "
                           "con mezcla de chi2 (Self y Liang, 1987)")
    resultado["limitacion"] = ("MixedLM no admite factores de expansion: el ICC es "
                               "muestral, no poblacional.")
    return resultado
