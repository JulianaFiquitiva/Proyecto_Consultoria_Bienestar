"""
Brecha ajustada (entrega 2, pregunta 2.3).

¿La brecha de bienestar posgrado − pregrado (≈ 4,7 puntos T) es propia del
nivel de formación o se explica por otras variables, sobre todo la edad
(los estudiantes de posgrado son mayores)?

Se ajustan modelos anidados con regresión ponderada y varianza de diseño
(SurveyDesign.regression), todos sobre la MISMA submuestra de casos
completos para que las brechas sean comparables:

    M0  T ~ posgrado                                    (brecha bruta)
    M1  T ~ posgrado + edad
    M2  T ~ posgrado + edad + género + estrato + seccional
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.statistics.survey_design import SurveyDesign

EDAD_COL = "Edad"
GENERO_COL = "Género"
ESTRATO_COL = "Estrato socioeconómico"
SECCIONAL_COL = "seccional"
NIVEL_COL = "modalidad_exp"


def _predictores(df: pd.DataFrame, focal: str, edad_media: float) -> pd.DataFrame:
    """Matriz de predictores (sin constante) con nombres legibles."""
    X = pd.DataFrame(index=df.index)
    X[focal] = (df[NIVEL_COL] == focal).astype(float)
    X["Edad (años, centrada)"] = df[EDAD_COL] - edad_media
    genero = df[GENERO_COL].astype(str)
    X["Femenino (vs. masculino)"] = (genero == "Femenino").astype(float)
    X["Otro género (vs. masculino)"] = (~genero.isin(["Femenino", "Masculino"])).astype(float)
    X["Estrato socioeconómico"] = pd.to_numeric(df[ESTRATO_COL], errors="coerce")
    referencia = sorted(df[SECCIONAL_COL].dropna().unique())[0]
    for sec in sorted(df[SECCIONAL_COL].dropna().unique())[1:]:
        X[f"{sec} (vs. {referencia})"] = (df[SECCIONAL_COL] == sec).astype(float)
    return X


MODELOS = {
    "M0: brecha bruta": [],
    "M1: + edad": ["Edad (años, centrada)"],
    "M2: + edad, género, estrato y seccional": None,   # None = todos
}


def adjusted_gap(
    df: pd.DataFrame,
    y_col: str = "T_Global",
    focal: str = "Posgrado",
    reference: str = "Pregrado",
    design_kwargs: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Brecha focal − referencia, bruta y ajustada, con IC de diseño.

    Returns:
        dict con la tabla de modelos, el porcentaje de la brecha que
        explican los controles, la edad media por grupo y el modelo completo.
    """
    sub = df[df[NIVEL_COL].isin([focal, reference])]
    edad_media = float(np.average(sub[EDAD_COL].dropna(),
                                  weights=sub.loc[sub[EDAD_COL].notna(), "factor_expansion"]))
    X_full = _predictores(df, focal, edad_media)

    # Misma submuestra para todos los modelos: casos completos en TODO
    completos = df[y_col].notna() & X_full.notna().all(axis=1) & df[NIVEL_COL].isin([focal, reference])
    # Un control sin variacion en la submuestra (p. ej. ningun caso de
    # "otro genero") haria singular la matriz: se descarta y se informa.
    sin_variacion = [c for c in X_full.columns
                     if c != focal and X_full.loc[completos, c].nunique() < 2]
    X_full = X_full.drop(columns=sin_variacion)
    design = SurveyDesign(df.assign(**{y_col: df[y_col].where(completos)}),
                          **(design_kwargs or {}))

    filas: List[Dict[str, Any]] = []
    ajustes = {}
    for nombre, extra in MODELOS.items():
        cols = list(X_full.columns) if extra is None else [focal] + extra
        reg = design.regression(y_col, X_full[cols])
        ajustes[nombre] = reg
        c = reg.coefficients.loc[focal]
        filas.append({
            "modelo": nombre,
            "brecha": round(float(c["coef"]), 3),
            "ic_inferior": round(float(c["ic_inferior"]), 3),
            "ic_superior": round(float(c["ic_superior"]), 3),
            "p_valor": round(float(c["p_valor"]), 4),
            "r_cuadrado": round(reg.r_squared, 4),
            "n": reg.n,
        })

    bruta = filas[0]["brecha"]
    ajustada = filas[-1]["brecha"]
    explicada = (bruta - ajustada) / bruta * 100 if bruta else None

    edad = {}
    for g in (focal, reference):
        m = completos & (df[NIVEL_COL] == g)
        edad[g] = round(float(np.average(df.loc[m, EDAD_COL],
                                         weights=df.loc[m, "factor_expansion"])), 1)

    completo = ajustes[list(MODELOS)[-1]]
    return {
        "pregunta": (f"¿La brecha {focal} − {reference} se mantiene al "
                     f"controlar edad, género, estrato y seccional?"),
        "modelos": filas,
        "brecha_bruta": bruta,
        "brecha_ajustada": ajustada,
        "porcentaje_explicado_por_controles": (
            round(explicada, 1) if explicada is not None else None),
        "edad_media_ponderada": edad,
        "modelo_completo": completo.as_dict(4),
        "n_excluidos_por_faltantes": int(
            (df[NIVEL_COL].isin([focal, reference]) & ~completos).sum()),
        "controles_descartados_sin_variacion": sin_variacion,
        "metodo": ("Regresión ponderada (factores de expansión) con errores "
                   "estándar de diseño: linealización de Binder en 8 estratos"),
    }
