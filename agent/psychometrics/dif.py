"""
Funcionamiento diferencial del ítem — DIF (entrega 2, pregunta 2.2).

¿La escala de Ryff mide igual a mujeres y hombres, y a estudiantes de
pregrado y posgrado? Si un ítem funciona distinto según el grupo (a igual
nivel de bienestar, un grupo responde sistemáticamente más alto), parte de
la diferencia entre grupos sería un artefacto de la medición.

Método: regresión logística ordinal (Zumbo, 1999), un ítem a la vez,
emparejando por el puntaje total de la escala:

    M1: ítem ~ total
    M2: ítem ~ total + grupo                  → M2 vs M1: DIF UNIFORME
    M3: ítem ~ total + grupo + grupo × total  → M3 vs M2: DIF NO UNIFORME

Significación: razón de verosimilitudes (χ², 1 gl) con Bonferroni sobre
los 29 ítems. Relevancia práctica: ΔR² de Nagelkerke entre M3 y M1 con los
umbrales de Jodoin y Gierl (2001): < 0,035 despreciable, 0,035–0,070
moderado, > 0,070 grande. Con n ≈ 1.800 casi cualquier diferencia es
"significativa"; lo que decide si un ítem es un problema es el tamaño.

El DIF es una propiedad del instrumento en la muestra: se estima sin
factores de expansión.
"""
import warnings
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

UMBRAL_MODERADO = 0.035
UMBRAL_GRANDE = 0.070


@dataclass
class ItemDIF:
    item: str
    n: int
    lr_uniforme: float
    p_uniforme: float
    lr_no_uniforme: float
    p_no_uniforme: float
    delta_r2: float            # Nagelkerke, M3 − M1
    direccion: float           # coef. de grupo en M2 (> 0: el grupo focal puntúa más alto)
    p_ajustado: Optional[float] = None   # Bonferroni sobre el p de M3 vs M1

    @property
    def magnitud(self) -> str:
        if self.delta_r2 >= UMBRAL_GRANDE:
            return "grande"
        if self.delta_r2 >= UMBRAL_MODERADO:
            return "moderado"
        return "despreciable"

    @property
    def tiene_dif(self) -> bool:
        """DIF relevante: significativo tras Bonferroni Y no despreciable."""
        p = self.p_ajustado if self.p_ajustado is not None else 1.0
        return p < 0.05 and self.magnitud != "despreciable"


def _chi2_1gl_p(lr: float) -> float:
    """p-valor de una χ² con 1 gl: P(χ²₁ > x) = 2·(1 − Φ(√x)). Sin scipy."""
    if lr <= 0:
        return 1.0
    return 2 * (1 - NormalDist().cdf(np.sqrt(lr)))


def _chi2_2gl_p(lr: float) -> float:
    """p-valor de una χ² con 2 gl: exp(−x/2)."""
    return float(np.exp(-lr / 2)) if lr > 0 else 1.0


def _loglik_ordinal(y: np.ndarray, X: np.ndarray):
    """Log-verosimilitud del modelo logístico ordinal (odds proporcionales)."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = OrderedModel(y, X, distr="logit")
        ajuste = modelo.fit(method="bfgs", disp=False, maxiter=500)
    return float(ajuste.llf), ajuste


def _nagelkerke(ll: float, ll0: float, n: int) -> float:
    r2_cs = 1 - np.exp(2 * (ll0 - ll) / n)
    maximo = 1 - np.exp(2 * ll0 / n)
    return float(r2_cs / maximo)


def dif_item(respuesta: pd.Series, total: pd.Series, grupo: pd.Series) -> ItemDIF:
    """
    DIF de un ítem.

    Args:
        respuesta: respuestas al ítem (categorías ordenadas, p. ej. 1-6).
        total: puntaje total de la escala (criterio de emparejamiento).
        grupo: 1 = grupo focal, 0 = grupo de referencia.
    """
    datos = pd.DataFrame({"y": respuesta, "t": total, "g": grupo}).dropna()
    # Codigos 0..K-1 en el orden de las categorias (OrderedModel los ordena)
    y = pd.Categorical(datos["y"].astype(int), ordered=True).codes
    t = ((datos["t"] - datos["t"].mean()) / datos["t"].std()).to_numpy()   # estabiliza BFGS
    g = datos["g"].astype(float).to_numpy()
    n = len(datos)

    # Modelo nulo (solo umbrales): LL a partir de las proporciones
    conteos = np.bincount(y)
    conteos = conteos[conteos > 0]
    ll0 = float(np.sum(conteos * np.log(conteos / n)))

    ll1, _ = _loglik_ordinal(y, np.column_stack([t]))
    ll2, aj2 = _loglik_ordinal(y, np.column_stack([t, g]))
    ll3, _ = _loglik_ordinal(y, np.column_stack([t, g, t * g]))

    lr_u = max(2 * (ll2 - ll1), 0.0)
    lr_nu = max(2 * (ll3 - ll2), 0.0)
    return ItemDIF(
        item=str(respuesta.name),
        n=n,
        lr_uniforme=lr_u,
        p_uniforme=_chi2_1gl_p(lr_u),
        lr_no_uniforme=lr_nu,
        p_no_uniforme=_chi2_1gl_p(lr_nu),
        delta_r2=_nagelkerke(ll3, ll0, n) - _nagelkerke(ll1, ll0, n),
        direccion=float(np.asarray(aj2.params)[1]),
    )


def dif_escala(
    items: pd.DataFrame,
    grupo: pd.Series,
    focal: str,
    referencia: str,
    etiquetas: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    DIF de todos los ítems frente a una variable de grupo.

    Args:
        items: una columna por ítem (ya invertidos los negativos).
        grupo: variable de agrupación (p. ej. Género).
        focal, referencia: categorías a comparar; las demás se excluyen.
        etiquetas: nombre corto para mostrar cada ítem.
    """
    mascara = grupo.isin([focal, referencia])
    X = items[mascara].astype(float)
    g = (grupo[mascara] == focal).astype(float)
    total = X.sum(axis=1)

    resultados: List[ItemDIF] = []
    for col in X.columns:
        r = dif_item(X[col], total, g)
        if etiquetas:
            r.item = etiquetas.get(col, r.item)
        resultados.append(r)

    m = len(resultados)
    for r in resultados:
        # p conjunto de M3 vs M1 (2 gl), ajustado por Bonferroni
        r.p_ajustado = min(_chi2_2gl_p(r.lr_uniforme + r.lr_no_uniforme) * m, 1.0)

    con_dif = [r for r in resultados if r.tiene_dif]
    significativos = [r for r in resultados if (r.p_ajustado or 1) < 0.05]
    tabla = pd.DataFrame([{
        "item": r.item,
        "n": r.n,
        "chi2_uniforme": round(r.lr_uniforme, 2),
        "chi2_no_uniforme": round(r.lr_no_uniforme, 2),
        "p_ajustado": round(r.p_ajustado, 4),
        "delta_r2": round(r.delta_r2, 4),
        "magnitud": r.magnitud,
        "favorece_a": (focal if r.direccion > 0 else referencia),
    } for r in resultados]).sort_values("delta_r2", ascending=False)

    return {
        "comparacion": f"{focal} vs. {referencia}",
        "n_focal": int(g.sum()),
        "n_referencia": int((1 - g).sum()),
        "n_items": m,
        "items_significativos_bonferroni": len(significativos),
        "items_con_dif_relevante": [r.item for r in con_dif],
        "delta_r2_maximo": round(max(r.delta_r2 for r in resultados), 4),
        "tabla": tabla,
        "conclusion": (
            "Ningún ítem presenta DIF de magnitud relevante (ΔR² < 0,035 en "
            "todos): la escala permite comparar estos grupos."
            if not con_dif else
            f"{len(con_dif)} ítem(s) con DIF relevante: las comparaciones "
            f"entre estos grupos deben interpretarse con cautela."
        ),
    }
