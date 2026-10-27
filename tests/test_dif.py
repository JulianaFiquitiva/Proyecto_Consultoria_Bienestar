# -*- coding: utf-8 -*-
"""
DIF por regresion logistica ordinal (entrega 2, pregunta 2.2).

Datos simulados con un modelo de respuesta graduada: a igual bienestar
latente, el grupo focal responde igual que el de referencia, salvo en el
item al que se le planta DIF.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from agent.core.orchestrator import StatisticalAgent
from agent.psychometrics.dif import _chi2_1gl_p, _chi2_2gl_p, dif_escala


def _simular(n=700, n_items=6, item_dif=None, desplazamiento=1.5, seed=11):
    rng = np.random.default_rng(seed)
    grupo = np.where(rng.random(n) < 0.5, "F", "M")
    # El grupo focal tiene MAS bienestar: eso no es DIF (es impacto)
    theta = rng.normal(0, 1, n) + np.where(grupo == "F", 0.3, 0)
    umbrales = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    items = {}
    for j in range(n_items):
        latente = 1.5 * theta + rng.logistic(0, 1, n)
        if j == item_dif:
            latente = latente + np.where(grupo == "F", desplazamiento, 0)
        items[f"item{j + 1}"] = 1 + np.searchsorted(umbrales, latente)
    return pd.DataFrame(items), pd.Series(grupo)


def test_p_valores_chi2_sin_scipy():
    assert _chi2_1gl_p(3.841459) == pytest.approx(0.05, abs=1e-4)
    assert _chi2_2gl_p(5.991465) == pytest.approx(0.05, abs=1e-4)
    assert _chi2_1gl_p(0) == 1.0


def test_sin_dif_no_marca_ningun_item():
    items, grupo = _simular(item_dif=None)
    r = dif_escala(items, grupo, "F", "M")
    assert r["items_con_dif_relevante"] == []
    assert r["delta_r2_maximo"] < 0.035
    assert "permite comparar" in r["conclusion"]


def test_detecta_el_item_con_dif_plantado():
    items, grupo = _simular(item_dif=2, desplazamiento=2.0)
    r = dif_escala(items, grupo, "F", "M")
    assert r["items_con_dif_relevante"] == ["item3"]
    fila = r["tabla"].iloc[0]
    assert fila["item"] == "item3" and fila["favorece_a"] == "F"
    assert fila["magnitud"] in ("moderado", "grande")


def test_excluye_otras_categorias_y_usa_etiquetas():
    items, grupo = _simular(n=400)
    grupo.iloc[:10] = "Otro"
    r = dif_escala(items, grupo, "F", "M", etiquetas={"item1": "1. Primero"})
    assert r["n_focal"] + r["n_referencia"] == 390
    assert "1. Primero" in set(r["tabla"]["item"])


@pytest.mark.parametrize("pregunta,grupo", [
    ("¿La escala mide igual a hombres y mujeres?", "genero"),
    ("¿Hay DIF en los items por genero?", "genero"),
    ("¿Hay funcionamiento diferencial de los ítems entre pregrado y posgrado?", "nivel"),
    ("¿Hay invarianza de la escala por nivel de formacion?", "nivel"),
])
def test_enrutado_a_dif(pregunta, grupo):
    plan = StatisticalAgent()._generate_plan(pregunta)
    assert plan.function_to_call == "dif_analysis"
    assert plan.parameters["group"] == grupo
