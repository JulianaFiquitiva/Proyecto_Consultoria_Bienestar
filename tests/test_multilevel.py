# -*- coding: utf-8 -*-
"""
Modelo multinivel (entrega 2, pregunta 2.4): ICC con solucion conocida.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from agent.core.orchestrator import StatisticalAgent
from agent.statistics.multilevel import _p_borde, icc


def _simular(tau2, sigma2=75.0, n_grupos=60, por_grupo=40, seed=5):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_grupos), por_grupo)
    u = rng.normal(0, np.sqrt(tau2), n_grupos)[g] if tau2 > 0 else 0.0
    return pd.DataFrame({
        "y": 50 + u + rng.normal(0, np.sqrt(sigma2), len(g)),
        "programa": [f"P{i:02d}" for i in g],
    })


def test_recupera_el_icc_simulado():
    r = icc(_simular(tau2=25.0), "y", "programa")       # ICC verdadero = 0.25
    assert r["icc"] == pytest.approx(0.25, abs=0.07)
    assert r["p_valor_tau2_cero"] < 0.001


def test_sin_efecto_de_grupo():
    r = icc(_simular(tau2=0.0), "y", "programa")
    assert r["icc"] < 0.03
    assert r["p_valor_tau2_cero"] > 0.01


def test_ranking_excluye_grupos_pequenos():
    df = _simular(tau2=25.0, n_grupos=10, por_grupo=40)
    df = pd.concat([df, pd.DataFrame({"y": [90.0] * 5, "programa": "Diminuto"})])
    r = icc(df, "y", "programa", min_n_ranking=30)
    nombres = {g["grupo"] for g in r["grupos_mas_altos"] + r["grupos_mas_bajos"]}
    assert "Diminuto" not in nombres


def test_p_valor_de_borde():
    # 0.5 · P(χ²₁ > 2.706) = 0.05
    assert _p_borde(2.705543) == pytest.approx(0.05, abs=1e-4)
    assert _p_borde(0.0) == 1.0


@pytest.mark.parametrize("pregunta", [
    "¿Cuanto depende el bienestar del programa?",
    "¿Cual es la correlacion intraclase por programa?",
    "Haz un modelo multinivel del bienestar",
])
def test_enrutado_a_multinivel(pregunta):
    assert StatisticalAgent()._generate_plan(pregunta).function_to_call == "multilevel_icc"
