# -*- coding: utf-8 -*-
"""
El agente responde con estimaciones POBLACIONALES (entrega 2).

Sin LLM, sin red y sin los Excel de la encuesta: el dataset es sintetico
pero tiene las columnas del diseno real (seccional, modalidad_exp,
factor_expansion, N_poblacion_estrato).
"""
import numpy as np
import pandas as pd
import pytest

from agent.core.orchestrator import AnalysisType, StatisticalAgent
from agent.ui.cli import main as _cli_main  # noqa: F401  (la CLI importa)

# Los comandos rapidos de la CLI (agent/ui/cli.py) y la funcion esperada
COMANDOS_CLI = {
    "Cual es el resumen general del bienestar?": "descriptive_overview",
    "Como difiere el bienestar por genero?": "design_comparison",
    "Como varia el bienestar entre seccionales?": "design_comparison",
    "Como difiere el bienestar entre pregrado y posgrado?": "design_comparison",
    "Como varia el bienestar por estrato socioeconomico?": "design_comparison",
    "Cual es el perfil por dimensiones de bienestar?": "dimension_profile",
    "Valida los resultados contra el estudio anterior": "validate_results",
    "Haz una validacion completa del pipeline": "full_validation",
    # Pregunta 2.3 (no es comando rapido, pero se pregunta en la demo)
    "¿La brecha de posgrado se mantiene controlando la edad?": "adjusted_gap",
    "¿La diferencia entre pregrado y posgrado se explica por la edad?": "adjusted_gap",
}


def _dataset(seed=7):
    rng = np.random.default_rng(seed)
    n = 600
    seccional = rng.choice(["Bogotá", "Tunja", "Villavicencio"], n)
    modalidad = rng.choice(["Pregrado", "Posgrado"], n, p=[0.75, 0.25])
    # La brecha de posgrado es TODA de edad: T sube 0.4 por año y el
    # posgrado es ~10 años mayor; el nivel no tiene efecto propio.
    edad = np.where(modalidad == "Posgrado", rng.normal(32, 6, n), rng.normal(21, 3, n))
    df = pd.DataFrame({
        "seccional": seccional,
        "modalidad_exp": modalidad,
        "Edad": edad.round(),
        "Género": rng.choice(["Femenino", "Masculino"], n),
        "Estrato socioeconómico": rng.choice([1.0, 2.0, 3.0, 4.0], n),
        "T_Global": 41 + 0.4 * edad + rng.normal(0, 8, n),
    })
    df.loc[:4, "Género"] = "Respuesta libre de una persona"   # n = 5 < 30
    tamanos = {("Bogotá", "Pregrado"): 9000, ("Bogotá", "Posgrado"): 3000,
               ("Tunja", "Pregrado"): 5000, ("Tunja", "Posgrado"): 1500,
               ("Villavicencio", "Pregrado"): 4000, ("Villavicencio", "Posgrado"): 500}
    df["N_poblacion_estrato"] = [tamanos[(s, m)] for s, m in zip(seccional, modalidad)]
    df["factor_expansion"] = df["N_poblacion_estrato"] / df.groupby(
        ["seccional", "modalidad_exp"])["T_Global"].transform("size")
    from agent.core.config import DIMENSION_ORDER
    for d in DIMENSION_ORDER:
        df[f"T_{d}"] = rng.normal(50, 10, n)
    return df


@pytest.fixture
def agente():
    a = StatisticalAgent()
    assert not a.llm.is_configured, "las pruebas sin red no deben tener clave"
    a._irt_data = _dataset()
    return a


class TestEnrutadoDeComandosCli:
    @pytest.mark.parametrize("pregunta,funcion", COMANDOS_CLI.items())
    def test_cada_comando_va_a_su_funcion(self, agente, pregunta, funcion):
        assert agente._generate_plan(pregunta).function_to_call == funcion

    def test_como_varia_no_es_como_va(self, agente):
        """Regresion: 'como varia' contenia 'como va' y caia en el resumen."""
        plan = agente._generate_plan("Como varia el bienestar entre seccionales?")
        assert plan.analysis_type == AnalysisType.COMPARISON


class TestResumenPoblacional:
    def test_media_ponderada_con_ic(self, agente):
        r = agente._descriptive_overview({})
        df = agente._irt_data
        esperada = np.average(df["T_Global"], weights=df["factor_expansion"])
        assert r["media"] == pytest.approx(esperada, abs=0.01)
        assert r["ic_95"][0] < r["media"] < r["ic_95"][1]
        assert r["media_muestral_sin_ponderar"] == pytest.approx(df["T_Global"].mean(), abs=0.01)

    def test_el_texto_muestra_el_intervalo(self, agente):
        resp = agente.ask("Cual es el resumen general del bienestar?")
        assert resp.success
        assert "poblacional" in resp.interpretation and "IC 95%" in resp.interpretation


class TestComparacionConDiseno:
    def test_dos_grupos_detecta_la_brecha(self, agente):
        r = agente._design_comparison(
            {"value_col": "T_Global", "group_col": "modalidad_exp"})
        prueba = r["prueba_estadistica"]
        assert prueba["test"].startswith("Wald")
        assert prueba["significativo"] is True
        assert set(r["estadisticos_por_grupo"]) == {"Pregrado", "Posgrado"}

    def test_varios_grupos_con_bonferroni(self, agente):
        r = agente._design_comparison(
            {"value_col": "T_Global", "group_col": "seccional"})
        assert len(r["comparaciones"]) == 3
        assert "Bonferroni" in r["prueba_estadistica"]["test"]

    def test_grupo_numerico(self, agente):
        r = agente._design_comparison(
            {"value_col": "T_Global", "group_col": "Estrato socioeconómico"})
        assert len(r["comparaciones"]) == 6

    def test_grupos_pequenos_se_excluyen_sin_mostrar_su_etiqueta(self, agente):
        r = agente._design_comparison({"value_col": "T_Global", "group_col": "Género"})
        assert "Respuesta libre de una persona" not in r["estadisticos_por_grupo"]
        texto = " ".join(r["advertencias"])
        assert "1 categoria" in texto and "5 casos" in texto
        assert "Respuesta libre" not in texto

    def test_variable_inexistente(self, agente):
        r = agente._design_comparison({"value_col": "T_Global", "group_col": "No existe"})
        assert "error" in r


class TestBrechaAjustada:
    def test_la_edad_explica_la_brecha(self, agente):
        """En los datos sinteticos la brecha es toda de edad."""
        r = agente._adjusted_gap({})
        m0, m1 = r["modelos"][0], r["modelos"][1]
        assert m0["brecha"] > 3 and m0["p_valor"] < 0.001
        assert m1["ic_inferior"] < 0 < m1["ic_superior"]
        assert r["porcentaje_explicado_por_controles"] > 60
        assert r["edad_media_ponderada"]["Posgrado"] > r["edad_media_ponderada"]["Pregrado"]

    def test_todos_los_modelos_usan_la_misma_muestra(self, agente):
        agente._irt_data.loc[:19, "Estrato socioeconómico"] = np.nan
        r = agente._adjusted_gap({})
        assert len({m["n"] for m in r["modelos"]}) == 1
        assert r["n_excluidos_por_faltantes"] == 20
        # las filas 0-4 eran las unicas de "otro genero": ese control se descarta
        assert r["controles_descartados_sin_variacion"] == ["Otro género (vs. masculino)"]

    def test_respuesta_completa(self, agente):
        resp = agente.ask("¿La brecha de posgrado se mantiene controlando la edad?")
        assert resp.success, resp.error
        assert "M0: brecha bruta" in resp.interpretation
        assert "Modelo completo" in resp.interpretation


def test_perfil_por_dimensiones_con_ic(agente):
    perfil = agente._dimension_profile({})
    assert all("ic_95" in v for k, v in perfil.items() if k != "_resumen")
