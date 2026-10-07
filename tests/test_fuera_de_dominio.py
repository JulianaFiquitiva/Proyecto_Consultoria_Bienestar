# -*- coding: utf-8 -*-
"""
P1 - Ruta de abstencion: preguntas FUERA del dominio del datos.

Cubre los 3 casos reales registrados por el profesor (texto NO modificado)
y verifica que:
  - caen en la categoria "fuera_de_dominio"
  - NO se llama al LLM para redactar
  - NO se ejecuta ningun calculo estadístico
  - la trazabilidad no cita ningun dataset

Ademas comprueba que las preguntas validas siguen funcionando igual.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.core.orchestrator import AnalysisType, StatisticalAgent


# ── Casos REALES registrados (texto literal, con sus typos originales) ──
PREGUNTAS_FUERA_DE_DOMINIO = [
    "Existen más estudios realcionados con el bienestar estudiantil?",
    "en donde puedo consultar más sobre temas de bienestar estudaintil?",
    "en otros paises de latinoamerica hay más estudios relacionados?",
]

PREGUNTA_VALIDA = "¿Hay diferencia de bienestar entre hombres y mujeres?"
PREGUNTA_BUSQUEDA = "dame referencias sobre bienestar estudiantil"


@pytest.fixture(scope="module")
def agent():
    """Se crea UNA sola vez por modulo (load_data tarda ~20s)."""
    a = StatisticalAgent()
    a.load_data()
    return a


@pytest.fixture
def llm_spy(agent):
    """Espia que cuenta las llamadas al LLM. Se restaura tras cada test."""
    counter = {"n": 0}
    original = agent.llm.chat

    def spy(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    agent.llm.chat = spy
    yield counter
    agent.llm.chat = original


# ── Los 3 casos reales ──────────────────────────────────────────────────

class TestTresCasosRegistrados:
    @pytest.mark.parametrize("pregunta", PREGUNTAS_FUERA_DE_DOMINIO)
    def test_categoria_fuera_de_dominio(self, agent, llm_spy, pregunta):
        plan = agent._generate_plan(pregunta)
        assert plan.analysis_type == AnalysisType.OUT_OF_DOMAIN, (
            f"Esperaba fuera_de_dominio, llego {plan.analysis_type}"
        )
        assert plan.function_to_call == "abstencion"

    @pytest.mark.parametrize("pregunta", PREGUNTAS_FUERA_DE_DOMINIO)
    def test_no_llama_al_llm(self, agent, llm_spy, pregunta):
        resp = agent.ask(pregunta)
        assert llm_spy["n"] == 0, (
            f"Se llamó al LLM {llm_spy['n']} veces y no debia llamarse"
        )
        assert resp.success is True

    @pytest.mark.parametrize("pregunta", PREGUNTAS_FUERA_DE_DOMINIO)
    def test_no_inventa_resultados(self, agent, llm_spy, pregunta):
        resp = agent.ask(pregunta)
        texto = resp.interpretation
        # El bug original decia "1,813 estudios" / "1813 paises"
        assert "1,813 estudios" not in texto
        assert "1813" not in texto
        assert "Media T-score" not in texto
        assert "NO SE EJECUTO NINGUN ANALISIS" in texto

    @pytest.mark.parametrize("pregunta", PREGUNTAS_FUERA_DE_DOMINIO)
    def test_explica_el_motivo(self, agent, llm_spy, pregunta):
        resp = agent.ask(pregunta)
        assert "Motivo:" in resp.interpretation

    @pytest.mark.parametrize("pregunta", PREGUNTAS_FUERA_DE_DOMINIO)
    def test_lista_ejemplos_validos(self, agent, llm_spy, pregunta):
        resp = agent.ask(pregunta)
        texto = resp.interpretation
        assert "Que SI puedo responder" in texto
        assert "hombres y mujeres" in texto

    @pytest.mark.parametrize("pregunta", PREGUNTAS_FUERA_DE_DOMINIO)
    def test_trazabilidad_sin_dataset(self, agent, llm_spy, pregunta):
        resp = agent.ask(pregunta)
        assert resp.traceability["dataset"] == "ninguno"
        assert resp.traceability["method"] == "abstencion"

    @pytest.mark.parametrize("pregunta", PREGUNTAS_FUERA_DE_DOMINIO)
    def test_no_usa_columnas(self, agent, llm_spy, pregunta):
        resp = agent.ask(pregunta)
        assert resp.traceability["columns_used"] == []


# ── Preguntas validas: no deben romperse ─────────────────────────────────

class TestPreguntasValidas:
    def test_comparacion_genero_sigue_funcionando(self, agent, llm_spy):
        plan = agent._generate_plan(PREGUNTA_VALIDA)
        assert plan.analysis_type != AnalysisType.OUT_OF_DOMAIN
        assert plan.analysis_type != AnalysisType.AMBIGUOUS
        assert plan.function_to_call != "abstencion"

        resp = agent.ask(PREGUNTA_VALIDA)
        assert resp.success is True
        # Este caso SI debe usar el dataset
        assert resp.traceability["dataset"] != "ninguno"
        assert ".xlsx" in resp.traceability["dataset"]
        # Y SI debe llamar al LLM para interpretar
        assert llm_spy["n"] >= 1

    def test_busqueda_sigue_yendo_a_literature_search(self, agent, llm_spy):
        plan = agent._generate_plan(PREGUNTA_BUSQUEDA)
        assert plan.analysis_type == AnalysisType.SEARCH
        assert plan.function_to_call == "literature_search"

    def test_busqueda_no_cita_dataset(self, agent, llm_spy):
        resp = agent.ask(PREGUNTA_BUSQUEDA)
        assert resp.traceability["dataset"] == "ninguno"
        assert resp.traceability["method"] == "literature_search"

    def test_resumen_explicito_sigue_dando_overview(self, agent, llm_spy):
        plan = agent._generate_plan("¿Cual es el resumen general del bienestar?")
        assert plan.function_to_call == "descriptive_overview"


# ── Categoria agregada en el planner ─────────────────────────────────────

class TestCategoriaEnElPlanner:
    def test_clasificador_devuelve_fuera_de_dominio(self):
        from agent.core.query_planner import classify_scope
        for p in PREGUNTAS_FUERA_DE_DOMINIO:
            cat, motivo = classify_scope(p)
            assert cat == "fuera_de_dominio", (p, cat)
            assert motivo, "debe llevar motivo"

    def test_clasificador_reconoce_dominio(self):
        from agent.core.query_planner import classify_scope
        cat, _ = classify_scope(PREGUNTA_VALIDA)
        assert cat == "en_dominio"

    def test_plan_query_no_usa_overview_por_defecto(self):
        import pandas as pd
        from agent.core.query_planner import QueryPlanner

        df = pd.DataFrame({"T_Global": [50.0, 60.0, 70.0]})
        planner = QueryPlanner(df)

        for p in PREGUNTAS_FUERA_DE_DOMINIO:
            plan = planner.plan_query(p)
            assert plan.analysis_type == "fuera_de_dominio", (p, plan.analysis_type)
            assert plan.method == "abstencion"
            assert plan.value_variable is None

    def test_plan_query_ambigua_cuando_no_reconoce(self):
        import pandas as pd
        from agent.core.query_planner import QueryPlanner

        df = pd.DataFrame({"T_Global": [50.0, 60.0, 70.0]})
        planner = QueryPlanner(df)
        plan = planner.plan_query("hola que tal")
        assert plan.analysis_type == "ambigua"
        assert plan.method == "abstencion"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
