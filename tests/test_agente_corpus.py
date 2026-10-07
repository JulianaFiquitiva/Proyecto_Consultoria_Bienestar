# -*- coding: utf-8 -*-
"""
Integracion RAG <-> agente estadistico.

Comprueba, SIN llamar al LLM y SIN tocar la API, que:

  1. el prefijo "corpus:" / "fuentes:" enruta a ask_corpus
  2. una pregunta que pide lo que DICEN los documentos no acaba ni en la
     abstinencia de "fuera de dominio" ni en un analisis sobre el dataset
  3. la trazabilidad cita el corpus, no el .xlsx de la encuesta
  4. las fuentes salen con formato [archivo, p. N]
  5. la abstinencia por umbral sigue siendo abstinencia (no inventa fuentes)
  6. si el indice no se puede cargar, el agente falla con mensaje claro
  7. una pregunta estadistica normal NO toca el RAG
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.core.orchestrator import (
    AnalysisType,
    StatisticalAgent,
    pregunta_sobre_corpus,
)

PREGUNTA_DOCUMENTOS = "¿Qué dicen los documentos sobre la relación entre el sueño y el bienestar?"
PREGUNTA_ESTADISTICA = "¿Hay diferencia de bienestar entre hombres y mujeres?"
FUERA_DE_DOMINIO = "Existen más estudios realcionados con el bienestar estudiantil?"


# ── dobles (nada de API, nada de LLM) ───────────────────────────────────

def _respuesta_ok(pregunta: str):
    from agent.rag.rag_engine import Fuente, RespuestaRAG
    return RespuestaRAG(
        pregunta=pregunta,
        respuesta="El estudio reporta 128 estudiantes. [1, p. 2]",
        fuentes=[Fuente(archivo="08-+7794.pdf", pagina=2,
                        fragmento_textual="...el 60% (n = 128)...",
                        puntaje=0.1033)],
        fragmentos=[], abstencion=False, puntaje_max=0.1033,
        backend="tfidf-local@60000", llamada_llm=True,
    )


def _respuesta_abstiene(pregunta: str):
    from agent.rag.rag_engine import RespuestaRAG
    return RespuestaRAG(
        pregunta=pregunta,
        respuesta="No encuentro una fuente en el corpus que responda esto.",
        fuentes=[], fragmentos=[], abstencion=True, puntaje_max=0.0686,
        backend="tfidf-local@60000", llamada_llm=False,
    )


def _respuesta_llm_falla(pregunta: str):
    """El corpus SI supera el umbral, pero la llamada al LLM revienta
    (sin cuota, sin red, modelo caido...)."""
    from agent.rag.rag_engine import RespuestaRAG
    return RespuestaRAG(
        pregunta=pregunta,
        respuesta="No encuentro una fuente en el corpus que responda esto.",
        fuentes=[], fragmentos=[], abstencion=True, puntaje_max=0.0769,
        backend="tfidf-local@60000", llamada_llm=True,
        error="Error code: 429 - credit_balance_exhausted",
    )


class _IndiceFalso:
    backend = "tfidf-local@test"


@pytest.fixture
def agente():
    """Sin load_data(): el camino del corpus no usa el dataset de la encuesta."""
    return StatisticalAgent()


@pytest.fixture
def rag_ok(monkeypatch):
    """ask_with_sources devuelve una respuesta con fuente. Devuelve el espiar."""
    llamadas = []

    def fake(pregunta, **kwargs):
        llamadas.append(pregunta)
        return _respuesta_ok(pregunta)

    monkeypatch.setattr("agent.rag.rag_engine.ask_with_sources", fake)
    monkeypatch.setattr("agent.rag.retriever.Retriever",
                        lambda *a, **k: _IndiceFalso())
    return llamadas


@pytest.fixture
def rag_abstiene(monkeypatch):
    def fake(pregunta, **kwargs):
        return _respuesta_abstiene(pregunta)

    monkeypatch.setattr("agent.rag.rag_engine.ask_with_sources", fake)
    monkeypatch.setattr("agent.rag.retriever.Retriever",
                        lambda *a, **k: _IndiceFalso())


@pytest.fixture
def indice_roto(monkeypatch):
    def roto(*a, **k):
        raise RuntimeError("chroma bloqueado")

    monkeypatch.setattr("agent.rag.retriever.Retriever", roto)


# ── 1. enrutado ─────────────────────────────────────────────────────────

class TestEnrutado:
    @pytest.mark.parametrize("texto,esperada", [
        ("corpus: ¿Qué dicen los documentos?", "¿Qué dicen los documentos?"),
        ("corpus ¿Qué dicen los documentos?", "¿Qué dicen los documentos?"),
        ("fuentes: ¿Qué dice la EBUP?", "¿Qué dice la EBUP?"),
        ("literatura: ¿qué tal?", "¿qué tal?"),
    ])
    def test_prefijo_extrae_la_pregunta(self, texto, esperada):
        assert pregunta_sobre_corpus(texto) == esperada

    @pytest.mark.parametrize("texto", [
        PREGUNTA_ESTADISTICA,
        FUERA_DE_DOMINIO,
        "dame referencias sobre bienestar estudiantil",
        "fuentes del estudio",            # sin separador: no es prefijo
        "",
    ])
    def test_sin_prefijo_devuelve_none(self, texto):
        assert pregunta_sobre_corpus(texto) is None

    @pytest.mark.parametrize("texto", [
        PREGUNTA_DOCUMENTOS,
        "¿Cuántos documentos conformaron la revisión de Méndez Durán?",
        "¿Qué encuentran los estudios del corpus sobre bienestar?",
        # "literatura" preguntada por CONTENIDO -> corpus, no busqueda web
        "¿Qué dice la literatura sobre el bienestar estudiantil?",
        "Según la literatura, ¿el sueño se asocia al bienestar?",
        "La literatura reporta que el ejercicio mejora el bienestar.",
        "En la literatura se describe el bienestar como multidimensional.",
    ])
    def test_pregunta_de_corpus_detectada(self, texto):
        from agent.core.orchestrator import _RE_PREGUNTA_CORPUS
        assert _RE_PREGUNTA_CORPUS.search(texto), texto

    @pytest.mark.parametrize("texto", [
        PREGUNTA_ESTADISTICA,
        FUERA_DE_DOMINIO,
        "dame referencias sobre bienestar estudiantil",
        "dame literatura sobre bienestar estudiantil",   # peticion -> web
        "busca artículos sobre ansiedad y bienestar",    # peticion -> web
        "¿Cuál es la media global de bienestar (T-score)?",
    ])
    def test_pregunta_estadistica_no_detectada(self, texto):
        from agent.core.orchestrator import _RE_PREGUNTA_CORPUS
        assert not _RE_PREGUNTA_CORPUS.search(texto), texto

    @pytest.mark.parametrize("texto", [
        "¿Qué dice la literatura sobre el bienestar estudiantil?",
        PREGUNTA_DOCUMENTOS,
    ])
    def test_pregunta_de_literatura_es_corpus_no_busqueda(self, agente, texto):
        assert agente._generate_plan(texto).analysis_type == AnalysisType.CORPUS

    @pytest.mark.parametrize("texto", [
        "dame referencias sobre bienestar estudiantil",
        "dame literatura sobre bienestar estudiantil",
        "busca artículos sobre ansiedad y bienestar",
    ])
    def test_peticion_de_bibliografia_sigue_siendo_busqueda(self, agente, texto):
        assert agente._generate_plan(texto).analysis_type == AnalysisType.SEARCH

    def test_fuera_de_dominio_no_se_desvia_al_corpus(self, agente):
        assert agente._generate_plan(FUERA_DE_DOMINIO).analysis_type \
            == AnalysisType.OUT_OF_DOMAIN


# ── 2-5. ask_corpus / ask() ─────────────────────────────────────────────

class TestElAgenteUsaElRag:
    def test_prefijo_enruta_a_rag(self, agente, rag_ok):
        resp = agente.ask("corpus: " + PREGUNTA_ESTADISTICA)
        assert rag_ok, "el prefijo no llego a ask_with_sources"
        assert resp.success is True
        assert resp.plan.analysis_type == AnalysisType.CORPUS
        assert resp.traceability["method"].startswith("rag")

    def test_pregunta_documentos_no_acaba_en_abstencion(self, agente, rag_ok):
        resp = agente.ask(PREGUNTA_DOCUMENTOS)
        assert rag_ok, "la pregunta no llego al RAG"
        assert resp.plan.analysis_type == AnalysisType.CORPUS
        assert resp.success is True
        assert "FUERA DE DOMINIO" not in resp.interpretation
        assert "PREGUNTA AMBIGUA" not in resp.interpretation

    def test_trazabilidad_cita_el_corpus_no_el_dataset(self, agente, rag_ok):
        resp = agente.ask(PREGUNTA_DOCUMENTOS)
        assert "rag_corpus" in resp.traceability["dataset"]
        assert ".xlsx" not in resp.traceability["dataset"]
        assert resp.traceability["columns_used"] == []

    def test_fuentes_con_archivo_y_pagina(self, agente, rag_ok):
        resp = agente.ask(PREGUNTA_DOCUMENTOS)
        assert resp.result["fuentes"], "no devolvio fuentes"
        fuente = resp.result["fuentes"][0]
        assert fuente["archivo"] == "08-+7794.pdf"
        assert fuente["pagina"] == 2
        assert "[08-+7794.pdf, p. 2]" in resp.interpretation

    def test_abstencion_por_umbral_no_inventa_fuentes(self, agente, rag_abstiene):
        resp = agente.ask(PREGUNTA_DOCUMENTOS)
        assert resp.success is True            # abstenerse es un resultado valido
        assert resp.result["abstencion"] is True
        assert resp.result["fuentes"] == []
        assert resp.result["llamada_llm"] is False
        assert "No encuentro una fuente en el corpus" in resp.interpretation
        assert any("Abstinencia" in w for w in resp.warnings)

    def test_fallo_del_llm_no_se_reporta_como_falta_del_corpus(
            self, agente, monkeypatch):
        """Si revienta la llamada (p. ej. sin cuota en OpenAI) el aviso debe
        decir que fallo la llamada, NO que el corpus no tiene la informacion:
        confundir las dos lleva a conclusiones falsas sobre la literatura."""
        monkeypatch.setattr(
            "agent.rag.rag_engine.ask_with_sources",
            lambda *a, **k: _respuesta_llm_falla(""))
        resp = agente.ask(PREGUNTA_DOCUMENTOS)
        assert resp.success is True
        assert resp.result["llamada_llm"] is True
        assert any("fallo la llamada al LLM" in w for w in resp.warnings)
        assert not any("no encontro la respuesta" in w for w in resp.warnings)
        assert resp.error and "credit_balance_exhausted" in resp.error


# ── 6. indice caido ─────────────────────────────────────────────────────

class TestIndiceNoDisponible:
    def test_falla_con_mensaje_claro(self, agente, indice_roto):
        resp = agente.ask("corpus: " + PREGUNTA_DOCUMENTOS)
        assert resp.success is False
        assert resp.error and "chroma bloqueado" in resp.error
        assert "indice rag" in resp.interpretation.lower()


# ── 7. el camino estadistico no se toca ─────────────────────────────────

class TestElRagNoInvadeElCaminoEstadistico:
    def test_pregunta_fuera_de_dominio_sigue_absteniendose(self, agente,
                                                           monkeypatch):
        """Los 3 casos registrados en P1 NO deben ir al corpus."""
        llamadas = []

        def no_debe_llamar(*a, **k):
            llamadas.append(1)
            raise AssertionError("el RAG no debe intervenir")

        monkeypatch.setattr("agent.rag.rag_engine.ask_with_sources",
                            no_debe_llamar)
        resp = agente.ask(FUERA_DE_DOMINIO)
        assert llamadas == []
        assert resp.plan.analysis_type == AnalysisType.OUT_OF_DOMAIN
        assert "NO SE EJECUTO NINGUN ANALISIS" in resp.interpretation

    def test_pregunta_de_busqueda_sigue_yendo_a_la_web(self, agente, monkeypatch):
        llamadas = []
        monkeypatch.setattr(
            "agent.rag.rag_engine.ask_with_sources",
            lambda *a, **k: llamadas.append(1) or _respuesta_ok(""))
        resp = agente.ask("dame referencias sobre bienestar estudiantil")
        assert llamadas == []
        assert resp.plan.analysis_type == AnalysisType.SEARCH
