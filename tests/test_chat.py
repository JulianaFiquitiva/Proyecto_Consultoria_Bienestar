# -*- coding: utf-8 -*-
"""
Motor del chat de la app (agent/core/chat.py).

Usa hallazgos SINTETICOS con la misma estructura del JSON de la entrega 2,
asi que no necesita data/ ni el JSON real. El corpus y la busqueda web se
reemplazan por dobles: ninguna prueba abre el indice ni sale a internet.
"""
import pytest

from agent.core.chat import MotorChat
from agent.core.orchestrator import AgentResponse, StatisticalAgent


def _est(media, ic, n=100, N=1000):
    return {"media": media, "ic": ic, "se": 0.3, "deff": 1.1, "n": n, "N": N}


H = {
    "generado": "2026-10-26T12:00:00",
    "min_n_grupo": 30,
    "resumen": {"T_global": _est(50.68, [50.19, 51.17], 1813, 29950),
                "riesgo": _est(0.148, [0.132, 0.165]), "alto": _est(0.194, [0.175, 0.213]),
                "media_sin_ponderar": 50.44, "n": 1813, "N": 29950},
    "dimensiones": {"Autonomía": _est(50.3, [49.9, 50.7]),
                    "Dominio del entorno": _est(50.5, [50.2, 50.9])},
    "comparaciones": {
        "Género": {"etiqueta": "Género",
                   "grupos": {"Masculino": _est(51.28, [50.57, 51.99], 918),
                              "Femenino": _est(50.13, [49.44, 50.83], 881)},
                   "pares": [{"grupo_1": "Femenino", "grupo_2": "Masculino",
                              "diferencia": -1.147, "p_ajustado_bonferroni": 0.025,
                              "significativo": True}],
                   "excluidos": {"categorias": 4, "casos": 14}},
        "seccional": {"etiqueta": "Seccional",
                      "grupos": {"Bogotá": _est(51.1, [50.2, 51.9], 653),
                                 "Tunja": _est(50.6, [49.8, 51.5], 538)},
                      "pares": [{"grupo_1": "Bogotá", "grupo_2": "Tunja", "diferencia": 0.5,
                                 "p_ajustado_bonferroni": 1.0, "significativo": False}],
                      "excluidos": {"categorias": 0, "casos": 0}},
        "Estrato socioeconómico": {"etiqueta": "Estrato socioeconómico",
                                   "grupos": {"Estrato 3": _est(50.5, [49.8, 51.2], 836)},
                                   "pares": [], "excluidos": {"categorias": 1, "casos": 3}},
    },
    "brecha": {"modelos": [{"modelo": "M0: brecha bruta", "brecha": 4.665,
                            "ic_inferior": 3.45, "ic_superior": 5.88},
                           {"modelo": "M2: + controles", "brecha": 2.574,
                            "ic_inferior": 0.97, "ic_superior": 4.18}],
               "brecha_bruta": 4.665, "brecha_ajustada": 2.574,
               "porcentaje_explicado_por_controles": 44.8,
               "edad_media_ponderada": {"Posgrado": 33.2, "Pregrado": 21.4}},
    "multinivel": {"programa": {
        "nulo": {"icc": 0.098, "n_grupos": 114, "grupos_con_menos_de_10": 76},
        "condicional_nivel_edad": {"icc": 0.047}}, "limitacion": "ICC muestral."},
    "dif": {"genero": {"comparacion": "Femenino vs. Masculino", "n_items": 29,
                       "items_significativos_bonferroni": 8, "delta_r2_maximo": 0.0147,
                       "conclusion": "La escala permite comparar estos grupos.",
                       "tabla": [{"item": "18. Confianza", "delta_r2": 0.0147,
                                  "magnitud": "despreciable"}]},
            "nivel": {"comparacion": "Posgrado vs. Pregrado", "n_items": 29,
                      "items_significativos_bonferroni": 3, "delta_r2_maximo": 0.0112,
                      "conclusion": "La escala permite comparar estos grupos.",
                      "tabla": []}},
    "reproducibilidad": [
        {"Indicador": "T global", "Referencia entrega 1": 50.7, "Reproducido": 50.68,
         "Estado vs. Referencia entrega 1": "VALIDADO"},
        {"Indicador": "Proporción alta", "Referencia entrega 1": 0.192, "Reproducido": 0.1937,
         "Estado vs. Referencia entrega 1": "DIFERENCIA MENOR"}],
}


class AgenteDoble(StatisticalAgent):
    """Agente real para enrutar; corpus, busqueda y ask() quedan registrados."""

    def __init__(self):
        super().__init__()
        self.llamadas = []

    def ask_corpus(self, question):
        self.llamadas.append(("corpus", question))
        plan = self._corpus_plan(question)
        return AgentResponse(
            question=question, plan=plan,
            result={"respuesta": "Usa la escala de Ryff de 29 ítems.",
                    "fuentes": [{"archivo": "doc.pdf", "pagina": 2, "puntaje": 0.3}],
                    "llamada_llm": True},
            interpretation="", warnings=[], visualization_hint=None,
            traceability={"method": "rag (umbral 0.0720)"}, success=True)

    def _literature_search(self, params):
        self.llamadas.append(("busqueda", params["query"]))
        return {"papers": [{"titulo": "Ryff revisitado", "anio": 2020, "doi": "10.1/x"}],
                "advertencias": ["Solo metadatos."]}

    def ask(self, question):
        self.llamadas.append(("ask", question))
        plan = self._fallback_plan(question)
        return AgentResponse(question=question, plan=plan, result={},
                             interpretation="[METODO: correlacion]", warnings=[],
                             visualization_hint=None,
                             traceability={"dataset": "03_dataset_IRT_ponderado.xlsx",
                                           "method": "correlation_analysis"},
                             success=True)


@pytest.fixture
def agente():
    return AgenteDoble()


@pytest.fixture
def publico(agente, monkeypatch):
    monkeypatch.setattr(MotorChat, "corpus_disponible", staticmethod(lambda: True))
    return MotorChat(H, modo="publico", agente=agente)


def test_sin_indice_avisa_y_no_consulta(agente, monkeypatch):
    monkeypatch.setattr(MotorChat, "corpus_disponible", staticmethod(lambda: False))
    r = MotorChat(H, agente=agente).responder("corpus ¿Qué escala usó el estudio?")
    assert r.ruta == "corpus" and "no está disponible" in r.texto
    assert agente.llamadas == []


def test_modo_invalido(agente):
    with pytest.raises(ValueError):
        MotorChat(H, modo="otro", agente=agente)


@pytest.mark.parametrize("pregunta,esperado", [
    ("¿Cuál es el nivel general de bienestar?", "50.7"),
    ("¿La brecha de posgrado se mantiene controlando la edad?", "+2.57"),
    ("¿Cuánto depende el bienestar del programa?", "9.8 %"),
    ("perfil por dimensiones", "Dominio del entorno"),
    ("¿se reproduce el estudio anterior? valida", "1 de 2"),
    ("compara el bienestar por estrato", "Estrato 3"),
])
def test_encuesta_responde_con_agregados(publico, agente, pregunta, esperado):
    r = publico.responder(pregunta)
    assert r.ruta == "encuesta"
    assert esperado in r.texto
    assert "agregados" in r.fuente
    assert not r.usa_servicio_externo
    assert agente.llamadas == []          # ni corpus, ni web, ni agente completo


def test_comparacion_en_positivo_y_resalta_el_grupo(publico):
    r = publico.responder("¿Hay diferencias entre hombres y mujeres?")
    assert r.ruta == "encuesta"
    assert "Masculino − Femenino: +1.15" in r.texto
    assert "No se muestran 4 categoría(s)" in r.texto
    r = publico.responder("¿Cómo es el bienestar en Tunja comparado con otras seccionales?")
    assert "**Tunja**" in r.texto
    assert "Ninguna diferencia" in r.texto


def test_dif_por_nivel_sin_nombrar_bienestar(publico):
    r = publico.responder("¿Hay DIF por nivel de formación?")
    assert r.ruta == "encuesta"
    assert "Posgrado vs. Pregrado" in r.texto


def test_publico_no_inventa_lo_que_no_esta_precalculado(publico, agente):
    r = publico.responder("¿La edad se relaciona con el bienestar?")
    assert r.ruta == "abstencion"
    assert "no está precalculado" in r.texto
    assert agente.llamadas == []


def test_completo_usa_el_agente_para_lo_no_precalculado(agente):
    motor = MotorChat(H, modo="completo", agente=agente)
    r = motor.responder("¿La edad se relaciona con el bienestar?")
    assert agente.llamadas == [("ask", "¿La edad se relaciona con el bienestar?")]
    assert r.preformateado and r.ruta == "encuesta"
    assert r.fuente == "03_dataset_IRT_ponderado.xlsx"


def test_fuera_de_dominio_se_abstiene(publico, agente):
    r = publico.responder("¿Quién ganó el mundial de 2022?")
    assert r.ruta == "abstencion"
    assert agente.llamadas == []


def test_vacia_se_abstiene(publico):
    assert publico.responder("   ").ruta == "abstencion"


def test_corpus_cita_documento_y_pagina(publico, agente):
    r = publico.responder("corpus ¿Qué escala usó el estudio de Bucaramanga?")
    assert agente.llamadas == [("corpus", "¿Qué escala usó el estudio de Bucaramanga?")]
    assert r.ruta == "corpus" and r.usa_servicio_externo
    assert "doc.pdf, p. 2" in r.texto


def test_busqueda_lista_articulos_con_doi(publico, agente):
    r = publico.responder("busca artículos sobre la escala de Ryff en estudiantes")
    assert r.ruta == "busqueda" and r.usa_servicio_externo
    assert "https://doi.org/10.1/x" in r.texto
    assert agente.llamadas[0][0] == "busqueda"
