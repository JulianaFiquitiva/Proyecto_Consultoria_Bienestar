# -*- coding: utf-8 -*-
"""El agente entiende sinónimos (agent/core/sinonimos.py) sin cambiar sus rutas."""
import pytest

from agent.core.orchestrator import StatisticalAgent
from agent.core.sinonimos import normalizar_pregunta


@pytest.fixture(scope="module")
def agente():
    return StatisticalAgent()


@pytest.mark.parametrize("pregunta, analisis, grupo", [
    ("¿Cómo está la felicidad de los estudiantes?", "descriptive_overview", None),
    ("¿Qué tan felices son los alumnos?", "descriptive_overview", None),
    ("¿Cómo está la salud mental de los universitarios?", "descriptive_overview", None),
    ("¿Cuál es la satisfacción con la vida de los estudiantes?", "descriptive_overview", None),
    ("¿Cuántos alumnos están en riesgo?", "descriptive_overview", None),
    ("¿Cuál es el puntaje promedio?", "descriptive_overview", None),
    ("¿Las chicas tienen más bienestar que los chicos?", "design_comparison", "Género"),
    ("¿Influye el género en el bienestar?", "design_comparison", "Género"),
    ("¿Qué campus tiene mejor bienestar?", "design_comparison", "seccional"),
    ("¿En qué ciudad están más felices?", "design_comparison", "seccional"),
    ("¿Hay brecha entre maestría y licenciatura?", "design_comparison", "modalidad_exp"),
    ("¿Hay diferencias por nivel de formación?", "design_comparison", "Nivel de formacion"),
    ("¿Hay diferencias por nivel educativo?", "design_comparison", "Nivel de formacion"),
    ("¿Influye la condición socioeconómica?", "design_comparison", "Estrato socioeconomico"),
    ("¿Los ricos son más felices que los pobres?", "design_comparison",
     "Estrato socioeconomico"),
    ("¿Cuál es la dimensión más baja?", "dimension_profile", None),
    ("¿Cuál es el aspecto más débil del bienestar?", "dimension_profile", None),
    ("¿Cómo está la autoestima?", "dimension_profile", None),
    ("¿Cómo está el sentido de vida?", "dimension_profile", None),
    ("¿Hay sesgo en los ítems?", "dif_analysis", None),
    ("¿La escala es justa para hombres y mujeres?", "dif_analysis", None),
    ("¿Los resultados coinciden con el informe original?", "validate_results", None),
    ("Dame un panorama general", "descriptive_overview", None),
])
def test_sinonimos_llevan_al_analisis(agente, pregunta, analisis, grupo):
    # Igual que el chat de la app (MotorChat.responder): si la pregunta no
    # nombra el bienestar, se interpreta en el contexto del estudio.
    assert agente.ruta_de_pregunta(pregunta) in ("encuesta", "abstencion")
    plan = agente._fallback_plan(pregunta)
    if plan.function_to_call == "abstencion":
        plan = agente._fallback_plan(f"bienestar {pregunta}")
    assert plan.function_to_call == analisis
    if grupo:
        assert plan.parameters["group_col"] == grupo


@pytest.mark.parametrize("pregunta, ruta", [
    ("¿Qué dicen los documentos sobre el bienestar psicológico?", "corpus"),
    ("Busca artículos sobre felicidad en universitarios", "busqueda"),
    ("¿Cuál es la capital de Francia?", "abstencion"),
    ("¿Qué relación hay entre las notas y la felicidad?", "abstencion"),
    ("¿Hay diferencias de bienestar en otros países?", "abstencion"),
])
def test_rutas_externas_y_abstencion_no_cambian(agente, pregunta, ruta):
    assert agente.ruta_de_pregunta(pregunta) == ruta


def test_normalizar_quita_tildes_y_traduce():
    assert normalizar_pregunta("¿Cuál es la DIMENSIÓN más débil?") == \
        "¿cual es la dimension mas baja?"
    assert "seccional" in normalizar_pregunta("¿Qué campus está mejor?")
