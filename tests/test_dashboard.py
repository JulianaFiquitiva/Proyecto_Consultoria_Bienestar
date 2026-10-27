# -*- coding: utf-8 -*-
"""
La app (app/dashboard.py y sus páginas en app/paginas/) se ejecuta sin errores.

Usa los hallazgos precalculados (resultados/entrega2/hallazgos_entrega2.json)
o, si faltan, los datos de data/. Si no hay ninguno de los dos, se salta.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from agent.reports import entrega2

RAIZ = Path(__file__).resolve().parent.parent
APP = str(RAIZ / "app" / "dashboard.py")


def _hay_insumos() -> bool:
    if entrega2.SALIDA.exists():
        return True
    from tests.conftest import _hay_datos
    return _hay_datos()


pytestmark = pytest.mark.skipif(not _hay_insumos(),
                                reason="sin hallazgos precalculados ni data/")


PAGINAS = ["inicio", "estudio", "resultados", "comparaciones", "dimensiones",
           "hallazgos", "chat", "metodologia"]


def _abrir(pagina=None, timeout=300):
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    if pagina:
        at.switch_page(f"paginas/{pagina}.py").run()
    return at


@pytest.fixture
def publica(monkeypatch):
    """La app como se publica: solo el JSON de agregados."""
    if not entrega2.SALIDA.exists():
        pytest.skip("sin hallazgos precalculados")
    monkeypatch.setenv("INDICE_MODO", "publico")
    monkeypatch.delenv("APP_PASSWORD", raising=False)


@pytest.mark.parametrize("pagina", PAGINAS)
def test_cada_pagina_sin_excepciones(pagina):
    at = _abrir(pagina)
    assert not at.exception, [e.value for e in at.exception]


def test_inicio_muestra_metricas():
    at = _abrir()
    textos = " ".join(m.value for m in at.markdown)
    assert "Puntaje T poblacional" in textos and "En riesgo" in textos


def test_chat_tiene_caja_de_entrada(publica):
    at = _abrir("chat", timeout=120)
    assert len(at.chat_input) == 1


def test_chat_responde_con_agregados(publica):
    at = _abrir("chat", timeout=120)
    at.chat_input[0].set_value("¿La brecha de posgrado se mantiene controlando la edad?").run()
    assert not at.exception, [e.value for e in at.exception]
    textos = " ".join(m.value for m in at.markdown)
    assert "Brecha de bienestar posgrado" in textos
    assert any("Fuente: Resultados agregados" in c.value for c in at.caption)


def test_chat_ejemplo_con_boton(publica):
    at = _abrir("chat", timeout=120)
    at.button(key="ejemplo_0").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Fuente:" in c.value for c in at.caption)


def test_chat_limite_de_consultas_externas(publica, monkeypatch):
    monkeypatch.setenv("CHAT_MAX_CONSULTAS", "0")
    at = _abrir("chat", timeout=120)
    # Con límite 0 la pregunta al corpus NO llega a abrir el índice
    at.chat_input[0].set_value("corpus ¿Qué escala usó el estudio?").run()
    assert not at.exception
    assert any("límite de 0 consultas" in m.value for m in at.markdown)


def test_contrasena(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "clave-de-prueba")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert len(at.metric) == 0 and len(at.text_input) == 1    # solo la contraseña
    at.text_input[0].input("otra")
    at.button[0].click().run()
    assert at.error
    at.text_input[0].input("clave-de-prueba")
    at.button[0].click().run()
    assert not at.exception
    assert len(at.text_input) == 0


def test_no_muestra_grupos_pequenos():
    h = entrega2.cargar()
    if h is None:
        pytest.skip("sin hallazgos precalculados")
    for datos in h["comparaciones"].values():
        assert all(g["n"] >= h["min_n_grupo"] for g in datos["grupos"].values())


def test_grupos_traen_niveles_y_dimensiones():
    h = entrega2.cargar()
    if h is None:
        pytest.skip("sin hallazgos precalculados")
    for datos in h["comparaciones"].values():
        for g in datos["grupos"].values():
            assert {"riesgo", "alto", "dimensiones"} <= set(g)
            assert 0 <= g["riesgo"]["media"] <= 1 and 0 <= g["alto"]["media"] <= 1
