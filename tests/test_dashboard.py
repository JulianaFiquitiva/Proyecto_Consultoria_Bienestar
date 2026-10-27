# -*- coding: utf-8 -*-
"""
El dashboard (app/dashboard.py) se ejecuta sin errores.

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


PESTANAS = ["El estudio", "Resumen", "Comparaciones", "Dimensiones",
            "Hallazgos entrega 2", "Pregúntale al agente", "Metodología"]


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


@pytest.fixture
def app_publica(monkeypatch):
    """La app como se publicaría: solo el JSON de agregados."""
    if not entrega2.SALIDA.exists():
        pytest.skip("sin hallazgos precalculados")
    monkeypatch.setenv("INDICE_MODO", "publico")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_arranca_sin_excepciones(app):
    assert not app.exception, [e.value for e in app.exception]


def test_pestanas_y_metricas(app):
    assert [t.label for t in app.tabs] == PESTANAS
    assert len(app.metric) == 8           # 4 del estudio + 4 del resumen


def test_chat_responde_con_agregados(app_publica):
    app_publica.chat_input[0].set_value("¿La brecha de posgrado se mantiene controlando "
                                        "la edad?").run()
    assert not app_publica.exception, [e.value for e in app_publica.exception]
    textos = " ".join(m.value for m in app_publica.markdown)
    assert "Brecha de bienestar posgrado" in textos
    assert any("Fuente: Resultados agregados" in c.value for c in app_publica.caption)


def test_chat_limite_de_consultas_externas(app_publica, monkeypatch):
    monkeypatch.setenv("CHAT_MAX_CONSULTAS", "0")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    # Con límite 0 la pregunta al corpus NO llega a abrir el índice
    at.chat_input[0].set_value("corpus ¿Qué escala usó el estudio?").run()
    assert not at.exception
    assert any("límite de 0 consultas" in m.value for m in at.markdown)


def test_contrasena(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "clave-de-prueba")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert len(at.tabs) == 0              # nada visible sin la contraseña
    at.text_input[0].input("otra")
    at.button[0].click().run()
    assert len(at.tabs) == 0 and at.error
    at.text_input[0].input("clave-de-prueba")
    at.button[0].click().run()
    assert [t.label for t in at.tabs] == PESTANAS


def test_cada_variable_de_comparacion(app):
    caja = app.selectbox[0]
    for opcion in caja.options:
        caja.select(opcion).run()
        assert not app.exception, (opcion, [e.value for e in app.exception])


def test_no_muestra_grupos_pequenos():
    h = entrega2.cargar()
    if h is None:
        pytest.skip("sin hallazgos precalculados")
    for datos in h["comparaciones"].values():
        assert all(g["n"] >= h["min_n_grupo"] for g in datos["grupos"].values())
