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


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def test_arranca_sin_excepciones(app):
    assert not app.exception, [e.value for e in app.exception]


def test_cinco_pestanas_y_metricas(app):
    assert [t.label for t in app.tabs] == [
        "Resumen", "Comparaciones", "Dimensiones", "Hallazgos entrega 2", "Metodología"]
    assert len(app.metric) == 4


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
