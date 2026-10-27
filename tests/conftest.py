# -*- coding: utf-8 -*-
"""
Configuracion comun de pytest.

- Sin `-m red`, la clave de OpenAI se oculta durante toda la sesion: una
  corrida normal nunca gasta llamadas pagadas, aunque la clave este en el
  entorno.
- Las pruebas marcadas `datos` se saltan si faltan los Excel de la
  encuesta (no se versionan: son de la contraparte).
- Red de seguridad: si una prueba modifica los datos, el log de consultas,
  la base vectorial o los resultados, la sesion falla y dice que archivo.
"""
import hashlib
import os

import pytest

from agent.core.config import PROJECT_ROOT, AgentConfig

_CLAVES_LLM = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")

# Archivos que ninguna prueba puede modificar
_PROTEGIDOS = ("data", "rag_store", "agent/rag/logs", "resultados")
_huellas_inicio = {}


def _huellas() -> dict:
    huellas = {}
    for carpeta in _PROTEGIDOS:
        raiz = PROJECT_ROOT / carpeta
        if not raiz.is_dir():
            continue
        for ruta in raiz.rglob("*"):
            if ruta.is_file():
                h = hashlib.sha256()
                with open(ruta, "rb") as fh:
                    for bloque in iter(lambda: fh.read(1 << 20), b""):
                        h.update(bloque)
                huellas[ruta.relative_to(PROJECT_ROOT).as_posix()] = h.hexdigest()
    return huellas


def pytest_sessionstart(session):
    _huellas_inicio.update(_huellas())


def pytest_sessionfinish(session, exitstatus):
    final = _huellas()
    cambiados = sorted(
        ruta for ruta in set(_huellas_inicio) | set(final)
        if _huellas_inicio.get(ruta) != final.get(ruta)
    )
    if cambiados:
        print("\n\nARCHIVOS PROTEGIDOS MODIFICADOS POR LAS PRUEBAS:")
        for ruta in cambiados:
            print(f"  {ruta}")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config):
    expresion = (config.getoption("markexpr") or "").replace(" ", "")
    if "notred" in expresion:
        for clave in _CLAVES_LLM:
            os.environ.pop(clave, None)


def _hay_datos() -> bool:
    config = AgentConfig()
    return all(
        config.get_data_path(f).exists()
        for f in (config.prepared_data_file, config.irt_data_file)
    )


def pytest_collection_modifyitems(config, items):
    if _hay_datos():
        return
    saltar = pytest.mark.skip(
        reason="faltan los Excel de la encuesta en data/ (no se versionan)")
    for item in items:
        if "datos" in item.keywords:
            item.add_marker(saltar)
