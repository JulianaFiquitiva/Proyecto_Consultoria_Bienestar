# -*- coding: utf-8 -*-
"""
Rutas de datos portables: el agente encuentra data/ desde la carpeta del
proyecto, sin depender del equipo ni del directorio de ejecucion.
"""
from pathlib import Path

from agent.core.config import PROJECT_ROOT, AgentConfig

RAIZ = Path(__file__).resolve().parent.parent


def test_la_raiz_es_la_carpeta_del_proyecto():
    assert PROJECT_ROOT == RAIZ
    assert (PROJECT_ROOT / "agent" / "core" / "config.py").exists()


def test_no_depende_del_directorio_de_ejecucion(tmp_path, monkeypatch):
    monkeypatch.delenv("INDICE_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    ruta = AgentConfig().get_data_path("01_dataset_preparado.xlsx")
    assert ruta == RAIZ / "data" / "01_dataset_preparado.xlsx"


def test_no_queda_ninguna_ruta_absoluta_de_un_equipo():
    texto = (RAIZ / "agent" / "core" / "config.py").read_text(encoding="utf-8")
    assert "C:\\Users" not in texto and "C:/Users" not in texto


def test_indice_data_dir_tiene_prioridad(tmp_path, monkeypatch):
    (tmp_path / "Parametros_GRM.xlsx").write_bytes(b"")
    monkeypatch.setenv("INDICE_DATA_DIR", str(tmp_path))
    ruta = AgentConfig().get_data_path("Parametros_GRM.xlsx")
    assert ruta == tmp_path / "Parametros_GRM.xlsx"


def test_indice_data_dir_sin_el_archivo_cae_en_data(tmp_path, monkeypatch):
    monkeypatch.setenv("INDICE_DATA_DIR", str(tmp_path))
    ruta = AgentConfig().get_data_path("Parametros_GRM.xlsx")
    assert ruta == RAIZ / "data" / "Parametros_GRM.xlsx"
