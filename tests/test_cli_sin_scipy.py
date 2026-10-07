# -*- coding: utf-8 -*-
"""
La CLI y el agente tienen que arrancar aunque Smart App Control bloquee un
DLL de scipy.

scipy aqui solo sostiene dos tests auxiliares de supuestos (Shapiro-Wilk y
correlacion), no el arranque: si su import se dejara caer en la linea 16 del
orquestador, `python -m agent.ui.cli` moriria antes de poder hacer ni una
consulta al corpus.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent

# Bloquea scipy ANTES de que el agente lo importe, igual que SAC (que falla
# con "Una directiva de Control de aplicaciones bloqueo este archivo").
SCRIPT_BLOQUEO = """
import sys


class BloqueoScipy:
    def find_spec(self, name, path=None, target=None):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError(
                "DLL load failed while importing _tools: Una directiva de "
                "Control de aplicaciones bloqueo este archivo")
        return None


sys.meta_path.insert(0, BloqueoScipy())

import agent.ui.cli                                   # noqa: F401
print("ARRANCA_SIN_SCIPY")

from agent.core.orchestrator import SCIPY_ERROR, sp_stats
print("SCIPY_ERROR_REGISTRADO=", SCIPY_ERROR is not None)
print("SCIPY_ERROR=", SCIPY_ERROR)
print("SP_STATS_ES_NONE=", sp_stats is None)
"""


def _ejecutar(script: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RAIZ) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(RAIZ), env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=300,
    )


def test_la_cli_arranca_aunque_scipy_este_bloqueado():
    proc = _ejecutar(SCRIPT_BLOQUEO)
    salida = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, salida
    assert "ARRANCA_SIN_SCIPY" in proc.stdout, salida
    assert "SCIPY_ERROR_REGISTRADO= True" in proc.stdout, salida
    assert "SP_STATS_ES_NONE= True" in proc.stdout, salida
    assert "Control de aplicaciones" in proc.stdout, salida


def test_scipy_cargado_no_deja_error():
    """Si scipy SI carga, SCIPY_ERROR tiene que ser None (no un falso positivo)."""
    proc = _ejecutar("from agent.core.orchestrator import SCIPY_ERROR\n"
                     "print('ERROR=', SCIPY_ERROR)")
    assert proc.returncode == 0, proc.stderr
    assert "ERROR= None" in proc.stdout, proc.stdout
