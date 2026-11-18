# -*- coding: utf-8 -*-
"""
Actualiza los resultados de la encuesta viva.

    python -m agent.encuesta_viva.actualizar --fuente <URL o archivo>

La fuente es la hoja de respuestas del formulario:
- Google Forms: en la hoja de respuestas, Archivo > Compartir > Publicar en
  la web > CSV. Ese enlace va en el secreto ENCUESTA_FUENTE de GitHub.
- Microsoft Forms / Excel: un enlace de descarga directa (.xlsx) o el archivo.

Sin --fuente se lee la variable de entorno ENCUESTA_FUENTE. El enlace nunca
se escribe en la salida ni en el log.

Escribe datos_vivos/hallazgos.json (solo agregados; la app lo lee en la
página «Encuesta en vivo»). Si el resultado no cambió, no reescribe el
archivo, para que la GitHub Action no haga commits vacíos.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agent.core.config import PROJECT_ROOT
from agent.encuesta_viva import pipeline as P

SALIDA = PROJECT_ROOT / "datos_vivos" / "hallazgos.json"
MIN_RESPUESTAS = 30       # por debajo, la app muestra los resultados con aviso


def leer_fuente(fuente: str) -> pd.DataFrame:
    """Lee CSV o Excel desde un archivo local o una URL."""
    if fuente.startswith(("http://", "https://")):
        req = urllib.request.Request(fuente, headers={"User-Agent": "encuesta-viva/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            datos = r.read()
    else:
        datos = Path(fuente).read_bytes()
    if datos[:2] == b"PK":                         # .xlsx es un zip
        return pd.read_excel(io.BytesIO(datos))
    return pd.read_csv(io.BytesIO(datos), encoding="utf-8-sig")


def _huella(h: dict) -> str:
    """Huella del contenido sin la fecha, para saber si algo cambió."""
    copia = {k: v for k, v in h.items() if k not in ("generado", "encuesta_viva")}
    copia["n"] = h.get("encuesta_viva", {}).get("respuestas_recibidas")
    texto = json.dumps(copia, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(texto.encode()).hexdigest()


def actualizar(fuente: str, salida: Path = SALIDA, incluir_dif: bool = False) -> dict:
    from agent.reports import entrega2

    crudo = leer_fuente(fuente)
    raw = P.normalizar_formulario(crudo)
    sin_nbc = P.programas_sin_nbc(raw)
    prep, irt, h = P.correr(raw, incluir_dif=incluir_dif)
    h = json.loads(json.dumps(h, ensure_ascii=False, default=entrega2._json_default))
    h.pop("reproducibilidad", None)   # compara con el estudio de 2025: no aplica aquí
    h["encuesta_viva"] = {
        "actualizado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "respuestas_recibidas": int(len(crudo)),
        "respuestas_validas": int(len(prep)),
        "con_puntaje": int(len(irt)),
        "pocas_respuestas": bool(len(irt) < MIN_RESPUESTAS),
        "programas_sin_nbc": sin_nbc,
        "metodo_puntaje": "θ EAP con los parámetros GRM fijos del estudio (T = 50 + 10·θ)",
    }
    h["huella"] = _huella(h)

    if salida.exists():
        anterior = json.loads(salida.read_text(encoding="utf-8"))
        if anterior.get("huella") == h["huella"]:
            h["sin_cambios"] = True
            return h
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
    return h


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fuente", default=os.environ.get("ENCUESTA_FUENTE", ""),
                    help="URL o archivo con las respuestas (por defecto ENCUESTA_FUENTE)")
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--dif", action="store_true", help="incluir el análisis DIF (más lento)")
    a = ap.parse_args(argv)
    if not a.fuente:
        print("Falta la fuente: use --fuente o la variable ENCUESTA_FUENTE.")
        return 2
    try:
        h = actualizar(a.fuente, a.salida, a.dif)
    except P.FormularioInvalido as e:
        print(f"El formulario no tiene la estructura esperada: {e}")
        return 1
    v = h["encuesta_viva"]
    estado = "sin cambios" if h.get("sin_cambios") else f"escrito en {a.salida}"
    print(f"Respuestas: {v['respuestas_recibidas']} recibidas, {v['respuestas_validas']} válidas, "
          f"{v['con_puntaje']} con puntaje. Resultado {estado}.")
    if v["programas_sin_nbc"]:
        print("Programas sin NBC (agregar a catalogos.py):", ", ".join(v["programas_sin_nbc"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
