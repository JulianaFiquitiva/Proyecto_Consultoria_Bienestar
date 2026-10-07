#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corredor del banco de preguntas: medicion sin sistema vs con sistema.

Lee tests/banco_preguntas_literatura.json y corre cada pregunta de DOS formas:

  sin : solo el LLM de agent/api_client.py, sin contexto ni fuentes
  con : agent.rag.rag_engine.ask_with_sources

Guarda resultados/resultados_banco.csv con id, tipo, condicion, respuesta,
fuentes_citadas, abstuvo y tres columnas VACIAS para revision humana:
correcta, cita_correcta, observacion.

NO se calcula ningun acierto comparando texto. La revision es humana; lo unico
automatico es ejecutar el banco y ordenar el archivo.

Uso:
    python tests/correr_banco_rag.py
    python tests/correr_banco_rag.py --completar   # solo filas sin respuesta
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

BANCO = RAIZ / "tests" / "banco_preguntas_literatura.json"
SALIDA = RAIZ / "resultados" / "resultados_banco.csv"

CAMPOS = [
    "id", "tipo", "condicion", "respuesta", "fuentes_citadas", "abstuvo",
    "correcta", "cita_correcta", "observacion",
]
ORDEN_CONDS = ("sin", "con")

# Columnas de revision humana: siempre vacias en la salida automatica.
COLUMNAS_HUMANAS = ("correcta", "cita_correcta", "observacion")

# gpt-4 limita 10000 tokens/min; un fragmento RAG pesa ~5000, asi que el
# banco choca con el limite. Se espera 65 s y se reintenta (max 3 veces).
ESPERA_429 = 65
REINTENTOS = 3


def cargar_banco() -> List[Dict]:
    with open(BANCO, encoding="utf-8") as fh:
        datos = json.load(fh)
    if not isinstance(datos, list) or not datos:
        raise SystemExit("El banco esta vacio o no es una lista: %s" % BANCO)
    requeridos = {"id", "pregunta", "tipo"}
    faltantes = requeridos - set(datos[0])
    if faltantes:
        raise SystemExit("Faltan campos en el banco: %s" % sorted(faltantes))
    return datos


def _es_rate_limit(error: str) -> bool:
    bajo = (error or "").lower()
    return ("429" in bajo) or ("rate limit" in bajo)


def _respuestas_fuentes(fuentes) -> str:
    if not fuentes:
        return ""
    return " | ".join("%s, p. %s" % (getattr(f, "archivo", ""),
                                     getattr(f, "pagina", ""))
                      for f in fuentes)


def _con_reintento(hacer: Callable[[bool], Dict]) -> Dict:
    """hacer(registrar) -> registro. Reintenta solo ante 429 (TPM)."""
    registro = None
    for intento in range(REINTENTOS):
        registro = hacer(intento == 0)   # la bitacora RAG solo en el 1er intento
        if not _es_rate_limit(registro["error"]):
            return registro
        if intento < REINTENTOS - 1:
            print("      429 rate limit: esperando %d s (intento %d/%d)"
                  % (ESPERA_429, intento + 2, REINTENTOS))
            time.sleep(ESPERA_429)
    return registro


def correr_sin_sistema(llm, pregunta: str, es_abstencion) -> Dict:
    """Solo el LLM, sin contexto ni fuentes."""
    def _hacer(_registrar: bool) -> Dict:
        registro = {"respuesta": "", "fuentes_citadas": "",
                    "abstuvo": "no", "error": "", "duracion": 0.0}
        if llm is None or not getattr(llm, "is_configured", False):
            registro["error"] = "sin OPENAI_API_KEY"
            registro["abstuvo"] = "sí"
            return registro
        t0 = time.time()
        try:
            salida = llm.chat([{"role": "user", "content": pregunta}])
            if not salida.success:
                raise RuntimeError(salida.error or "fallo la llamada al LLM")
            texto = (salida.content or "").strip()
            registro["respuesta"] = texto
            registro["abstuvo"] = "sí" if es_abstencion(texto) else "no"
        except Exception as exc:  # noqa: BLE001 - el corredor no debe parar
            registro["error"] = str(exc)
            registro["abstuvo"] = "sí"
        registro["duracion"] = round(time.time() - t0, 1)
        return registro
    return _con_reintento(_hacer)


def correr_con_sistema(pregunta: str, ask, registrar_cb=None) -> Dict:
    """RAG completo: recuperacion + umbral + cita obligatoria."""
    def _hacer(registrar: bool) -> Dict:
        registro = {"respuesta": "", "fuentes_citadas": "",
                    "abstuvo": "no", "error": "", "duracion": 0.0}
        t0 = time.time()
        try:
            resp = ask(pregunta, registrar)
            registro["respuesta"] = resp.respuesta or ""
            registro["fuentes_citadas"] = _respuestas_fuentes(resp.fuentes)
            registro["abstuvo"] = "sí" if resp.abstencion else "no"
            registro["error"] = resp.error or ""
            registro["_resp"] = resp
        except Exception as exc:  # noqa: BLE001
            registro["error"] = str(exc)
            registro["abstuvo"] = "sí"
        registro["duracion"] = round(time.time() - t0, 1)
        return registro

    registro = _con_reintento(_hacer)
    resp = registro.pop("_resp", None)
    # Una sola fila en logs/rag_consultas.csv: solo si la consulta termino.
    if resp is not None and not registro["error"] and registrar_cb:
        registrar_cb(resp)
    return registro


def ordenar(fila: Dict) -> Tuple[int, int]:
    """Lo automatico solo ordena: id, y dentro de cada id sin -> con."""
    try:
        num_id = int(fila["id"])
    except (TypeError, ValueError):
        num_id = 10 ** 6
    try:
        pos = ORDEN_CONDS.index(fila["condicion"])
    except ValueError:
        pos = len(ORDEN_CONDS)
    return (num_id, pos)


def _cargar_existentes() -> Dict[Tuple[int, str], Dict]:
    if not SALIDA.exists():
        raise SystemExit("--completar: no existe %s" % SALIDA)
    existentes: Dict[Tuple[int, str], Dict] = {}
    with open(SALIDA, encoding="utf-8-sig", newline="") as fh:
        for fila in csv.DictReader(fh):
            existentes[(int(fila["id"]), fila["condicion"])] = fila
    return existentes


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--desde", type=int, default=None,
                        help="solo ids >= a este valor")
    parser.add_argument("--hasta", type=int, default=None,
                        help="solo ids <= a este valor")
    parser.add_argument("--sin-llm", action="store_true",
                        help="no llama al LLM: llena respuestas vacias")
    parser.add_argument("--completar", action="store_true",
                        help="conserva el CSV y solo reejecuta filas sin respuesta")
    args = parser.parse_args(argv)

    banco = cargar_banco()
    filas: List[Dict] = []
    existentes: Dict[Tuple[int, str], Dict] = {}
    if args.completar:
        existentes = _cargar_existentes()
        pend = sum(1 for v in existentes.values() if not v["respuesta"])
        print("CSV existente: %d filas, %d sin respuesta -> a reejecutar"
              % (len(existentes), pend))

    client = None
    ask = None
    registrar_cb = None
    es_abstencion = None
    if not args.sin_llm:
        from agent.api_client import create_client
        from agent.rag import rag_engine as _rag
        from agent.rag.retriever import Retriever

        client = create_client("openai")
        retriever = Retriever()          # un solo cargado para las 19

        def _preguntar(pregunta: str, registrar: bool):
            return _rag.ask_with_sources(pregunta, retriever=retriever,
                                          registrar=registrar)

        def _registrar(resp):
            _rag.registrar_consulta(resp.backend, resp.pregunta, resp.fragmentos,
                                    resp.respuesta, resp.abstencion)

        ask = _preguntar
        registrar_cb = _registrar
        es_abstencion = _rag._es_abstencion_llm

        print("backend RAG : %s" % retriever.backend)
        print("LLM         : %s" % ("openai" if client.is_configured
                                    else "SIN API KEY"))
        print("-" * 74)

    errores = 0
    for item in banco:
        num_id = item["id"]
        if args.desde is not None and num_id < args.desde:
            continue
        if args.hasta is not None and num_id > args.hasta:
            continue
        pendientes = [c for c in ORDEN_CONDS
                      if not args.completar
                      or (int(num_id), c) not in existentes
                      or not existentes[(int(num_id), c)]["respuesta"]]
        if not pendientes:
            continue

        pregunta = item["pregunta"]
        tipo = item["tipo"]
        print("[%s] %s" % (num_id, pregunta[:78]))

        for condicion in pendientes:
            if args.sin_llm:
                reg = {"respuesta": "", "fuentes_citadas": "",
                       "abstuvo": "no", "error": "", "duracion": 0.0}
            elif condicion == "sin":
                reg = correr_sin_sistema(client, pregunta, es_abstencion)
            else:
                reg = correr_con_sistema(pregunta, ask, registrar_cb)

            if reg["error"]:
                errores += 1

            filas.append({
                "id": num_id,
                "tipo": tipo,
                "condicion": condicion,
                "respuesta": reg["respuesta"],
                "fuentes_citadas": reg["fuentes_citadas"],
                "abstuvo": reg["abstuvo"],
                "correcta": "",
                "cita_correcta": "",
                "observacion": "",
            })
            print("    %-3s abstuvo=%-3s  %ss  %s"
                  % (condicion, reg["abstuvo"], reg.get("duracion", 0),
                     ("ERROR: " + reg["error"]) if reg["error"] else ""))

    if existentes:
        base = {k: dict(v) for k, v in existentes.items()}
        for fila in filas:
            base[(int(fila["id"]), fila["condicion"])] = fila
        filas = list(base.values())

    filas.sort(key=ordenar)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel abre los acentos correctos al hacer doble clic.
    with open(SALIDA, "w", newline="", encoding="utf-8-sig") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
        escritor.writeheader()
        for fila in filas:
            if any(fila[c] for c in COLUMNAS_HUMANAS):
                raise SystemExit("Las columnas de revision deben ir vacias.")
            escritor.writerow(fila)

    pendientes = sum(1 for f in filas if not f["correcta"])
    print("-" * 74)
    print("filas escritas        : %d  ->  %s" % (len(filas), SALIDA))
    print("columnas de revision  : correcta, cita_correcta, observacion "
          "(vacias)")
    print("llamadas con error    : %d" % errores)
    print("FILAS PENDIENTES DE REVISION HUMANA: %d" % pendientes)
    print("(la revision es humana: no se calculo ningun acierto)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
