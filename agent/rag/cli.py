# -*- coding: utf-8 -*-
"""
CLI del modulo RAG (Fase B).

  python -m agent.rag.cli "pregunta"
  python -m agent.rag.cli --calibrar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.rag import rag_engine
from agent.rag.rag_engine import RespuestaRAG, ask_with_sources, calibrar, umbral
from agent.rag.retriever import Retriever


def _imprimir_fragmentos_crudos(resp: RespuestaRAG) -> None:
    print("\n--- FRAGMENTOS CRUDOS RECUPERADOS (k=5) ---")
    if not resp.fragmentos:
        print("  (ninguno: el indice esta vacio)")
    for i, h in enumerate(resp.fragmentos, 1):
        pag = ",".join(str(p) for p in h.paginas) or "?"
        print(f"\n  [{i}] archivo={h.archivo}  pagina={pag}  "
              f"puntaje={h.puntaje:.4f}")
        print(f"      {h.texto}")


def _imprimir(resp: RespuestaRAG) -> None:
    print("=" * 78)
    print(f"PREGUNTA: {resp.pregunta}")
    print("=" * 78)
    print("\n--- RESPUESTA ---")
    print(resp.respuesta)
    if resp.error:
        print(f"[aviso] {resp.error}")

    print("\n--- FUENTES CITADAS ---")
    if not resp.fuentes:
        print("  (ninguna: hubo abstinence)")
    for i, f in enumerate(resp.fuentes, 1):
        print(f"  {i}. {f.archivo}, p. {f.pagina}  (puntaje {f.puntaje:.4f})")

    _imprimir_fragmentos_crudos(resp)

    print("\n--- TRAZABILIDAD ---")
    print(f"  abstinencia     : {'SI' if resp.abstencion else 'no'}")
    print(f"  puntaje maximo  : {resp.puntaje_max:.4f}  (umbral {umbral():.4f})")
    print(f"  llamada al LLM  : {'si' if resp.llamada_llm else 'NO'}")
    print(f"  backend         : {resp.backend}")
    print(f"  log             : {rag_engine.LOG_CSV}")


def _calibrar() -> int:
    print("Calibrando el umbral con las 6 preguntas de calibracion "
          "(sin el banco, sin LLM)...\n")
    filas = calibrar()
    print(f"{'TIPO':<16} {'PUNTAJE':>8}  {'ARCHIVO':<40} P.PDF")
    print("-" * 96)
    for f in filas:
        print(f"{f['tipo']:<16} {f['puntaje_max']:>8.4f}  "
              f"{f['archivo_top'][:40]:<40} {f['pagina_top']}")

    con = [f["puntaje_max"] for f in filas if f["tipo"] == "con_respuesta"]
    sin = [f["puntaje_max"] for f in filas if f["tipo"] == "sin_respuesta"]
    print()
    if con and sin:
        print(f"min(con_respuesta) = {min(con):.4f}")
        print(f"max(sin_respuesta) = {max(sin):.4f}")
        if min(con) > max(sin):
            punto = (min(con) + max(sin)) / 2
            print(f"separacion limpia -> umbral sugerido = {punto:.4f}")
        else:
            print("LOS INTERVALOS SE SOLAPAN: hay que mirar caso a caso")
    print(f"\nguardado en {rag_engine.CALIBRACION_CSV}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.rag.cli", description="RAG con cita")
    ap.add_argument("pregunta", nargs="*", help="pregunta a formular al corpus")
    ap.add_argument("--calibrar", action="store_true",
                    help="calibra el umbral con las 6 preguntas de apoyo")
    ap.add_argument("--umbral", type=float, default=None,
                    help="sobreescribe el umbral solo para esta consulta")
    ap.add_argument("--sin-fragmentos", action="store_true",
                    help="no imprime los fragmentos crudos")
    args = ap.parse_args(argv)

    if args.calibrar:
        return _calibrar()

    pregunta = " ".join(args.pregunta).strip()
    if not pregunta:
        ap.print_help()
        return 2

    resp = ask_with_sources(pregunta, umbral_valor=args.umbral)
    if args.sin_fragmentos:
        print("=" * 78)
        print(f"PREGUNTA: {resp.pregunta}")
        print("=" * 78)
        print("\n--- RESPUESTA ---")
        print(resp.respuesta)
    else:
        _imprimir(resp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
