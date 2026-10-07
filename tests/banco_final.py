#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Salidas del banco FINAL: hoja ciega, clave, metadatos y diagnostico.

Fase A (protocolo igual al de la corrida anterior):
    resultados/hoja_revision_ciega_<ts>.csv  -> hoja SIN la columna condicion,
        barajada con semilla 20261004, columnas de revision vacias.
    resultados/clave_condicion_<ts>.csv       -> fila,id,condicion (no se abre).
    resultados/metadatos_<ts>.json            -> commit, backend, umbral, k,
        temperatura, sha256+nombre de los PDF indexados, hash del prompt,
        sha256 del banco y del CSV, y la ruta del orquestador por pregunta.

Fase B (SOLO LECTURA: no llama al LLM ni toca indice, umbral ni banco):
    resultados/diagnostico_final_<ts>.txt     -> por pregunta con respuesta:
        archivo y pagina esperados, si estuvo en el top 5 y en que posicion,
        puntaje del primero y recall@1/@5/@10; para las de abstencion, el
        puntaje del primer fragmento frente al umbral.

Uso:
    python tests/banco_final.py
    python tests/banco_final.py --csv resultados/banco_final_20261006_120000.csv
    python tests/banco_final.py --solo-diagnostico
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

RES = RAIZ / "resultados"
BANCO = RAIZ / "tests" / "banco_preguntas_final.json"
CORPUS_DIR = RAIZ / "docs" / "rag_corpus"

SEMILLA_HOJA = 20261004

CAMPOS_HOJA = [
    "fila", "pregunta", "respuesta_esperada", "documento_esperado",
    "pagina_pdf_esperada", "cita_textual_esperada", "respuesta",
    "fuentes_citadas", "correcta", "cita_correcta", "observacion",
]

# Columnas que el revisor llena a mano. La hoja sale con ellas vacias.
COLUMNAS_REVISION = ("correcta", "cita_correcta", "observacion")


# ── utilidades ───────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def leer_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def ts_del_csv(path: Path) -> str:
    """20261006_120000 de .../banco_final_20261006_120000.csv; si no, ahora."""
    stem = path.stem
    if "_" in stem:
        posible = stem.rsplit("_", 2)
        if len(posible) == 3:
            cand = "%s_%s" % (posible[1], posible[2])
            if len(cand) == 15 and cand.replace("_", "").isdigit():
                return cand
    return time.strftime("%Y%m%d_%H%M%S")


def csv_mas_reciente() -> Optional[Path]:
    candidatos = sorted(RES.glob("banco_final_*.csv"),
                        key=lambda p: p.stat().st_mtime)
    return candidatos[-1] if candidatos else None


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=RAIZ, capture_output=True,
                           text=True, timeout=30)
    except Exception:                                     # noqa: BLE001
        return ""
    return (r.stdout or "").strip()


def archivos_indexados() -> Tuple[List[str], Dict[str, str]]:
    """Los PDF que entran en el indice: orden alfabetico + 1a copia por sha256.

    Es exactamente la regla de agent.rag.loader.load_corpus, pero sin parsear
    los PDF: aqui solo hacen falta los nombres y las huellas.
    """
    vistos: Dict[str, str] = {}
    nombres: List[str] = []
    huellas: Dict[str, str] = {}
    for pdf in sorted(CORPUS_DIR.glob("*.pdf")):
        digest = sha256_file(pdf)
        if digest in vistos:
            continue
        vistos[digest] = pdf.name
        nombres.append(pdf.name)
        huellas[pdf.name] = digest
    return nombres, huellas


def ruta_orquestador(pregunta: str, agente, clasificar) -> str:
    """Que ruta tomaria _generate_plan. Determinista: NO llama al LLM."""
    from agent.core.orchestrator import _RE_PREGUNTA_CORPUS

    if _RE_PREGUNTA_CORPUS.search(pregunta):
        return "corpus"
    if agente._is_search_request(pregunta):
        return "busqueda"
    scope, _motivo = clasificar(pregunta)
    if scope != "en_dominio":
        return "abstencion(%s)" % scope
    return "encuesta"          # aqui SI se llamaria al LLM


# ── fase A: salidas del protocolo ────────────────────────────────────────

def fase_a(banco, banco_path: Path, filas, ts: str, csv_path: Path,
           rutas: Dict[str, str]) -> Dict:
    if len(filas) != 2 * len(banco):
        raise SystemExit("El CSV tiene %d filas y se esperaban %d (2 por "
                         "pregunta)." % (len(filas), 2 * len(banco)))

    por_cond: Dict[str, Dict[int, Dict[str, str]]] = {"sin": {}, "con": {}}
    for f in filas:
        por_cond.setdefault(f["condicion"], {})[int(f["id"])] = f
    for condicion in ("sin", "con"):
        faltantes = [i["id"] for i in banco
                     if int(i["id"]) not in por_cond.get(condicion, {})]
        if faltantes:
            raise SystemExit("Faltan filas de condicion '%s' para los ids: %s"
                             % (condicion, faltantes))

    hoja: List[Dict[str, str]] = []
    for item in banco:
        for condicion in ("sin", "con"):
            f = por_cond[condicion][int(item["id"])]
            hoja.append({
                "pregunta": item["pregunta"],
                "respuesta_esperada": item.get("respuesta_esperada", ""),
                "documento_esperado": item.get("documento", ""),
                "pagina_pdf_esperada": item.get("pagina_pdf", ""),
                "cita_textual_esperada": item.get("cita_textual", ""),
                "respuesta": f["respuesta"],
                "fuentes_citadas": f["fuentes_citadas"],
                "correcta": "", "cita_correcta": "", "observacion": "",
                "_condicion": condicion, "_id": item["id"],
            })

    random.Random(SEMILLA_HOJA).shuffle(hoja)
    for i, fila in enumerate(hoja, 1):
        fila["fila"] = str(i)

    hoja_path = RES / ("hoja_revision_ciega_%s.csv" % ts)
    clave_path = RES / ("clave_condicion_%s.csv" % ts)
    with open(hoja_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_HOJA, extrasaction="ignore")
        w.writeheader()
        for fila in hoja:
            w.writerow(fila)
    with open(clave_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["fila", "id", "condicion"])
        w.writeheader()
        for fila in hoja:
            w.writerow({"fila": fila["fila"], "id": fila["_id"],
                        "condicion": fila["_condicion"]})

    nombres, huellas = archivos_indexados()
    from agent.api_client import LLMConfig
    from agent.rag import rag_engine as _rag
    from agent.rag.retriever import Retriever

    cfg = LLMConfig()
    retriever = Retriever()
    try:
        umbral_actual = _rag.umbral()
    except Exception:                                      # noqa: BLE001
        umbral_actual = _rag.UMBRAL_DEFECTO

    prompt_path = RAIZ / "prompts" / "rag_v1.txt"
    metadatos = {
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corrida": "banco_final",
        "condiciones": ["sin", "con"],
        "banco": banco_path.relative_to(RAIZ).as_posix(),
        "banco_preguntas": len(banco),
        "banco_sha256": sha256_file(banco_path),
        "csv_origen": csv_path.name,
        "csv_sha256": sha256_file(csv_path),
        "filas_csv": len(filas),
        "modelo_llm": cfg.model,
        "temperatura": cfg.temperature,
        "max_tokens_llm": cfg.max_tokens,
        "timeout_llm": cfg.timeout,
        "max_retries_llm": cfg.max_retries,
        "backend_embeddings": retriever.backend,
        "dim_vector": retriever.store.dimension,
        "fragmentos_indexados": retriever.store.count(),
        "umbral": umbral_actual,
        "umbral_defecto": _rag.UMBRAL_DEFECTO,
        "k": _rag.K,
        "MAX_TOKENS_RAG": _rag.MAX_TOKENS_RAG,
        "PRESUPUESTO_PROMPT_CHARS": _rag.PRESUPUESTO_PROMPT_CHARS,
        "commit": git("rev-parse", "HEAD"),
        "git_agent_prompts_sucio": bool(git(
            "status", "--porcelain", "--", "agent", "prompts",
            ":!agent/rag/logs")),
        "git_logs_rag_consultas_sucio": bool(git(
            "status", "--porcelain", "--", "agent/rag/logs")),
        "sha256_prompts_rag_v1": sha256_file(prompt_path),
        "semilla_orden_hoja": SEMILLA_HOJA,
        "corpus_dir": CORPUS_DIR.relative_to(RAIZ).as_posix(),
        "pdf_en_corpus": len(list(CORPUS_DIR.glob("*.pdf"))),
        "pdf_indexados_n": len(nombres),
        "pdf_indexados": [{"archivo": n, "sha256": huellas[n]}
                          for n in nombres],
        "ruta_orquestador": rutas,
        "ruta_orquestador_nota": (
            "La corrida NO pasa por el orquestador (mismo protocolo que la "
            "corrida anterior: ask_with_sources). Esta columna registra la "
            "ruta que elegiria _generate_plan; para 'encuesta' el plan SI "
            "requeria LLM y aqui no se llamó."),
        "hoja_revision_ciega": hoja_path.name,
        "clave_condicion": clave_path.name,
    }
    meta_path = RES / ("metadatos_%s.json" % ts)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadatos, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return {"hoja": hoja_path, "clave": clave_path, "meta": meta_path,
            "n_hoja": len(hoja), "metadatos": metadatos}


# ── fase B: diagnostico de recuperacion (solo lectura) ───────────────────

def paginas_esperadas(item: Dict) -> List[int]:
    paginas: List[int] = []
    for crudo in (item.get("pagina_pdf"), item.get("paginas_alternativas")):
        for trozo in str(crudo or "").replace("|", ",").split(","):
            trozo = trozo.strip()
            if trozo.isdigit():
                paginas.append(int(trozo))
    return paginas


def es_abstuvo(valor: str) -> bool:
    return (valor or "").strip().lower().startswith("sí") or \
           (valor or "").strip().lower() == "si"


def fase_b(banco, banco_path: Path, filas, ts: str, csv_path: Path,
           rutas: Dict[str, str]) -> Path:
    from agent.rag import rag_engine as _rag
    from agent.rag.loader import resolve_document
    from agent.rag.retriever import Retriever

    retriever = Retriever()
    try:
        umbral = _rag.umbral()
    except Exception:                                      # noqa: BLE001
        umbral = _rag.UMBRAL_DEFECTO

    nombres, _huellas = archivos_indexados()
    con = {int(f["id"]): f for f in filas if f["condicion"] == "con"}

    lineas: List[str] = []
    add = lineas.append
    add("DIAGNOSTICO DE RECUPERACION - SOLO LECTURA")
    add("=" * 78)
    add("fecha                : %s" % time.strftime("%Y-%m-%dT%H:%M:%S"))
    add("banco                : %s" % banco_path.name)
    add("banco_sha256         : %s" % sha256_file(banco_path))
    add("csv                  : %s" % csv_path.name)
    add("csv_sha256           : %s" % sha256_file(csv_path))
    add("commit               : %s" % git("rev-parse", "HEAD"))
    add("backend              : %s" % retriever.backend)
    add("umbral               : %.4f   k=%d" % (umbral, _rag.K))
    add("fragmentos_indexados : %d" % retriever.store.count())
    add("pdf_indexados        : %d de %d en el directorio"
        % (len(nombres), len(list(CORPUS_DIR.glob("*.pdf")))))
    add("Llamadas al LLM      : 0 (solo se consulta el indice)")
    add("")
    add("El umbral se aplica sobre el puntaje del PRIMER fragmento, que es el")
    add("unico que el LLM cita; los k=5 fragmentos entran en el prompt.")
    add("recall@k = paginas esperadas halladas entre los k primeros / paginas")
    add("esperadas. 'pos' es la posicion (1..10) del primer fragmento que")
    add("contiene el archivo y la pagina esperados.")
    add("")

    resueltos: Dict[str, Optional[str]] = {}

    add("-" * 78)
    add("PREGUNTAS CON RESPUESTA (tipo=responde, condicion=con, abstuvo=no)")
    add("-" * 78)
    add("%-3s %-34s %-5s %-4s %-8s %-8s %-6s %-6s"
        % ("id", "esperado (archivo)", "top5", "pos", "top1", "recall@1",
           "@5", "@10"))
    filas_diag: List[str] = []
    n_con = n_resp = 0
    for item in banco:
        if item["tipo"] != "responde":
            continue
        n_con += 1
        f = con.get(int(item["id"]))
        if f is None or es_abstuvo(f["abstuvo"]) or not f["respuesta"]:
            filas_diag.append("%-3s SIN RESPUESTA EN LA CORRIDA (abstuvo=%s)"
                              % (item["id"], (f or {}).get("abstuvo", "?")))
            continue
        n_resp += 1
        doc_banco = item.get("documento", "")
        if doc_banco not in resueltos:
            resueltos[doc_banco] = resolve_document(doc_banco, nombres)
        doc = resueltos[doc_banco]
        pagina = str(item.get("pagina_pdf", "")).strip()
        esperadas = [str(p) for p in paginas_esperadas(item)]

        hits = retriever.retrieve(item["pregunta"], k=10)

        def coincide(h, pagina_objetivo: str) -> bool:
            return h.archivo == doc and pagina_objetivo in \
                [str(p) for p in h.paginas]

        pos = next((i for i, h in enumerate(hits, 1)
                    if coincide(h, pagina)), None)
        pos_alt = next((i for i, h in enumerate(hits, 1)
                        if pos is None and h.archivo == doc
                        and any(p in [str(x) for x in h.paginas]
                                for p in esperadas)), None)

        def recall(k: int) -> float:
            return 1.0 if any(coincide(h, pagina) for h in hits[:k]) else 0.0

        top1 = hits[0].puntaje if hits else float("nan")
        corto = (doc or "NO RESUELTO")
        if len(corto) > 34:
            corto = corto[:31] + "..."
        nota = pos if pos is not None else (pos_alt if pos_alt is not None
                                            else "-")
        filas_diag.append(
            "%-3s %-34s %-5s %-4s %-8.4f %-8.2f %-6.2f %-6.2f"
            % (item["id"], corto, "si" if (pos is not None and pos <= 5)
               else "no", nota, top1, recall(1), recall(5), recall(10)))
        if pos is None and pos_alt is not None:
            filas_diag.append("     (solo coincide con pagina alternativa "
                              "p.%s; la p.%s no esta en el top 10)"
                              % (item.get("paginas_alternativas"), pagina))
        if doc is None:
            filas_diag.append("     (el archivo del banco NO se resuelve "
                              "contra los %d PDF indexados)" % len(nombres))

    for l in filas_diag:
        add(l)
    add("")
    add("responde: %d en el banco, %d con respuesta en la corrida"
        % (n_con, n_resp))

    abstuvieron = [i for i in banco if i["tipo"] == "responde"
                   and (int(i["id"]) not in con
                        or es_abstuvo(con[int(i["id"])]["abstuvo"])
                        or not con[int(i["id"])]["respuesta"])]
    if abstuvieron:
        add("")
        add("-" * 78)
        add("tipo=responde QUE ABSTUVIERON (sin respuesta: por que)")
        add("-" * 78)
        add("%-3s %-10s %-10s %-8s %-24s %-14s"
            % ("id", "top1", "umbral", ">umbral", "fuentes_citadas",
               "ruta_orquestador"))
        for item in abstuvieron:
            hits = retriever.retrieve(item["pregunta"], k=10)
            top1 = hits[0].puntaje if hits else float("nan")
            f = con.get(int(item["id"]), {})
            add("%-3s %-10.4f %-10.4f %-8s %-24s %-14s"
                % (item["id"], top1, umbral, "si" if top1 > umbral else "no",
                   (f.get("fuentes_citadas") or "-")[:24],
                   rutas.get(str(item["id"]), "-")))
        add("")
        add("'>umbral=no' significa que el umbral bloqueo la consulta ANTES")
        add("de llamar al LLM; si dice 'si' la abstencion vino del propio LLM.")

    add("")
    add("-" * 78)
    add("PREGUNTAS DE ABSTENCION (tipo=abstiene) - puntaje del primer")
    add("fragmento frente al umbral")
    add("-" * 78)
    add("%-3s %-10s %-10s %-8s %-8s %-24s"
        % ("id", "top1", "umbral", ">umbral", "abstuvo", "ruta_orquestador"))
    n_abst = 0
    for item in banco:
        if item["tipo"] != "abstiene":
            continue
        n_abst += 1
        hits = retriever.retrieve(item["pregunta"], k=10)
        top1 = hits[0].puntaje if hits else float("nan")
        f = con.get(int(item["id"]), {})
        add("%-3s %-10.4f %-10.4f %-8s %-8s %-24s"
            % (item["id"], top1, umbral, "si" if top1 > umbral else "no",
               f.get("abstuvo", "?"), rutas.get(str(item["id"]), "-")))

    add("")
    add("abstiene: %d en el banco" % n_abst)
    add("La corrida NO pasa por el orquestador (mismo protocolo que la")
    add("corrida anterior: ask_with_sources). 'ruta_orquestador' registra la")
    add("ruta que elegiria _generate_plan; para 'encuesta' el plan SI")
    add("requeria LLM y aqui no se llamo.")
    add("")
    add("Fin del diagnostico. No se calculo ningun acierto de respuesta.")

    salida = RES / ("diagnostico_final_%s.txt" % ts)
    with open(salida, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")
    return salida


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV de la corrida (por defecto: el mas reciente "
                             "resultados/banco_final_*.csv)")
    parser.add_argument("--banco", type=Path, default=BANCO)
    parser.add_argument("--ts", default=None,
                        help="etiqueta de tiempo para los nombres de salida")
    parser.add_argument("--solo-salidas", action="store_true",
                        help="hoja + clave + metadatos, sin diagnostico")
    parser.add_argument("--solo-diagnostico", action="store_true",
                        help="diagnostico de recuperacion, sin hoja ni clave")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv) if args.csv else csv_mas_reciente()
    if csv_path is None or not csv_path.exists():
        raise SystemExit("No hay CSV de corrida. Pasa --csv o corre el banco "
                         "primero (tests/correr_banco_rag.py).")
    banco_path = Path(args.banco)
    banco = leer_json(banco_path)
    filas = leer_csv(csv_path)
    ts = args.ts or ts_del_csv(csv_path)
    print("csv    : %s (%d filas)" % (csv_path.name, len(filas)))
    print("banco  : %s (%d preguntas)" % (banco_path.name, len(banco)))
    print("ts     : %s" % ts)

    # La ruta del orquestador se calcula una sola vez y se reutiliza.
    from agent.core.orchestrator import StatisticalAgent
    from agent.core.query_planner import classify_scope

    agente = StatisticalAgent()
    rutas = {str(item["id"]): ruta_orquestador(item["pregunta"], agente,
                                               classify_scope)
             for item in banco}

    if not args.solo_diagnostico:
        r = fase_a(banco, banco_path, filas, ts, csv_path, rutas)
        print("hoja   : %s (%d filas, sin columna condicion)"
              % (r["hoja"].name, r["n_hoja"]))
        print("clave  : %s" % r["clave"].name)
        print("meta   : %s" % r["meta"].name)

    if not args.solo_salidas:
        salida = fase_b(banco, banco_path, filas, ts, csv_path, rutas)
        print("diag   : %s" % salida.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
