# -*- coding: utf-8 -*-
"""
FASE A — indexar el corpus y reportar.

  python -m agent.rag.index [--rebuild]

Este script NO responde preguntas (eso es la Fase B). Aqui solo:
  1. extrae las paginas con pypdf
  2. segmenta en fragmentos
  3. elige y declara el backend de embeddings/almacen
  4. imprime el informe del corpus
  5. verifica, SOLO LECTURA, las citas del banco contra el texto extraido
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.rag.chunker import Chunk, chunk_corpus, stats
from agent.rag.embedder import create_embedder, openai_embeddings_activos
from agent.rag.loader import (CORPUS_DIR, Page, load_corpus,
                              pages_by_document, resolve_document)
from agent.rag.vectorstore import STORE_DIR, create_vectorstore

BANCO = Path(__file__).resolve().parents[2] / "tests" / "banco_preguntas_literatura.json"
BACKEND_TXT = STORE_DIR / "BACKEND.txt"
PAGINA_MIN_CARCATERES = 50
RATIO_PAGINA_CORTA = 0.5          # < 50% del promedio de su documento
N_FRAGMENTOS_ALEATORIOS = 5
SEMILLA = 42                      # fija para que el informe sea reproducible

# Bitacora de instalacion literal (evidencia del camino seguido)
BITACORA_INSTALACION = """\
INTENTOS DE INSTALACION (mensaje literal)
  [pypdf]          Successfully installed pypdf-6.19.0
                   verificacion: pypdf 6.19.0 OK
  [torch]          Successfully installed torch-2.14.1+cpu
                   verificacion: torch OK 2.14.1+cpu
  [sentence-transformers, intento 1]
                   el proceso se interrumpio a los 900 s; quedo incompleto:
                     ModuleNotFoundError: No module named 'sentence_transformers'
  [sentence-transformers, intento 2]
                   Successfully installed sentence-transformers-6.1.0
                   verificacion: sentence_transformers 6.1.0 OK
  [chromadb]       Successfully installed chromadb-1.5.9 (y onnxruntime-1.30.0)
                   verificacion: chromadb 1.5.9 OK
  [API de Chroma]  AttributeError: 'Client' object has no attribute 'get_or_create'
                   -> uso correcto: client.get_or_create_collection(...)
  [PyMuPDF/fitz]   NO se instalo: el enunciado lo prohibe (usa .pyd que
                   Smart App Control podria bloquear).
  [pdfplumber]     NO se instalo: no hace falta, pypdf cubre la extraccion.
  [torch, 2026-10-05]
                   Smart App Control bloquea torch\\_C.cp314-win_amd64.pyd:
                     ImportError: DLL load failed while importing _C: Una
                     directiva de Control de aplicaciones bloqueo este archivo.
                   el fichero esta integro y su sha256 coincide con el
                   RECORD de pip, y copiarlo a otra ruta NO ayuda: SAC
                   decide por hash, no por camino. Reinstalar el mismo
                   wheel produce el mismo hash -> no arregla nada.
                   solo ese modulo esta bloqueado; los DLL de torch\\lib
                   se cargan bien por ctypes.WinDLL.
                    -> solucion: se usa el respaldo 2 (TF-IDF local) del
                   enunciado. torch/sentence-transformers quedan
                   instalados pero inutilizables en este equipo.
  [2026-10-06]      COMPROBADO EN EJECUCION: sentence-transformers SI carga
                   hoy (199 tensores, 384 dimensiones), asi que el bloqueo
                   de SAC ya no se reproduce. Se descarta igual por
                   incompatibilidad con el indice:
                     [sentence-transformers] dimension 384 != la del indice 60000
                   Volver a la opcion 1 del enunciado exige, en este orden:
                     python -m agent.rag.index --rebuild
                     python -m agent.rag.cli --calibrar
                     tests/correr_banco_rag.py
                   porque la escala de los puntajes y el umbral 0.0720 son
                   propios de TF-IDF y no sirven para similitud coseno.
"""


def _norm_espacios(texto: str) -> str:
    return " ".join(str(texto).split())


def _sin_acentos(texto: str) -> str:
    import unicodedata
    descompuesto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in descompuesto if not unicodedata.combining(c)).lower()


def _solo_alnum(texto: str) -> str:
    """Ignora espacios, puntuacion y guiones de fin de linea."""
    return "".join(c for c in _sin_acentos(texto) if c.isalnum())


def informe_corpus(pages: Sequence[Page], chunks: Sequence[Chunk]) -> None:
    grupos = pages_by_document(list(pages))
    print("=" * 78)
    print("1. CORPUS INDEXADO")
    print("=" * 78)
    print(f"Directorio : {CORPUS_DIR}")
    print(f"Documentos : {len(grupos)}")
    print(f"Paginas    : {len(pages)}")
    print(f"Fragmentos : {len(chunks)}")
    print()
    print("Paginas por documento:")
    for archivo, ps in grupos.items():
        total = sum(p.n_chars for p in ps)
        media = total / len(ps) if ps else 0
        print(f"  {len(ps):>3} pags  {total:>7} car.  media {media:>7.0f}  {archivo}")
    print()
    s = stats(chunks)
    if s["n"]:
        print("Fragmentos (palabras ~ tokens):")
        print(f"  n={s['n']}  min={s['min']}  max={s['max']}  "
              f"media={s['media']:.0f}  fuera de 500-800: {s['fuera_de_horquilla']}")
    print()


def informe_fragmentos_aleatorios(chunks: Sequence[Chunk]) -> None:
    print("=" * 78)
    print(f"2. {N_FRAGMENTOS_ALEATORIOS} FRAGMENTOS AL AZAR (completos, "
          f"semilla={SEMILLA})")
    print("=" * 78)
    if not chunks:
        print("  (sin fragmentos)")
        return
    rng = random.Random(SEMILLA)
    elegidos = rng.sample(list(chunks), min(N_FRAGMENTOS_ALEATORIOS, len(chunks)))
    for i, c in enumerate(elegidos, 1):
        pags = ",".join(str(p) for p in c.paginas)
        print(f"\n--- fragmento {i}: {c.id}")
        print(f"    archivo={c.archivo}  paginas=[{pags}]  palabras={c.n_palabras}")
        print(f"    {c.texto}")
    print()


def informe_paginas_sospechosas(pages: Sequence[Page]) -> None:
    print("=" * 78)
    print(f"3. PAGINAS SOSPECHOSAS (posible PDF escaneado o texto corto)")
    print("=" * 78)
    grupos = pages_by_document(list(pages))

    cortas = [p for p in pages if p.n_chars < PAGINA_MIN_CARCATERES]
    print(f"\n3a. Paginas con menos de {PAGINA_MIN_CARCATERES} caracteres: "
          f"{len(cortas)}")
    for p in cortas:
        print(f"    [{p.archivo}] p.{p.pagina} -> {p.n_chars} car. "
              f"texto={p.texto.strip()!r}")

    print(f"\n3b. Paginas con < {int(RATIO_PAGINA_CORTA * 100)}% del promedio "
          f"de su documento:")
    n_cortas_b = 0
    for archivo, ps in grupos.items():
        validas = [p.n_chars for p in ps if p.n_chars > 0]
        if not validas:
            print(f"    [{archivo}] TODAS las paginas sin texto extraible")
            continue
        media = sum(validas) / len(validas)
        umbral = media * RATIO_PAGINA_CORTA
        for p in ps:
            if 0 < p.n_chars < umbral:
                n_cortas_b += 1
                print(f"    [{archivo}] p.{p.pagina} -> {p.n_chars} car. "
                      f"(promedio doc {media:.0f}, umbral {umbral:.0f})")
    if n_cortas_b == 0:
        print("    (ninguna)")
    print()


def verificar_banco(pages: Sequence[Page]) -> None:
    print("=" * 78)
    print("4. VERIFICACION DE CITAS CONTRA EL BANCO (solo lectura)")
    print("=" * 78)
    if not BANCO.exists():
        print(f"  No se encuentra el banco: {BANCO}")
        return
    banco = json.loads(BANCO.read_text(encoding="utf-8"))
    grupos = pages_by_document(list(pages))
    nombres = list(grupos.keys())

    total = 0
    coinciden = 0
    fallos: List[str] = []

    for item in banco:
        if item.get("tipo") != "responde":
            continue
        docs = [d.strip() for d in str(item["documento"]).split("|")]
        pags = [p.strip() for p in str(item["pagina_pdf"]).split("|")]
        citas_raw = str(item.get("cita_textual") or "")
        if "||" in citas_raw:
            citas = [c.strip() for c in citas_raw.split("||")]
        else:
            citas = [citas_raw]
        if len(citas) != len(docs):
            citas = [citas_raw] if len(docs) == 1 else citas * len(docs)

        for doc, pag, cita in zip(docs, pags, citas):
            total += 1
            ident = f"#{item['id']} {doc} p.{pag}"
            if not cita.strip():
                fallos.append(f"  {ident}\n      MOTIVO: cita_textual vacia")
                continue
            real = resolve_document(doc, nombres)
            if real is None:
                fallos.append(f"  {ident}\n      MOTIVO: el archivo del banco "
                              f"no existe en {CORPUS_DIR.name}/")
                continue
            try:
                npag = int(str(pag).strip())
            except ValueError:
                fallos.append(f"  {ident}\n      MOTIVO: pagina_pdf no es numero")
                continue
            pagina = next((p for p in grupos[real] if p.pagina == npag), None)
            if pagina is None:
                fallos.append(f"  {ident}\n      MOTIVO: el PDF tiene "
                              f"{len(grupos[real])} paginas, no llega a {npag}")
                continue

            en_banco = _norm_espacios(cita)
            en_pdf = _norm_espacios(pagina.texto)
            if en_banco and en_banco in en_pdf:
                coinciden += 1
            else:
                extra = ""
                if _sin_acentos(en_banco) in _sin_acentos(en_pdf):
                    extra = ("\n      [diagnostico NO valido como coincidencia] "
                             "coincide si se ignoran acentos y mayusculas")
                elif _solo_alnum(en_banco) in _solo_alnum(en_pdf):
                    extra = ("\n      [diagnostico NO valido como coincidencia] "
                             "la cita SI esta en esta misma pagina: solo falla por "
                             "la segmentacion de pypdf (parte palabras con guion "
                             "de fin de linea, p. ej. 'cons-truccion')")
                fallos.append(
                    f"  {ident}\n"
                    f"      banco    : {en_banco}\n"
                    f"      pagina extraida (p.{npag}, {pagina.n_chars} car.):\n"
                    f"      --------------------------------------------------\n"
                    f"      {en_pdf}\n"
                    f"      --------------------------------------------------"
                    + extra
                )

    print(f"\nPreguntas tipo 'responde' revisadas: "
          f"{sum(1 for i in banco if i.get('tipo') == 'responde')}")
    print(f"Citas comprobadas (documento x pagina): {total}")
    print(f"COINCIDEN : {coinciden}")
    print(f"NO COINCIDEN: {total - coinciden}")
    if fallos:
        print("\nListado de las que NO coinciden (con el texto extraido):")
        for f in fallos:
            print(f)
    else:
        print("\nTodas las citas aparecen en el texto extraido.")
    print()


def escribir_backend(backend: str, almacen: str, dimension: int,
                     descartes: Sequence[str]) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    lineas = [
        "BACKEND ACTIVO — modulo RAG de IndiceFelicidad",
        "=" * 70,
        f"fecha              : {datetime.now().isoformat(timespec='seconds')}",
        f"embeddings         : {backend}",
        f"dimension          : {dimension}",
        f"almacen            : {almacen}",
        f"directorio         : {STORE_DIR}",
        f"flag openai        : {'ACTIVO' if openai_embeddings_activos() else 'apagado por defecto (RAG_OPENAI_EMBEDDINGS)'}",
        "",
        "ORDEN DE PREFERENCIA DEL ENUNCIADO",
        "  1. sentence-transformers paraphrase-multilingual-MiniLM-L12-v2 + ChromaDB",
        "  2. TF-IDF local (scikit-learn) + arrays NumPy/JSON",
        "  3. OpenAI embeddings SOLO si se pide explicitamente (flag apagado)",
        "",
        "DESCARTES",
    ]
    if descartes:
        lineas.extend(f"  {d}" for d in descartes)
    else:
        lineas.append("  (ninguno: funciono la opcion 1, la de mayor prioridad)")
    lineas += ["", BITACORA_INSTALACION.rstrip()]
    BACKEND_TXT.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"BACKEND escrito en {BACKEND_TXT}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fase A: indexar el corpus RAG")
    ap.add_argument("--rebuild", action="store_true",
                    help="reconstruye el indice aunque ya exista")
    args = ap.parse_args()

    print(f"Iniciando Fase A — {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # 1. cargar y segmentar
    pages = load_corpus()
    if not pages:
        print("No se extrajo texto de ningun PDF.")
        return 1
    chunks = chunk_corpus(pages)
    informe_corpus(pages, chunks)

    # 2. backend de embeddings + almacen (intercambiables).
    #    El indice ya construido manda: si trae una dimension, solo vale un
    #    embedder compatible. Sin esto, un --rebuild cambiaria de backend y
    #    dejaria sin validez el umbral calibrado y los resultados del banco.
    store = create_vectorstore()
    embedder, descartes = create_embedder(dim_esperada=store.dimension)
    print("=" * 78)
    print("1b. BACKEND DE EMBEDDINGS Y ALMACEN")
    print("=" * 78)
    print(f"  embeddings : {embedder.backend}")
    print(f"  almacen    : {store.name}")
    if descartes:
        print("  descartes  :")
        for d in descartes:
            print(f"    {d}")
    else:
        print("  descartes  : (ninguno)")
    print()

    # 3. indexar (se salta si el indice ya coincide)
    if not args.rebuild and store.count() == len(chunks) and len(chunks) > 0:
        print(f"Indice ya existente y coherente ({store.count()} fragmentos): "
              f"se reutiliza.\n")
    else:
        print(f"Indexando {len(chunks)} fragmentos ...")
        store.clear()          # upsert no borra los fragmentos obsoletos
        texts = [c.texto for c in chunks]
        embedder.fit(texts)
        vecs = embedder.embed(texts)
        store.add(
            ids=[c.id for c in chunks],
            embeddings=vecs,
            documents=texts,
            metadatas=[{"archivo": c.archivo, "paginas": list(c.paginas)}
                       for c in chunks],
        )
        print(f"Indexados {store.count()} fragmentos.\n")

    # 4. informes
    informe_fragmentos_aleatorios(chunks)
    informe_paginas_sospechosas(pages)
    verificar_banco(pages)

    escribir_backend(embedder.backend, store.name, embedder.dimension, descartes)
    print("\nFASE A completada. Detenido aqui: no se ha llamado al LLM "
          "ni se ha respondido ninguna pregunta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
