# -*- coding: utf-8 -*-
"""
Carga del corpus RAG usando pypdf (puro Python).

Guarda un registro por PAGINA FISICA del PDF:
    archivo -> nombre del fichero
    pagina  -> pagina fisica (indice 0-based + 1)
    texto   -> texto que pypdf extrae de esa pagina
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pypdf import PdfReader

CORPUS_DIR = Path(__file__).resolve().parents[2] / "docs" / "rag_corpus"


@dataclass(frozen=True)
class Page:
    archivo: str
    pagina: int          # pagina fisica, 1-based
    texto: str

    @property
    def n_chars(self) -> int:
        return len(self.texto.strip())


def load_document(path) -> List[Page]:
    """Extrae el texto de cada pagina de un unico PDF."""
    path = Path(path)
    pages: List[Page] = []
    try:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            try:
                texto = page.extract_text() or ""
            except Exception as exc:                # pypdf puede fallar en una pagina concreta
                texto = ""
                print(f"[loader] error en {path.name} p.{i + 1}: {exc!r}",
                      file=sys.stderr)
            pages.append(Page(archivo=path.name, pagina=i + 1, texto=texto))
    except Exception as exc:
        print(f"[loader] no se pudo abrir {path.name}: {exc!r}", file=sys.stderr)
    return pages


def _sha256(path: Path) -> str:
    """Huella de los bytes: dos descargas del mismo PDF con nombre distinto."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def load_corpus(corpus_dir=None) -> List[Page]:
    """Carga todos los PDF del corpus, en orden alfabetico de archivo.

    Si dos ficheros tienen los MISMOS bytes se indexan una sola vez: dejaria
    cada pagina y cada fragmento duplicados en el indice y la recuperacion
    devolveria el dosico.
    """
    corpus_dir = Path(corpus_dir) if corpus_dir else CORPUS_DIR
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio del corpus: {corpus_dir}")
    pages: List[Page] = []
    vistos: Dict[str, str] = {}
    for pdf in sorted(corpus_dir.glob("*.pdf")):
        digest = _sha256(pdf)
        if digest in vistos:
            print(f"[loader] duplicado exacto de {vistos[digest]!r}: "
                  f"se omite {pdf.name!r}", file=sys.stderr)
            continue
        vistos[digest] = pdf.name
        pages.extend(load_document(pdf))
    return pages


def pages_by_document(pages: List[Page]) -> Dict[str, List[Page]]:
    """Agrupa las paginas preservando el orden del corpus."""
    grouped: Dict[str, List[Page]] = {}
    for p in pages:
        grouped.setdefault(p.archivo, []).append(p)
    return grouped


def normalize_doc_name(name: str) -> str:
    """Clave comparable entre los nombres del banco y los del disco."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_document(bank_name: str, files) -> Optional[str]:
    """
    Mapa banco -> archivo real SIN modificar el banco.

    Solo se usa para buscar el PDF: los nombres del banco estan
    normalizados (p. ej. '+' o '.' por '_' o ' '), el disco no.
    """
    target = normalize_doc_name(bank_name)
    if not target:
        return None
    for f in files:
        if normalize_doc_name(f) == target:
            return f
    return None
