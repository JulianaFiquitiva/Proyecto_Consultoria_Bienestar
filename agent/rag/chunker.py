# -*- coding: utf-8 -*-
"""
Segmentacion del corpus en fragmentos.

Reglas del enunciado:
    - ~500-800 tokens, approximados aqui por palabras -> ventana objetivo 650.
    - 15% de solapamiento entre fragmentos consecutivos.
    - Cada fragmento guarda texto, archivo y pagina(s) de origen.
    - Un fragmento NUNCA cruza de un documento a otro.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from agent.rag.loader import Page, pages_by_document

TARGET_WORDS = 650          # punto medio de la horquilla 500-800
OVERLAP_RATIO = 0.15        # 15% de solapamiento
MIN_WORDS = 300             # una cola mas pequena se absorbe en el fragmento anterior
_SENTENCE_END = ".!?:;"


@dataclass(frozen=True)
class Chunk:
    id: str
    archivo: str
    paginas: Tuple[int, ...]
    texto: str
    n_palabras: int


def _spans_of(pages: Sequence[Page], target: int, overlap_ratio: float) -> List[Tuple[int, int]]:
    """
    Divide UN documento en ventanas de palabras con solapamiento.

    Devuelve pares (inicio, fin) sobre la lista plana de palabras del
    documento. El solapamiento se aplica entre ventanas consecutivas.
    """
    words: List[str] = []
    for p in pages:
        words.extend(p.texto.split())

    n = len(words)
    if n == 0:
        return []

    spans: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + target, n)

        # Cortar en frontera de oracion si no llegamos al final del documento
        if end < n:
            floor = start + int(target * 0.6)
            for i in range(end - 1, floor, -1):
                if words[i] and words[i][-1] in _SENTENCE_END:
                    end = i + 1
                    break

        spans.append((start, end))
        if end >= n:
            break
        # 15% de solapamiento con el fragmento anterior
        overlap = max(1, int((end - start) * overlap_ratio))
        start = max(start + 1, end - overlap)

    # Una cola final demasiado pequena se absorbe en el fragmento anterior
    # (se extiende el extremo: NO se concatenan textos, asi no se duplica
    #  la zona de solapamiento).
    if len(spans) >= 2 and (spans[-1][1] - spans[-1][0]) < MIN_WORDS:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()

    return spans


def _spans_to_pages(words_pages: List[int], start: int, end: int) -> Tuple[int, ...]:
    return tuple(sorted(set(words_pages[start:end])))


def chunk_corpus(pages: Sequence[Page],
                 target_words: int = TARGET_WORDS,
                 overlap_ratio: float = OVERLAP_RATIO) -> List[Chunk]:
    """Segmenta todo el corpus. Un fragmento nunca cruza de documento a otro."""
    chunks: List[Chunk] = []
    for archivo, doc_pages in pages_by_document(list(pages)).items():
        words: List[str] = []
        word_pages: List[int] = []
        for p in doc_pages:
            for w in p.texto.split():
                words.append(w)
                word_pages.append(p.pagina)

        for i, (start, end) in enumerate(_spans_of(doc_pages, target_words, overlap_ratio)):
            texto = " ".join(words[start:end])
            paginas = _spans_to_pages(word_pages, start, end)
            first = paginas[0] if paginas else 0
            last = paginas[-1] if paginas else 0
            chunks.append(Chunk(
                id=f"{archivo}#p{first}-{last}#{i}",
                archivo=archivo,
                paginas=paginas,
                texto=texto,
                n_palabras=len(texto.split()),
            ))
    return chunks


def stats(chunks: Sequence[Chunk]) -> Dict[str, float]:
    if not chunks:
        return {"n": 0}
    sizes = [c.n_palabras for c in chunks]
    return {
        "n": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "media": sum(sizes) / len(sizes),
        "fuera_de_horquilla": sum(1 for s in sizes if s < 500 or s > 800),
    }
