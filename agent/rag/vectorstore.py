# -*- coding: utf-8 -*-
"""
Almacen vectorial intercambiable.

  1. ChromaDB persistido en rag_store/
  2. Respaldo: arrays NumPy + JSON en rag_store/

Ambos exponen la misma interfaz (add / query / count) y devuelven los
mismos campos, de modo que el resto del codigo no nota el cambio.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

STORE_DIR = Path(__file__).resolve().parents[2] / "rag_store"
COLLECTION = "corpus_felicidad"


@dataclass
class Hit:
    id: str
    texto: str
    archivo: str
    paginas: List[int]
    puntaje: float            # similitud coseno, cuanto mas alto mejor


def _hit(id_: str, texto: str, archivo: str, paginas, puntaje: float) -> Hit:
    if isinstance(paginas, str):
        paginas = [int(x) for x in paginas.split(",") if x.strip()]
    return Hit(id=id_, texto=texto, archivo=archivo,
               paginas=list(paginas or []), puntaje=float(puntaje))


class VectorStore:
    name = "abstract"

    @property
    def dimension(self) -> Optional[int]:
        """Dimension de los vectores ya guardados; None si el indice esta vacio."""
        return None

    def add(self, ids, embeddings, documents, metadatas=None) -> None:
        raise NotImplementedError

    def query(self, embedding: np.ndarray, k: int = 5) -> List[Hit]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    name = "chroma"

    def __init__(self, persist_dir=None, collection: str = COLLECTION):
        import chromadb
        self._dir = str(persist_dir or STORE_DIR)
        Path(self._dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._dir)
        self._name = collection
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, documents, metadatas=None) -> None:
        metadatas = metadatas or [{} for _ in ids]
        # Chroma solo admite str/int/float/bool en metadata
        normalizado = []
        for m in metadatas:
            normalizado.append({
                k: (",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v)
                for k, v in m.items()
            })
        self._collection.upsert(
            ids=list(ids),
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            documents=list(documents),
            metadatas=normalizado,
        )

    def query(self, embedding: np.ndarray, k: int = 5) -> List[Hit]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: List[Hit] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(_hit(
                i, doc,
                meta.get("archivo", ""),
                meta.get("paginas", []),
                1.0 - float(dist),      # cosine distance -> similitud
            ))
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    @property
    def dimension(self) -> Optional[int]:
        """Chroma rechaza cualquier vector de otra dimension
        ("expecting 60000, got 384"), asi que hay que poder leerla antes
        de elegir el embedder."""
        if self.count() == 0:
            return None
        try:
            fila = self._collection.get(include=["embeddings"], limit=1)
            emb = fila.get("embeddings") if fila else None
            if emb is None or len(emb) == 0:
                return None
            return int(np.asarray(emb[0]).reshape(-1).shape[0])
        except BaseException:
            return None

    def clear(self) -> None:
        """Borra todo el indice (necesario al reindexar: upsert no elimina sobras)."""
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )


class NumpyVectorStore(VectorStore):
    """Respaldo 2: matrices NumPy + JSON en rag_store/."""

    name = "numpy-json"

    def __init__(self, persist_dir=None):
        self._dir = Path(persist_dir or STORE_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._emb_path = self._dir / "embeddings.npy"
        self._meta_path = self._dir / "chunks.json"
        self._metas: List[Dict] = []
        self._emb: Optional[np.ndarray] = None
        if self._emb_path.exists() and self._meta_path.exists():
            self._emb = np.load(str(self._emb_path))
            self._metas = json.loads(self._meta_path.read_text(encoding="utf-8"))

    def add(self, ids, embeddings, documents, metadatas=None) -> None:
        metadatas = metadatas or [{} for _ in ids]
        nuevo = np.asarray(embeddings, dtype=np.float32)
        filas = [
            {"id": i, "texto": d, **{k: v for k, v in m.items()}}
            for i, d, m in zip(ids, documents, metadatas)
        ]
        self._emb = nuevo if self._emb is None else np.vstack([self._emb, nuevo])
        self._metas.extend(filas)
        np.save(str(self._emb_path), self._emb)
        self._meta_path.write_text(
            json.dumps(self._metas, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def query(self, embedding: np.ndarray, k: int = 5) -> List[Hit]:
        if self._emb is None or len(self._metas) == 0:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        norma = float(np.linalg.norm(q))
        if norma > 0:
            q = q / norma
        sims = self._emb @ q
        order = np.argsort(-sims)[:k]
        return [
            _hit(self._metas[i]["id"], self._metas[i]["texto"],
                 self._metas[i].get("archivo", ""),
                 self._metas[i].get("paginas", []),
                 float(sims[i]))
            for i in order
        ]

    def count(self) -> int:
        return len(self._metas)

    @property
    def dimension(self) -> Optional[int]:
        if self._emb is None or len(self._emb) == 0:
            return None
        return int(self._emb.shape[1])

    def clear(self) -> None:
        self._emb = None
        self._metas = []
        for p in (self._emb_path, self._meta_path):
            if p.exists():
                p.unlink()


def create_vectorstore(name: str = "auto", persist_dir=None):
    """Igual que create_embedder: primero Chroma, respaldo NumPy+JSON."""
    if name == "numpy":
        return NumpyVectorStore(persist_dir)
    if name == "chroma":
        return ChromaVectorStore(persist_dir)
    try:
        return ChromaVectorStore(persist_dir)
    except BaseException:
        return NumpyVectorStore(persist_dir)
