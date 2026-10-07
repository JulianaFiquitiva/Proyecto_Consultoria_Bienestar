# -*- coding: utf-8 -*-
"""
Embeddings intercambiables. Orden de preferencia del enunciado:

  1. sentence-transformers con "paraphrase-multilingual-MiniLM-L12-v2" (local)
  2. TF-IDF local (scikit-learn) si 1 falla en el entorno
  3. Embeddings de OpenAI SOLO si se piden explicitamente
     (flag RAG_OPENAI_EMBEDDINGS apagado por defecto)

El resto del codigo solo ve la interfaz Embedder: cambiar de backend no
toca ni al motor RAG ni al CLI. Lo que si se tiene en cuenta es la
dimension del indice ya construido (ver create_embedder).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

MODELO_LOCAL = "paraphrase-multilingual-MiniLM-L12-v2"
BACKEND_OPENAI = "openai"
# El vectorizador TF-IDF se persiste aqui. Sin este fichero, un proceso
# nuevo entraria sin vocabulario y lo entrenaria con la sola consulta:
# la matriz tendria ~17 terminos y Chroma exigiria 54359.
TFIDF_VECTORIZER = Path(__file__).resolve().parents[2] / "rag_store" / "tfidf_vectorizer.joblib"


# ── flag de configuracion: apagado por defecto ──────────────────────────
def openai_embeddings_activos() -> bool:
    """Solo True si el usuario lo pide explicitamente por variable de entorno."""
    return os.environ.get("RAG_OPENAI_EMBEDDINGS", "0").strip().lower() in ("1", "true", "yes", "on")


class Embedder:
    """Interfaz minima: embed / embed_query. Nada mas se usa fuera."""

    name = "abstract"
    backend = "abstract"
    dimension = 0

    def fit(self, texts: Sequence[str]) -> None:
        """Solo lo usan los backends que aprenden del corpus (TF-IDF)."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = MODELO_LOCAL):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        # IMPORTANTE: multilingual-MiniLM declara max_seq_length=128, pero su
        # arquitectura admite 512 posiciones. Con 128 cada fragmento de ~650
        # palabras se trunca a sus primeras ~90 y la recuperacion falla
        # (lo demuestra un fragmento que cita literalmente la respuesta y
        # aun asi no se recupera). Se sube al maximo real de la arquitectura.
        try:
            tope = int(self._model[0].auto_model.config.max_position_embeddings)
        except Exception:
            tope = 512
        self._model.max_seq_length = min(512, tope)
        self.backend = (f"sentence-transformers/{model_name}"
                        f"@seq{self._model.max_seq_length}")
        get_dim = getattr(self._model, "get_embedding_dimension", None) \
            or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class TfidfEmbedder(Embedder):
    """Respaldo 2: TF-IDF local. No sale ni un byte de la maquina.

    El vectorizador AJUSTADO AL CORPUS se guarda en rag_store/ y se recarga
    en cada arranque. Sin persistirlo, el primer proceso que indexa lo entrena
    con los 139 fragmentos pero cualquier proceso posterior lo volvia a entrenar
    con su propia consulta, y el numero de terminos no coincidia.
    """

    name = "tfidf"

    def __init__(self, max_features: int = 60000, ngram_range: Tuple[int, int] = (1, 2),
                 ruta=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._ruta = Path(ruta) if ruta else TFIDF_VECTORIZER
        self._vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, sublinear_tf=True
        )
        self.backend = "tfidf-local"
        self.dimension = 0
        self._fitted = False
        self._cargar()

    def _cargar(self) -> bool:
        """Reutiliza el vectorizador ya entrenado sobre el corpus."""
        if not self._ruta.exists():
            return False
        try:
            import joblib
            self._vectorizer = joblib.load(self._ruta)
            self._fitted = True
            self.dimension = int(len(self._vectorizer.vocabulary_))
            self.backend = f"tfidf-local@{self.dimension}"
            return True
        except BaseException:
            # si el fichero esta corrupto se reentrena desde cero en fit()
            self._fitted = False
            self.dimension = 0
            return False

    def _guardar(self) -> None:
        try:
            import joblib
            self._ruta.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._vectorizer, self._ruta)
        except BaseException:
            pass

    def fit(self, texts: Sequence[str]) -> None:
        self._vectorizer.fit(list(texts))
        self._fitted = True
        self.dimension = int(len(self._vectorizer.get_feature_names_out()))
        self.backend = f"tfidf-local@{self.dimension}"
        self._guardar()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        mat = self._vectorizer.transform(list(texts))
        return np.asarray(mat.todense(), dtype=np.float32)


class OpenAIEmbedder(Embedder):
    """Respaldo 3: SOLO se construye si RAG_OPENAI_EMBEDDINGS esta activo."""

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY no esta definida en el entorno")
        self._client = OpenAI()
        self._model = model
        self.backend = f"{BACKEND_OPENAI}/{model}"
        self.dimension = 1536

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        mat = [d.embedding for d in resp.data]
        arr = np.asarray(mat, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)


def _literal(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def create_embedder(dim_esperada: Optional[int] = None) -> Tuple[Embedder, List[str]]:
    """
    Devuelve (embedder, registro_de_descartes).

    `dim_esperada` es la dimension del indice ya construido (la del almacén al
    consultar, o la del indice previo al reindexar). El orden de preferencia
    del enunciado solo se aplica **entre los backends compatibles** con ese
    indice: elegir por lo que se pueda importar y no por lo que pide el indice
    provoca "Collection expecting embedding with dimension of 60000, got 384",
    que es exactamente lo que le pasa a un equipo donde sentence-transformers
    logra cargar pero el indice se construyo con TF-IDF.

    Cada descarte conserva el mensaje de error LITERAL para BACKEND.txt.
    """
    descartes: List[str] = []

    def incompatible(candidato: Embedder) -> bool:
        if dim_esperada is None or candidato.dimension == dim_esperada:
            return False
        descartes.append(
            f"[{candidato.name}] dimension {candidato.dimension} "
            f"!= la del indice {dim_esperada}"
        )
        return True

    if openai_embeddings_activos():
        try:
            candidato = OpenAIEmbedder()
            if not incompatible(candidato):
                return candidato, descartes
        except Exception as exc:
            descartes.append(f"[3] openai: {_literal(exc)}")

    try:
        candidato = SentenceTransformerEmbedder()
        if not incompatible(candidato):
            return candidato, descartes
    except BaseException as exc:            # incluye SystemExit/HardError de SAC
        descartes.append(f"[1] sentence-transformers + {MODELO_LOCAL}: {_literal(exc)}")

    try:
        candidato = TfidfEmbedder()
        if not incompatible(candidato):
            return candidato, descartes
    except BaseException as exc:
        descartes.append(f"[2] tfidf (scikit-learn): {_literal(exc)}")

    exigido = f" con dimension {dim_esperada}" if dim_esperada else ""
    raise RuntimeError(
        f"Ningun backend de embeddings disponible{exigido}:\n"
        + "\n".join(descartes)
    )
