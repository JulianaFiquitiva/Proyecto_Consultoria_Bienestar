# -*- coding: utf-8 -*-
"""
Recuperacion de fragmentos: los k mas similares con su puntaje.

Con el backend activo (sentence-transformers + Chroma, cosine) el puntaje
es la similitud coseno: -1..1, cuanto mas alto mejor.
"""
from __future__ import annotations

from typing import List, Optional

from agent.rag.embedder import create_embedder
from agent.rag.vectorstore import Hit, create_vectorstore

K_DEFECTO = 5


class Retriever:
    """Une el embedder activo con el almacén activo. Ambos son intercambiables."""

    def __init__(self, embedder=None, store=None, k: int = K_DEFECTO):
        # Primero el almacén: el indice existente fija la dimension, y solo
        # despues se elige un embedder que la cumpla.
        self.store = store if store is not None else create_vectorstore()
        if embedder is None:
            embedder, _ = create_embedder(dim_esperada=self.store.dimension)
        self.embedder = embedder
        self.k = k

    @property
    def backend(self) -> str:
        return f"embedder={self.embedder.backend} | store={self.store.name}"

    def retrieve(self, pregunta: str, k: Optional[int] = None) -> List[Hit]:
        """Devuelve hasta k fragmentos ordenados de mas a menos similar."""
        k = self.k if k is None else k
        if self.store.count() == 0:
            return []
        vector = self.embedder.embed_query(pregunta)
        return self.store.query(vector, k=k)


def retrieve(pregunta: str, k: int = K_DEFECTO, embedder=None, store=None) -> List[Hit]:
    """Funcion de conveniencia: construye el retriever bajo demanda."""
    return Retriever(embedder=embedder, store=store, k=k).retrieve(pregunta, k=k)
