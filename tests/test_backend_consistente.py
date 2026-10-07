# -*- coding: utf-8 -*-
"""
El backend de embeddings elegido debe ser el que pide el indice.

`create_embedder()` decidia por lo que se pudiera importar. En este equipo
sentence-transformers carga a veces y otras no (Smart App Control), y el
indice se construyo con TF-IDF (60000). Elegir sin mirar el indice acaba en:

    chromadb.errors.InvalidArgumentError:
    Collection expecting embedding with dimension of 60000, got 384

Se comprueba SIN llamar al LLM, SIN tocar la API y SIN cargar
sentence-transformers: solo con dobles de dimension.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.rag import embedder as mod_embedder


class _Doble(mod_embedder.Embedder):
    """Embedder de mentira que solo sirve para declarar una dimension."""

    def __init__(self, dimension, nombre):
        self.name = nombre
        self.backend = f"{nombre}@{dimension}"
        self.dimension = dimension

    def embed(self, texts):
        raise AssertionError("estas pruebas no deben vectorizar nada")


class _AlmacenFalso:
    name = "falso"

    def __init__(self, dimension):
        self._dim = dimension

    @property
    def dimension(self):
        return self._dim

    def count(self):
        return 1


@pytest.fixture
def dobles(monkeypatch):
    """sentence-transformers disponible (384) y TF-IDF (60000)."""
    monkeypatch.setenv("RAG_OPENAI_EMBEDDINGS", "0")
    monkeypatch.setattr(
        mod_embedder, "SentenceTransformerEmbedder",
        lambda: _Doble(384, "sentence-transformers"))
    monkeypatch.setattr(
        mod_embedder, "TfidfEmbedder", lambda: _Doble(60000, "tfidf"))


def test_indice_60000_elige_tfidf_y_descarta_sentence_transformers(dobles):
    emb, descartes = mod_embedder.create_embedder(dim_esperada=60000)
    assert emb.dimension == 60000
    assert any("384" in d and "60000" in d for d in descartes)


def test_sin_indice_previo_respeta_el_orden_del_enunciado(dobles):
    emb, descartes = mod_embedder.create_embedder()
    assert emb.dimension == 384
    assert descartes == []


def test_ninguno_compatibles_falla_con_la_dimension_exigida(dobles):
    with pytest.raises(RuntimeError) as exc:
        mod_embedder.create_embedder(dim_esperada=768)
    texto = str(exc.value)
    assert "768" in texto and "384" in texto and "60000" in texto


def test_retriever_toma_la_dimension_del_almacen(dobles):
    from agent.rag.retriever import Retriever
    r = Retriever(store=_AlmacenFalso(60000))
    assert r.embedder.dimension == 60000


def test_indice_real_y_vectorizador_persistido_coinciden():
    from agent.rag.vectorstore import STORE_DIR, create_vectorstore
    store = create_vectorstore()
    if store.count() == 0:
        pytest.skip("indice vacio: no hay nada que comprobar")
    dim_indice = store.dimension
    assert dim_indice, "el indice no reporta su dimension"

    backend_txt = STORE_DIR / "BACKEND.txt"
    if backend_txt.exists():
        for linea in backend_txt.read_text(encoding="utf-8").splitlines():
            if linea.startswith("dimension"):
                assert int(linea.split(":", 1)[1]) == dim_indice
                break

    if mod_embedder.TFIDF_VECTORIZER.exists():
        assert mod_embedder.TfidfEmbedder().dimension == dim_indice
