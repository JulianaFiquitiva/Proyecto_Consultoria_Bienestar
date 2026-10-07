# -*- coding: utf-8 -*-
"""
Motor RAG — Fase B.

ask_with_sources(pregunta):
  1. recupera los fragmentos (retriever, k=5)
  2. si el mejor puntaje < UMBRAL -> respuesta fija, NO se llama al LLM
  3. si no, arma el prompt con prompts/rag_v1.txt + fragmentos
     y llama a agent/api_client.py
  4. devuelve respuesta + fuentes [{archivo, pagina, fragmento_textual, puntaje}]

El umbral solo se calibra con las 6 preguntas de calibracion (NO con el
banco). Los puntajes quedan en rag_store/calibracion.csv.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from agent.api_client import create_client
from agent.rag.retriever import K_DEFECTO, Retriever
from agent.rag.vectorstore import Hit

PROMPT_V1 = Path(__file__).resolve().parents[2] / "prompts" / "rag_v1.txt"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_CSV = LOG_DIR / "rag_consultas.csv"
CALIBRACION_CSV = Path(__file__).resolve().parents[2] / "rag_store" / "calibracion.csv"

UMBRAL_DEFECTO = 0.0720
# Calibrado SOLO con las 6 preguntas de apoyo (rag_store/calibracion.csv):
#   min(con_respuesta) = 0.0741    max(sin_respuesta) = 0.0804
# Los intervalos se solapan (0.0741 < 0.0804), asi que no hay corte limpio.
# Se elige 0.0720: por debajo del menor puntaje "con respuesta" (0.0741) para
# no abstenerte nunca de una pregunta que el corpus si responde (la abstinencia
# dura es irreversible), y por encima de las dos "sin respuesta" mas claras
# (0.0699 y 0.0686). La tercera (0.0804) pasa al LLM, que por la plantilla
# rag_v1.txt esta obligado a decir que no la encuentra si no esta en el
# contexto: son las dos capas de abstinencia del disenio.
# La escala es del coseno TF-IDF (60000 terminos), NO la de
# sentence-transformers: con ese backend los mismos datos daban
# 0.4192/0.4235 y el umbral era 0.4150. Cambiar de backend obliga a
# volver a correr `python -m agent.rag.cli --calibrar`.
TEXTO_ABSTENCION = "No encuentro una fuente en el corpus que responda esto."
K = K_DEFECTO

# gpt-4-0613 tiene 8192 tokens de contexto. 5 fragmentos de 500-800 palabras
# en español suman ~6600 tokens (el tokenizador rinde ~3 caracteres/token con
# acentos), y con 2000 de completion se pasaba de 8192: la API devolvia
# context_length_exceeded y el motor caia en abstinencia por error.
# Solucion: respuesta corta (700) + techo de caracteres para el prompt.
MAX_TOKENS_RAG = 700
# 8192 - 700 (respuesta) - 150 (margen) = 7342 tokens para el prompt
# => ~22000 caracteres. Se usa 18000 para tener margen real.
PRESUPUESTO_PROMPT_CHARS = 18000

# Las 6 preguntas de calibracion. NO forman parte del banco de 19.
CALIBRACION = [
    ("con_respuesta", "¿Cuántas mujeres participaron en el estudio de la WEMWBS-E?"),
    ("con_respuesta", "¿Qué media obtuvo la dimensión relaciones positivas en estudiantes de psicología?"),
    ("con_respuesta", "¿Qué porcentaje de la varianza total explicaron los cinco factores de la EBUP?"),
    ("sin_respuesta", "¿Qué dicen los documentos sobre tener mascotas y el bienestar de los estudiantes?"),
    ("sin_respuesta", "¿Qué relación reportan los documentos entre el uso de videojuegos y el bienestar universitario?"),
    ("sin_respuesta", "¿Qué prevalencia de tabaquismo reportan los estudios del corpus en universitarios?"),
]


def umbral() -> float:
    """Umbral configurable por variable de entorno RAG_UMBRAL."""
    try:
        return float(os.environ.get("RAG_UMBRAL", UMBRAL_DEFECTO))
    except ValueError:
        return UMBRAL_DEFECTO


@dataclass
class Fuente:
    archivo: str
    pagina: int
    fragmento_textual: str
    puntaje: float


@dataclass
class RespuestaRAG:
    pregunta: str
    respuesta: str
    fuentes: List[Fuente]
    fragmentos: List[Hit]
    abstencion: bool
    puntaje_max: float
    backend: str
    llamada_llm: bool
    error: Optional[str] = None


def _a_fuentes(hits: Sequence[Hit]) -> List[Fuente]:
    fuentes = []
    for h in hits:
        pagina = h.paginas[0] if h.paginas else 0
        fuentes.append(Fuente(archivo=h.archivo, pagina=pagina,
                              fragmento_textual=h.texto, puntaje=h.puntaje))
    return fuentes


def _bloque_fragmento(indice: int, h: Hit) -> str:
    pagina = h.paginas[0] if h.paginas else 0
    rango = "-".join(str(p) for p in h.paginas) or "?"
    # Se entrega la cita ya escrita en el formato exacto [archivo, p. N]
    # para que el modelo la copie en vez de inventar una referencia "1".
    return (
        f"[{indice}] FUENTE: [{h.archivo}, p. {pagina}]\n"
        f"      (el fragmento abarca las paginas {rango}; "
        f"puntaje {h.puntaje:.4f})\n"
        f"{h.texto}"
    )


def _texto_fragmentos(hits: Sequence[Hit],
                      presupuesto: Optional[int] = None) -> str:
    """Une los fragmentos; si hay presupuesto de caracteres, descarta los de
    menor puntaje (gpt-4 se corta con 8192 tokens). El primero siempre cabe."""
    elegidos: List[str] = []
    usado = 0
    for i, h in enumerate(hits, 1):
        bloque = _bloque_fragmento(i, h)
        if presupuesto is not None and elegidos and usado + len(bloque) > presupuesto:
            break
        elegidos.append(bloque)
        usado += len(bloque)
    return "\n\n".join(elegidos)


def _armar_prompt(pregunta: str, hits: Sequence[Hit]) -> str:
    plantilla = PROMPT_V1.read_text(encoding="utf-8")
    fijo = f"{plantilla}\n\n--- CONTEXTO ---\n\n"
    cola = f"\n\n--- PREGUNTA ---\n{pregunta}\n"
    disponible = PRESUPUESTO_PROMPT_CHARS - len(fijo) - len(cola)
    return f"{fijo}{_texto_fragmentos(hits, disponible)}{cola}"


def _es_abstencion_llm(texto: str) -> bool:
    bajo = (texto or "").lower()
    return ("no la encuentro" in bajo) or ("no encuentro" in bajo) or ("no encuentras" in bajo)


def registrar_consulta(backend: str, pregunta: str, hits: Sequence[Hit],
                       respuesta: str, abstencion: bool) -> None:
    """Una fila por consulta en logs/rag_consultas.csv (trazabilidad)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    nuevo = not LOG_CSV.exists()
    fragmentos = " | ".join(
        f"{h.archivo} p.{'-'.join(str(p) for p in h.paginas) or '?'} ({h.puntaje:.4f})"
        for h in hits
    )
    with LOG_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if nuevo:
            writer.writerow(["fecha", "backend", "pregunta", "fragmentos",
                             "respuesta", "abstencion"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            backend, pregunta, fragmentos, respuesta, "si" if abstencion else "no",
        ])


def ask_with_sources(pregunta: str,
                     retriever: Optional[Retriever] = None,
                     llm=None,
                     umbral_valor: Optional[float] = None,
                     registrar: bool = True) -> RespuestaRAG:
    """Pregunta al corpus RAG con cita obligatoria y posibilidad de abstinence."""
    retriever = retriever or Retriever()
    limite = umbral() if umbral_valor is None else umbral_valor
    backend = retriever.backend

    hits = retriever.retrieve(pregunta, k=K)
    puntaje_max = hits[0].puntaje if hits else 0.0

    # --- 2. umbral: NO se llama al LLM -------------------------------------
    if puntaje_max < limite:
        respuesta = TEXTO_ABSTENCION
        salida = RespuestaRAG(
            pregunta=pregunta, respuesta=respuesta, fuentes=[],
            fragmentos=list(hits), abstencion=True, puntaje_max=puntaje_max,
            backend=backend, llamada_llm=False,
        )
        if registrar:
            registrar_consulta(backend, pregunta, hits, respuesta, True)
        return salida

    # --- 3. prompt con prompts/rag_v1.txt + fragmentos -> api_client -------
    if llm is None:
        llm = create_client("openai")
    if not llm.is_configured:
        salida = RespuestaRAG(
            pregunta=pregunta, respuesta=TEXTO_ABSTENCION,
            fuentes=[], fragmentos=list(hits), abstencion=True,
            puntaje_max=puntaje_max, backend=backend, llamada_llm=False,
            error="No hay API key configurada (OPENAI_API_KEY).",
        )
        if registrar:
            registrar_consulta(backend, pregunta, hits, salida.respuesta, True)
        return salida

    try:
        resp = llm.chat([{"role": "user", "content": _armar_prompt(pregunta, hits)}],
                        max_tokens=MAX_TOKENS_RAG)
        if not resp.success:
            raise RuntimeError(resp.error or "fallo la llamada al LLM")
        texto = (resp.content or "").strip() or TEXTO_ABSTENCION
        abstencion = _es_abstencion_llm(texto)
        salida = RespuestaRAG(
            pregunta=pregunta, respuesta=texto, fuentes=_a_fuentes(hits),
            fragmentos=list(hits), abstencion=abstencion,
            puntaje_max=puntaje_max, backend=backend, llamada_llm=True,
        )
    except Exception as exc:
        salida = RespuestaRAG(
            pregunta=pregunta, respuesta=TEXTO_ABSTENCION, fuentes=[],
            fragmentos=list(hits), abstencion=True, puntaje_max=puntaje_max,
            backend=backend, llamada_llm=True, error=str(exc),
        )

    if registrar:
        registrar_consulta(backend, pregunta, hits, salida.respuesta,
                           salida.abstencion)
    return salida


# ── calibracion del umbral ──────────────────────────────────────────────

def calibrar(retriever: Optional[Retriever] = None,
             guardar: bool = True) -> List[dict]:
    """
    Puntaje maximo de las 6 preguntas de calibracion. NO usa el banco
    y NO llama al LLM.
    """
    retriever = retriever or Retriever()
    filas: List[dict] = []
    for tipo, pregunta in CALIBRACION:
        hits = retriever.retrieve(pregunta, k=K)
        mejor = hits[0] if hits else None
        filas.append({
            "pregunta": pregunta,
            "tipo": tipo,
            "puntaje_max": round(mejor.puntaje, 6) if mejor else 0.0,
            "archivo_top": mejor.archivo if mejor else "",
            "pagina_top": (mejor.paginas[0] if mejor and mejor.paginas else ""),
            "puntajes_top5": ";".join(f"{h.puntaje:.4f}" for h in hits),
        })

    if guardar:
        CALIBRACION_CSV.parent.mkdir(parents=True, exist_ok=True)
        with CALIBRACION_CSV.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
    return filas
