# -*- coding: utf-8 -*-
"""
Sinónimos para el enrutador del agente.

Las reglas del planificador (query_planner.classify_scope y
StatisticalAgent._fallback_plan) reconocen un vocabulario fijo y sin tildes:
«genero», «seccional», «dimension», «reproduce»... Una persona pregunta con
otras palabras: «¿las chicas están más felices que los chicos?», «¿qué campus
tiene mejor bienestar?», «¿cuál es la dimensión más baja?».

`normalizar_pregunta` traduce la pregunta a ese vocabulario ANTES de
clasificarla: pasa a minúsculas, quita tildes y reemplaza cada sinónimo por
el término canónico. No cambia ningún cálculo; solo amplía las formas de
pedir los análisis que ya existen. La pregunta original se sigue usando para
el corpus documental y para redactar la respuesta.

Para agregar un sinónimo basta con añadir una línea a SINONIMOS.
"""
import re
import unicodedata
from typing import List, Tuple


def sin_tildes(texto: str) -> str:
    """Minúsculas y sin tildes (la ñ pasa a n, igual que en las reglas)."""
    t = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


# (expresión regular sobre el texto sin tildes, término canónico).
# El orden importa: las expresiones largas van antes que las cortas que
# contienen (p. ej. «nivel socioeconomico» antes que «nivel educativo»).
SINONIMOS: List[Tuple[str, str]] = [
    # ── Constructo: bienestar ────────────────────────────────────────────
    (r"satisfaccion (?:con|de) (?:la|su|sus) vidas?", "bienestar"),
    (r"calidad de vida", "bienestar"),
    (r"salud (?:mental|emocional|psicologica)", "bienestar"),
    (r"bienestar (?:psicologico|subjetivo|emocional|estudiantil)", "bienestar"),
    (r"estado (?:emocional|de animo|animico)", "bienestar"),
    (r"felicidad|felices|feliz|contentos?|contentas?", "bienestar"),
    (r"indice de bienestar|indice de felicidad", "bienestar"),

    # ── Población ────────────────────────────────────────────────────────
    (r"alumn[oa]s?|universitari[oa]s|estudiantado|jovenes|educand[oa]s", "estudiantes"),

    # ── Puntaje y niveles ────────────────────────────────────────────────
    (r"puntuaciones?|calificacion del bienestar|score", "puntaje"),
    (r"en (?:situacion de )?riesgo|vulnerables?|bienestar (?:bajo|deficiente|critico)",
     "en riesgo"),
    (r"bienestar (?:alto|elevado|optimo|sobresaliente)", "bienestar alto"),

    # ── Variable: estrato (va antes que «nivel») ─────────────────────────
    (r"(?:nivel|condicion|situacion|clase|posicion) (?:socio ?economic[oa]|social|economic[oa])",
     "estrato socioeconomico"),
    (r"\bricos?\b|\bpobres?\b|estratos? (?:altos?|bajos?)", "estrato socioeconomico"),

    # ── Variable: nivel de formación ─────────────────────────────────────
    (r"nivel(?:es)? (?:de formacion|educativos?|academicos?|de estudios?|de escolaridad)",
     "nivel de formacion"),
    (r"tipo de (?:formacion|estudios?)|grado academico", "nivel de formacion"),
    (r"tecnologos?|tecnologic[oa]s?|tecnic[oa]s? profesional(?:es)?", "nivel de formacion"),

    # ── Variable: pregrado / posgrado ────────────────────────────────────
    (r"maestrias?|doctorados?|especializacion(?:es)?|magister|postgrados?", "posgrado"),
    (r"licenciaturas?|profesional(?:es)? universitari[oa]s?", "pregrado"),

    # ── Variable: género ─────────────────────────────────────────────────
    (r"\b(?:chicas|nenas|senoritas|damas|estudiantes mujeres)\b", "mujeres"),
    (r"\b(?:chicos|varones?|caballeros|estudiantes hombres)\b", "hombres"),
    (r"\bfemenin[oa]s?\b", "mujeres"),
    (r"\bmasculin[oa]s?\b", "hombres"),

    # ── Variable: seccional ──────────────────────────────────────────────
    (r"\b(?:campus|sedes|seccionales|ciudad(?:es)?|regional(?:es)?|"
     r"region(?:es)?|ubicacion(?:es)?)\b", "seccional"),

    # ── Variable: programa ───────────────────────────────────────────────
    (r"\b(?:facultad(?:es)?|disciplinas?|areas? (?:de|del) conocimiento)\b", "programa"),

    # ── Dimensiones de Ryff ──────────────────────────────────────────────
    (r"\b(?:aspectos?|componentes?|facetas?|subescalas?|areas? del bienestar)\b",
     "dimension"),
    (r"autoestima|aceptacion de si(?: mismo)?|aceptarse", "dimension autoaceptacion"),
    (r"sentido (?:de|en) (?:la )?vida|proposito (?:de|en) (?:la )?vida|metas (?:de|en) (?:la )?vida",
     "dimension proposito de vida"),
    (r"relaciones (?:con (?:los )?(?:demas|otros)|interpersonales|sociales)|amistades|vinculos",
     "dimension relaciones positivas"),
    (r"independencia|tomar (?:sus|mis) propias decisiones", "dimension autonomia"),
    (r"desarrollo personal|superacion personal", "dimension crecimiento personal"),
    (r"(?:control|manejo) (?:del|de su) (?:entorno|ambiente|vida)", "dimension dominio del entorno"),
    (r"\bmas debil\b|\bpeor\b(?= dimension)", "mas baja"),
    (r"\bmas fuerte\b|\bmejor\b(?= dimension)", "mas alta"),

    # ── DIF: ¿la escala mide igual? ─────────────────────────────────────
    (r"sesgo (?:en|de) (?:los |las )?(?:items|preguntas|escala)|items? sesgad[oa]s?|"
     r"preguntas? sesgad[oa]s?", "escala mide igual"),
    (r"(?:escala|instrumento|encuesta|items?|preguntas?) (?:es |son )?"
     r"(?:justa|justo|justas|justos|equitativ[oa]s?|imparcial(?:es)?)", "escala mide igual"),
    (r"funcionan? (?:distinto|diferente|de forma distinta|de manera distinta)",
     "escala mide igual"),

    # ── Reproducibilidad del estudio anterior ───────────────────────────
    (r"coincide[n]?|concuerda[n]?|replica[nr]?|reproducir|se reproducen?", "reproduce"),
    (r"informe (?:original|anterior)|estudio original|consultorio de estadistica",
     "estudio anterior"),

    # ── Brecha ajustada ──────────────────────────────────────────────────
    (r"teniendo en cuenta|descontando|aun considerando|quitando el efecto de|"
     r"sin el efecto de|tomando en cuenta", "controlando"),

    # ── Resumen ──────────────────────────────────────────────────────────
    (r"panorama|vision general|en terminos generales|a grandes rasgos|sintesis", "resumen general"),
]

_COMPILADOS = [(re.compile(p), r) for p, r in SINONIMOS]


def normalizar_pregunta(pregunta: str) -> str:
    """Pregunta en minúsculas, sin tildes y con los sinónimos traducidos."""
    q = sin_tildes(pregunta or "")
    for patron, canonico in _COMPILADOS:
        q = patron.sub(canonico, q)
    return re.sub(r"\s+", " ", q).strip()
