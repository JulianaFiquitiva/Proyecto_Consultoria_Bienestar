# -*- coding: utf-8 -*-
"""Encuesta en vivo: resultados que se actualizan con cada respuesta nueva.

Lee datos_vivos/hallazgos.json, que genera la GitHub Action
.github/workflows/encuesta_viva.yml con agent/encuesta_viva/actualizar.py.
Solo agregados, igual que el resto de la app.
"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

import comun as c
from agent.encuesta_viva.actualizar import MIN_RESPUESTAS, SALIDA


@st.cache_data(show_spinner=False)
def _leer(ruta: str, version: float):   # sin "_": Streamlit ignora esos argumentos
    """Se vuelve a leer cuando cambia el archivo (la versión es su fecha)."""
    return json.loads(open(ruta, encoding="utf-8").read())


def _vivos():
    if not SALIDA.exists():
        return None
    return _leer(str(SALIDA), SALIDA.stat().st_mtime)


c.hero("Encuesta en vivo", "El bienestar, respuesta a respuesta",
       "Cada vez que alguien responde el formulario, los resultados se recalculan con la "
       "misma metodología del estudio: depuración, factores de expansión calibrados, puntaje "
       "IRT y varianza del diseño muestral.")

V = _vivos()
if V is None:
    c.tarjetas([
        ("1. Formulario", "Los estudiantes responden el formulario de la escala de Ryff "
         "(29 ítems), el mismo del estudio.", "1"),
        ("2. Actualización automática", "Cada hora, una tarea en GitHub descarga las respuestas y "
         "corre el pipeline completo (agent/encuesta_viva).", "2"),
        ("3. Resultados", "Esta página muestra los agregados: nunca respuestas individuales.", "3"),
    ])
    st.info("Todavía no hay respuestas procesadas de la encuesta en vivo.")
    c.pie_pagina()
    st.stop()

E = V["encuesta_viva"]
R = V["resumen"]
t = R["T_global"]
estudio = c.H()["resumen"]["T_global"]["media"]

hora = datetime.fromisoformat(E["actualizado"]).astimezone(ZoneInfo("America/Bogota"))
st.caption(f"Última actualización: {hora:%d/%m/%Y %H:%M} (hora de Colombia) · "
           f"{E['respuestas_recibidas']:,} respuestas recibidas, {E['respuestas_validas']:,} "
           "válidas (consentimiento, tratamiento de datos y mayoría de edad).")
if E["pocas_respuestas"]:
    st.warning(f"Hay menos de {MIN_RESPUESTAS} respuestas con puntaje: las cifras son muy "
               "inestables y los intervalos, muy anchos.")

c.kpis([
    ("Puntaje T (en vivo)", f"{t['media']:.1f}", c.ic(t), "acento"),
    ("Frente al estudio 2025", f"{t['media'] - estudio:+.1f}",
     f"estudio: T = {estudio:.1f}", ""),
    ("En riesgo", f"{R['riesgo']['media'] * 100:.1f} %", c.ic(R["riesgo"], pct=True), "riesgo"),
    ("Bienestar alto", f"{R['alto']['media'] * 100:.1f} %", c.ic(R["alto"], pct=True), "alto"),
])
st.caption("El puntaje usa los parámetros IRT fijos del estudio, así que está en la misma "
           "escala y se puede comparar directamente con T = "
           f"{estudio:.1f}.")

col1, col2 = st.columns(2)
with col1:
    c.mostrar(c.histograma(R))
with col2:
    c.mostrar(c.dona_niveles(R["riesgo"]["media"], R["alto"]["media"],
                             "Estudiantes por nivel de bienestar", 340))

comp = {k: v for k, v in V.get("comparaciones", {}).items() if v.get("grupos")}
c.seccion("Por grupo", "Comparaciones en vivo",
          f"Solo aparecen los grupos con al menos {V['min_n_grupo']} respuestas.")
if comp:
    clave = st.segmented_control("Comparar por", list(comp), default=list(comp)[0],
                                 format_func=lambda k: comp[k]["etiqueta"], key="vivo_variable")
    clave = clave or list(comp)[0]
    grupos = comp[clave]["grupos"]
    c.mostrar(c.forest(list(grupos), [g["media"] for g in grupos.values()],
                       [g["ic"] for g in grupos.values()],
                       f"Puntaje T por {comp[clave]['etiqueta'].lower()} (IC 95 %)",
                       referencia=t["media"]))
else:
    st.info("Aún no hay grupos con suficientes respuestas para comparar.")

if E["programas_sin_nbc"]:
    st.caption("Programas nuevos sin núcleo básico asignado: " + ", ".join(E["programas_sin_nbc"]))
c.pie_pagina()
