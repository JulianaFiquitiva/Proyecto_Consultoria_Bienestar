# -*- coding: utf-8 -*-
"""Resultados generales: distribución, niveles e intervalos."""
import streamlit as st

import comun as c

H = c.H()
R = H["resumen"]
t = R["T_global"]

c.hero("Resultados generales", "El bienestar de la población",
       "Estimaciones poblacionales con los factores de expansión y la varianza del diseño "
       "muestral estratificado.")

c.kpis([
    ("Puntaje T poblacional", f"{t['media']:.2f}", c.ic(t, 2), "acento"),
    ("En riesgo", f"{R['riesgo']['media'] * 100:.1f} %", c.ic(R["riesgo"], pct=True), "riesgo"),
    ("Bienestar alto", f"{R['alto']['media'] * 100:.1f} %", c.ic(R["alto"], pct=True), "alto"),
    ("Efecto de diseño", f"{t['deff']:.2f}", f"n efectivo ≈ {R['n'] / t['deff']:,.0f}", ""),
])

c.mostrar(c.histograma(R))
st.caption(f"La media muestral sin ponderar es {R['media_sin_ponderar']:.2f}: ponderar corrige "
           "la sobre- o subrepresentación de cada estrato.")

col1, col2 = st.columns(2)
with col1:
    c.mostrar(c.dona_niveles(R["riesgo"]["media"], R["alto"]["media"],
                             "Estudiantes por nivel de bienestar", 340))
with col2:
    M = H["ic_metodos"]
    c.mostrar(c.forest([m["metodo"] for m in M], [m["media"] for m in M],
                       [m["ic"] for m in M], "El mismo promedio, tres formas de calcularlo",
                       color=c.AZUL, alto=340))
    st.caption("Ponderar mueve la media; la varianza del diseño ensancha un poco el intervalo. "
               "La app usa siempre la tercera.")

c.seccion("Por grupo", "Niveles de bienestar en cada grupo",
          "Porcentaje de estudiantes en riesgo, en el rango típico y con bienestar alto.")
comp = H["comparaciones"]
clave = st.segmented_control("Variable", list(comp), format_func=lambda k: comp[k]["etiqueta"],
                             default=list(comp)[0], key="niveles_variable")
if clave:
    fig = c.barras_niveles(comp[clave]["grupos"], f"Niveles por {comp[clave]['etiqueta'].lower()}")
    if fig:
        c.mostrar(fig)
c.pie_pagina()
