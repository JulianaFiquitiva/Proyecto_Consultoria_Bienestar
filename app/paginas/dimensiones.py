# -*- coding: utf-8 -*-
"""Las seis dimensiones de Ryff: perfil poblacional y por grupo."""
import streamlit as st

import comun as c

H = c.H()
D = H["dimensiones"]
comp = H["comparaciones"]
orden = sorted(D, key=lambda k: -D[k]["media"])

c.hero("Dimensiones", "Las seis caras del bienestar",
       "Puntaje T poblacional en cada dimensión de la escala de Ryff.")

c.kpis([
    ("Dimensión más alta", orden[0], f"T = {D[orden[0]]['media']:.2f}", "alto"),
    ("Dimensión más baja", orden[-1], f"T = {D[orden[-1]]['media']:.2f}", "riesgo"),
    ("Distancia entre ambas", f"{D[orden[0]]['media'] - D[orden[-1]]['media']:.2f}",
     "puntos T: los intervalos se solapan", ""),
])

col1, col2 = st.columns(2)
with col1:
    c.mostrar(c.radar([("Población", {k: D[k]["media"] for k in D})],
                      "Perfil de la población"))
with col2:
    c.mostrar(c.forest(orden, [D[k]["media"] for k in orden], [D[k]["ic"] for k in orden],
                       "Puntaje T por dimensión (IC 95 %)", referencia=50, alto=420))
st.caption("Las seis dimensiones están muy cerca de 50 y sus intervalos se solapan: no hay una "
           "dimensión claramente más débil en la población.")

c.seccion("Por grupo", "¿Cambia el perfil entre grupos?",
          "Azul: por encima de 50. Rojo: por debajo. Gris: cerca del centro.")
clave = st.segmented_control("Variable", list(comp), default=list(comp)[0],
                             format_func=lambda k: comp[k]["etiqueta"], key="dim_variable")
clave = clave or list(comp)[0]
grupos = comp[clave]["grupos"]
fig = c.heatmap_dimensiones(grupos, f"Puntaje T por dimensión y {comp[clave]['etiqueta'].lower()}")
if fig:
    c.mostrar(fig)
    con_dim = [k for k, g in grupos.items() if "dimensiones" in g]
    elegidos = st.multiselect("Comparar perfiles (hasta 3 grupos)", con_dim,
                              default=con_dim[:2], max_selections=3, key="dim_radar")
    if elegidos:
        c.mostrar(c.radar([(g, {d: v["media"] for d, v in grupos[g]["dimensiones"].items()})
                           for g in elegidos], "Perfiles comparados"))
else:
    st.info("Los resultados por grupo no están en esta versión de los hallazgos. Vuelva a "
            "generarlos con `python -m agent.reports.entrega2`.")
c.pie_pagina()
