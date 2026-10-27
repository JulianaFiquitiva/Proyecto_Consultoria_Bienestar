# -*- coding: utf-8 -*-
"""Comparaciones por grupo con la varianza del diseño."""
import pandas as pd
import streamlit as st

import comun as c

H = c.H()
R = H["resumen"]
comp = H["comparaciones"]

c.hero("Comparaciones", "¿Hay diferencias entre grupos?",
       "Medias poblacionales con su intervalo de confianza. Las diferencias se prueban con "
       "la varianza del diseño muestral y la corrección de Bonferroni.")

clave = st.segmented_control("Comparar por", list(comp), default=list(comp)[0],
                             format_func=lambda k: comp[k]["etiqueta"], key="comp_variable")
clave = clave or list(comp)[0]
datos = comp[clave]
grupos = datos["grupos"]
sig = [p for p in datos["pares"] if p["significativo"]]

c.kpis([
    ("Grupos comparados", str(len(grupos)), f"con al menos {H['min_n_grupo']} respuestas",
     "acento"),
    ("Diferencias significativas", f"{len(sig)} de {len(datos['pares'])}",
     "tras la corrección de Bonferroni", ""),
    ("Mayor puntaje", max(grupos, key=lambda k: grupos[k]["media"]),
     f"T = {max(g['media'] for g in grupos.values()):.1f}", "alto"),
    ("Menor puntaje", min(grupos, key=lambda k: grupos[k]["media"]),
     f"T = {min(g['media'] for g in grupos.values()):.1f}", "riesgo"),
])

col1, col2 = st.columns([3, 2])
with col1:
    c.mostrar(c.forest(list(grupos), [g["media"] for g in grupos.values()],
                       [g["ic"] for g in grupos.values()],
                       f"Puntaje T por {datos['etiqueta'].lower()} (IC 95 %)",
                       referencia=R["T_global"]["media"]))
    st.caption("La línea punteada es el promedio de toda la población.")
with col2:
    if clave == "seccional":
        fig = c.mapa_seccionales(grupos)
        if fig:
            c.mostrar(fig)
    else:
        c.mostrar(c.peso_grupos(grupos))

fig = c.barras_niveles(grupos, f"Niveles de bienestar por {datos['etiqueta'].lower()}")
if fig:
    c.mostrar(fig)

c.seccion("Pruebas", "Diferencias por pares")
if not datos["pares"]:
    st.info("No hay suficientes grupos con 30 casos o más para comparar.")
else:
    if sig:
        for p in sig:
            alto, bajo = ((p["grupo_1"], p["grupo_2"]) if p["diferencia"] > 0
                          else (p["grupo_2"], p["grupo_1"]))
            st.success(f"**{alto}** supera a **{bajo}** por {abs(p['diferencia']):.2f} puntos "
                       f"(p ajustado = {p['p_ajustado_bonferroni']:.3f}).",
                       icon=":material/check_circle:")
    else:
        st.info("Ninguna diferencia es estadísticamente significativa tras Bonferroni.",
                icon=":material/info:")
    with st.expander("Ver la tabla completa de comparaciones"):
        pares = pd.DataFrame(datos["pares"])[["grupo_1", "grupo_2", "diferencia", "ic_inferior",
                                               "ic_superior", "p_valor", "p_ajustado_bonferroni",
                                               "significativo"]]
        pares.columns = ["Grupo 1", "Grupo 2", "Diferencia", "IC inf.", "IC sup.", "p",
                         "p ajustado", "Significativa"]
        st.dataframe(pares, hide_index=True, width="stretch")

with st.expander("Ver los datos de cada grupo"):
    st.dataframe(pd.DataFrame([{"Grupo": k, "n": g["n"], "N estimado": g["N"], "T": g["media"],
                                "IC inf.": g["ic"][0], "IC sup.": g["ic"][1],
                                "% riesgo": round(g["riesgo"]["media"] * 100, 1) if "riesgo" in g else None,
                                "% alto": round(g["alto"]["media"] * 100, 1) if "alto" in g else None}
                               for k, g in grupos.items()]), hide_index=True, width="stretch")
if datos["excluidos"]["categorias"]:
    st.caption(f"No se muestran {datos['excluidos']['categorias']} categoría(s) con menos de "
               f"{H['min_n_grupo']} casos ({datos['excluidos']['casos']} estudiantes): la "
               "estimación sería inestable y podría identificar personas.")
c.pie_pagina()
