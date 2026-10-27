# -*- coding: utf-8 -*-
"""Hallazgos de la entrega 2: las preguntas 2.0 a 2.4."""
import pandas as pd
import streamlit as st

import comun as c

H = c.H()
R = H["resumen"]

c.hero("Entrega 2", "Cinco preguntas de investigación",
       "Lo que aprendimos al revisar el estudio anterior con métodos para encuestas por "
       "muestreo.")

p20, p21, p22, p23, p24 = st.tabs(["2.0 Reproducibilidad", "2.1 Intervalos", "2.2 DIF",
                                   "2.3 Brecha ajustada", "2.4 Multinivel"])

with p20:
    filas = H["reproducibilidad"]
    col = "Estado vs. Referencia entrega 1"
    validadas = sum(1 for f in filas if f.get(col) == "VALIDADO")
    c.kpis([("Cifras reproducidas", f"{validadas} de {len(filas)}",
             "frente a la referencia de la entrega 1", "alto"),
            ("Informe Ejecutivo", "No se reproduce", "T = 51,2 y tamaños de muestra distintos",
             "riesgo")])
    c.mostrar(c.reproducibilidad(filas))
    st.markdown(
        "- Las cifras de referencia se reproducen **al ponderar** con los factores de expansión. "
        "Antes, el agente respondía con medias sin ponderar.\n"
        "- El **Informe Ejecutivo** trae cifras (T = 51,2; 19,5 % en riesgo; 29,6 % alto) y "
        "tamaños por seccional que no coinciden con la base: parece venir de otra versión de "
        "los datos. Queda por aclarar con la contraparte.")
    with st.expander("Ver la tabla"):
        tabla = pd.DataFrame(filas)
        st.dataframe(tabla.astype({k: str for k in tabla.columns if tabla[k].dtype == object}),
                     hide_index=True, width="stretch")

with p21:
    M = H["ic_metodos"]
    c.kpis([("Error estándar de diseño", f"{M[2]['se']:.3f}", "linealización de Taylor",
             "acento"),
            ("Efecto de diseño", f"{R['T_global']['deff']:.2f}",
             "el diseño infla la varianza ~10 %", ""),
            ("Media sin ponderar → ponderada", f"{M[0]['media']:.2f} → {M[2]['media']:.2f}",
             "ponderar cambia la estimación", "")])
    c.mostrar(c.forest([m["metodo"] for m in M], [m["media"] for m in M], [m["ic"] for m in M],
                       "Puntaje T global según el método (IC 95 %)", alto=300))

with p22:
    if "dif" not in H:
        st.warning("Los hallazgos se generaron sin DIF (`--sin-dif`).")
    else:
        dif = H["dif"]
        c.kpis([(f"ΔR² máximo · {r['comparacion']}", f"{r['delta_r2_maximo']:.4f}",
                 f"{r['items_significativos_bonferroni']} ítems significativos, ninguno relevante",
                 "alto") for r in dif.values()] +
               [("Umbral de DIF moderado", "0.035", "Jodoin y Gierl (2001)", "")])
        st.markdown("Ningún ítem supera el umbral: **significativo no es lo mismo que "
                    "relevante**. La escala permite comparar estos grupos.")
        col_a, col_b = st.columns(2)
        for columna, clave in zip((col_a, col_b), dif):
            with columna:
                c.mostrar(c.dif_barras(dif[clave]))

with p23:
    B = H["brecha"]
    edad = B["edad_media_ponderada"]
    c.kpis([("Brecha bruta", f"{B['brecha_bruta']:+.2f}", "puntos T, posgrado − pregrado", ""),
            ("Brecha ajustada", f"{B['brecha_ajustada']:+.2f}", "controlando edad, género, "
             "estrato y seccional", "acento"),
            ("Explicado por los controles", f"{B['porcentaje_explicado_por_controles']:.0f} %",
             "sobre todo por la edad", ""),
            ("Edad media", f"{edad['Posgrado']:.0f} vs {edad['Pregrado']:.0f}",
             "años: posgrado vs pregrado", "")])
    col1, col2 = st.columns(2)
    with col1:
        c.mostrar(c.cascada_brecha(B))
    with col2:
        c.mostrar(c.forest([m["modelo"] for m in B["modelos"]], [m["brecha"] for m in B["modelos"]],
                           [[m["ic_inferior"], m["ic_superior"]] for m in B["modelos"]],
                           "Brecha en cada modelo (IC 95 %)", referencia=0, color=c.NARANJA,
                           alto=360))
    c.mostrar(c.coeficientes(B))
    st.caption("Regresión con errores estándar de diseño (Binder, 1983). Es una asociación, no "
               "un efecto causal.")

with p24:
    ML = H["multinivel"]
    prog = ML["programa"]
    c.kpis([("Programas", str(prog["nulo"]["n_grupos"]),
             f"{prog['nulo']['grupos_con_menos_de_10']} con menos de 10 respuestas", ""),
            ("Varianza entre programas", f"{prog['nulo']['icc'] * 100:.1f} %", "sin controles",
             "acento"),
            ("Controlando nivel y edad", f"{prog['condicional_nivel_edad']['icc'] * 100:.1f} %",
             "más del 95 % ocurre dentro del programa", "alto")])
    cols = st.columns(4)
    donas = [("programa", "nulo", "Programa · sin controles"),
             ("programa", "condicional_nivel_edad", "Programa · con controles"),
             ("nbc", "nulo", "Área (NBC) · sin controles"),
             ("nbc", "condicional_nivel_edad", "Área (NBC) · con controles")]
    for col, (k, m, titulo) in zip(cols, donas):
        if k in ML:
            with col:
                c.mostrar(c.dona_varianza(ML[k][m]["icc"], titulo))
    fig = c.efectos_programa(prog["condicional_nivel_edad"],
                             "Programas con más y menos bienestar (n ≥ 30, con controles)")
    if fig:
        c.mostrar(fig)
    st.caption(f"{ML.get('limitacion', '')} Los efectos son pequeños frente a la variación entre "
               "estudiantes: no se usan como ranking.")
c.pie_pagina()
