# -*- coding: utf-8 -*-
"""Metodología: diseño, medición, estimación y limitaciones."""
import streamlit as st

import comun as c

H = c.H()
R = H["resumen"]

c.hero("Metodología", "Cómo se calcularon los resultados",
       "Los métodos detrás de cada cifra de la app y sus limitaciones.")

c.tarjetas([
    ("Población y muestra",
     f"N = {R['N']:,} estudiantes; n = {R['n']:,} respuestas válidas en 8 estratos (seccional × "
     "pregrado/posgrado), con factores de expansión calibrados por género."),
    ("Medición",
     "Escala de bienestar psicológico de Ryff (29 ítems, Likert 1–6, 6 dimensiones). Puntajes "
     "con el modelo de respuesta graduada: T = 50 + 10·θ."),
    ("Estimación",
     "Medias y proporciones ponderadas. Errores estándar por linealización de Taylor con "
     "corrección por población finita; deff = V(diseño) / V(MAS)."),
    ("Comparaciones",
     "Pruebas de Wald con la varianza del diseño y corrección de Bonferroni por comparaciones "
     "múltiples."),
    ("Privacidad",
     f"Solo agregados. No se muestran grupos con menos de {H['min_n_grupo']} casos, ni datos "
     "individuales."),
    ("El chat",
     "Usa las rutas del agente. Las cifras de la encuesta salen de estos mismos agregados, sin "
     "modelo de lenguaje; el corpus cita documento y página o se abstiene."),
])

c.seccion("Entrega 2", "Métodos de cada pregunta")
st.markdown("""
| Pregunta | Método |
|---|---|
| **2.0** Reproducibilidad | Comparación con las cifras reportadas, con tolerancia según el redondeo de cada una |
| **2.1** Intervalos | Varianza del diseño estratificado (linealización de Taylor) |
| **2.2** DIF | Regresión logística ordinal (Zumbo, 1999); ΔR² de Nagelkerke con los umbrales de Jodoin y Gierl (2001) |
| **2.3** Brecha ajustada | Regresión ponderada con errores estándar de diseño (Binder, 1983); modelos anidados |
| **2.4** Multinivel | Intercepto aleatorio (MixedLM, REML); prueba de τ² = 0 con mezcla de χ² (Self y Liang, 1987) |
""")

c.seccion("Limitaciones", "Lo que hay que tener en cuenta")
st.markdown(
    "- Los pesos calibrados se tratan como fijos: la varianza es algo conservadora.\n"
    "- El DIF y el modelo multinivel no usan factores de expansión.\n"
    "- La edad y el nivel de formación se solapan poco, así que el ajuste por edad extrapola en "
    "los extremos.\n"
    "- Es un estudio transversal: los resultados son asociaciones, no efectos causales.")
c.pie_pagina()
