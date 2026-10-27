# -*- coding: utf-8 -*-
"""Metodología: diseño, medición, estimación y limitaciones."""
import pandas as pd
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

c.seccion("Entrega 1", "Cómo elige el método el agente",
          "El núcleo del proyecto: ante una pregunta sobre los datos, el agente decide el "
          "análisis y el método, en lugar de aplicar siempre la misma prueba.")
st.markdown('<div class="linea">' + "".join(
    f'<div class="hito"><b>{t}</b><br><span>{d}</span></div>' for t, d in [
        ("1. Ámbito", "¿La pregunta es sobre la encuesta, sobre la literatura o está fuera "
                      "del estudio? Si está fuera o es ambigua, se abstiene."),
        ("2. Análisis", "Reconoce qué se pide: resumen, comparación, correlación, regresión, "
                        "perfil por dimensiones o validación."),
        ("3. Variables", "Identifica la variable de respuesta (puntaje T) y la de agrupación, "
                         "comprueba que existan y que cada grupo tenga al menos 30 casos."),
        ("4. Método", "Revisa los supuestos (normalidad con Shapiro-Wilk, varianzas con "
                      "Levene) y elige la prueba: t de Student o de Welch, Mann-Whitney, "
                      "ANOVA o Kruskal-Wallis. Usa los factores de expansión."),
        ("5. Interpretación", "Explica el resultado con el tamaño del efecto, el intervalo de "
                              "confianza y las advertencias."),
    ]) + "</div>", unsafe_allow_html=True)

comp = H["comparaciones"]
filas = [{"Comparación": d["etiqueta"], "Grupos": len(d["grupos"]),
          "Prueba elegida": d["seleccion"]["prueba"], "Por qué": d["seleccion"]["razon"],
          "p": c.valor_p(d["seleccion"]["resultado"]["p"])
               if d["seleccion"].get("resultado") else "—"}
         for d in comp.values() if d.get("seleccion")]
if filas:
    st.markdown("**Lo que eligió el agente en cada comparación de la app**")
    st.table(pd.DataFrame(filas).set_index("Comparación"))
    st.caption("En la entrega 2 estas pruebas se complementan con la varianza del diseño "
               "muestral (Wald + Bonferroni), que es lo que muestran las cifras de la app.")

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
