# -*- coding: utf-8 -*-
"""Inicio: resumen ejecutivo del estudio."""
import streamlit as st

import comun as c

H = c.H()
R = H["resumen"]
t = R["T_global"]
B = H["brecha"]
edad = B.get("edad_media_ponderada", {})

c.hero("Universidad Santo Tomás · Índice de Bienestar Estudiantil",
       "¿Cómo está el bienestar de los estudiantes?",
       f"Estimaciones para los {R['N']:,} estudiantes de la universidad, a partir de "
       f"{R['n']:,} respuestas a la escala de bienestar psicológico de Ryff.")

c.kpis([
    ("Puntaje T poblacional", f"{t['media']:.1f}", c.ic(t, 2), "acento"),
    ("En riesgo (T < 40)", f"{R['riesgo']['media'] * 100:.1f} %", c.ic(R["riesgo"], pct=True),
     "riesgo"),
    ("Bienestar alto (T ≥ 60)", f"{R['alto']['media'] * 100:.1f} %", c.ic(R["alto"], pct=True),
     "alto"),
    ("Respuestas válidas", f"{R['n']:,}", f"de N = {R['N']:,} estudiantes", ""),
])

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("##### Puntaje T de la población")
    c.mostrar(c.gauge_t(R))
    st.caption("Centro de la escala: 50. Bajo 40 = en riesgo; desde 60 = bienestar alto.")
with col2:
    st.markdown("##### ¿Cómo se reparten los estudiantes?")
    c.mostrar(c.dona_niveles(R["riesgo"]["media"], R["alto"]["media"]))

c.seccion("Lo más importante", "Cinco hallazgos en una mirada")
comp = H["comparaciones"]
genero = comp.get("Género", {}).get("grupos", {})
ml = H["multinivel"]["programa"]
dif_max = max((r["delta_r2_maximo"] for r in H.get("dif", {}).values()), default=None)
items = [
    ("La población está en el centro de la escala",
     f"T = {t['media']:.1f}: ni por encima ni por debajo del punto de referencia.",
     f"{t['media']:.1f}"),
    ("El posgrado reporta más bienestar",
     f"La brecha de {B['brecha_bruta']:.1f} puntos baja a {B['brecha_ajustada']:.1f} al "
     f"tener en cuenta la edad: los de posgrado son, en promedio, "
     f"{edad.get('Posgrado', 0) - edad.get('Pregrado', 0):.0f} años mayores.",
     f"+{B['brecha_ajustada']:.1f}"),
    ("El programa pesa poco",
     f"Solo el {ml['condicional_nivel_edad']['icc'] * 100:.1f} % de la variación está entre "
     f"programas; el resto ocurre entre estudiantes del mismo programa.",
     f"{ml['condicional_nivel_edad']['icc'] * 100:.1f} %"),
]
if len(genero) >= 2:
    m, f = genero.get("Masculino"), genero.get("Femenino")
    if m and f:
        items.append(("Diferencia pequeña por género",
                      f"Hombres {m['media']:.1f} y mujeres {f['media']:.1f}: significativa, "
                      f"pero de apenas un punto.", f"{m['media'] - f['media']:+.1f}"))
if dif_max is not None:
    items.append(("La escala mide igual a todos",
                  "Ningún ítem funciona distinto por género o nivel: las comparaciones son "
                  "válidas.", f"ΔR² {dif_max:.3f}"))
c.tarjetas(items)

st.info("Explore las páginas de arriba para ver los resultados por grupo y por dimensión, o "
        "vaya a **Pregúntale al agente** para consultar en lenguaje natural.", icon=":material/lightbulb:")
c.pie_pagina()
