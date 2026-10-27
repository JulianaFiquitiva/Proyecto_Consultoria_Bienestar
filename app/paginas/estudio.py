# -*- coding: utf-8 -*-
"""El estudio: contexto, escala, muestra y recorrido."""
import pandas as pd
import streamlit as st

import comun as c

H = c.H()
R = H["resumen"]
secc = H["comparaciones"].get("seccional", {}).get("grupos", {})
modal = H["comparaciones"].get("modalidad_exp", {}).get("grupos", {})

c.hero("Contexto", "El estudio de bienestar estudiantil",
       "Un estudio de la Universidad Santo Tomás que mide el bienestar psicológico de sus "
       "estudiantes de pregrado y posgrado, para estimar el nivel de toda la población "
       "estudiantil y compararlo entre seccionales, niveles de formación y grupos.")

c.kpis([
    ("Estudiantes en la población", f"{R['N']:,}", "N, según los marcos muestrales", "acento"),
    ("Respuestas válidas", f"{R['n']:,}", "n, después de la depuración", ""),
    ("Seccionales", str(len(secc) or 4), "Bogotá, Tunja, Bucaramanga, Villavicencio", ""),
    ("Afirmaciones de la escala", "29", "respuestas de 1 a 6", ""),
])

c.seccion("Qué mide", "La escala de bienestar psicológico de Ryff",
          "29 afirmaciones que se responden de 1 (totalmente en desacuerdo) a 6 (totalmente "
          "de acuerdo), agrupadas en seis dimensiones.")
c.tarjetas([
    ("Autoaceptación", "Actitud positiva hacia uno mismo y hacia la propia historia."),
    ("Relaciones positivas", "Vínculos cercanos, cálidos y de confianza con otras personas."),
    ("Autonomía", "Capacidad de decidir por uno mismo y resistir la presión social."),
    ("Dominio del entorno", "Sensación de manejar las exigencias de la vida cotidiana."),
    ("Propósito de vida", "Tener metas y sentir que la vida tiene dirección y sentido."),
    ("Crecimiento personal", "Sentir que uno sigue aprendiendo y desarrollando su potencial."),
])

c.seccion("A quiénes", "Una muestra que representa a toda la universidad",
          "La muestra se tomó por estratos: cada seccional, separada en pregrado y posgrado "
          "(8 estratos). Cada respuesta tiene un factor de expansión, que indica cuántos "
          "estudiantes de la población representa.")
col1, col2 = st.columns([3, 2])
with col1:
    fig = c.mapa_seccionales(secc)
    if fig:
        c.mostrar(fig)
        st.caption("El tamaño de cada punto es proporcional a los estudiantes que representa.")
with col2:
    # Tablas estáticas: son cortas y así se ven completas sin desplazamiento.
    if secc:
        st.table(pd.DataFrame([{"Seccional": k, "Respuestas": f"{g['n']:,}",
                                "Estudiantes representados": f"{g['N']:,}"}
                               for k, g in secc.items()]).set_index("Seccional"))
    if modal:
        st.table(pd.DataFrame([{"Nivel": k, "Respuestas": f"{g['n']:,}",
                                "Estudiantes representados": f"{g['N']:,}"}
                               for k, g in modal.items()]).set_index("Nivel"))

c.seccion("Cómo leer los resultados", "El puntaje T",
          "Cada estudiante recibe un puntaje calculado con un modelo de teoría de respuesta al "
          "ítem (modelo de respuesta graduada). Tiene centro 50 y desviación 10.")
c.tarjetas([
    ("En riesgo", "Puntaje T menor que 40: bienestar claramente por debajo del centro.", "T < 40"),
    ("Rango típico", "Entre 40 y 60: donde se ubica la mayoría de los estudiantes.", "40 – 60"),
    ("Bienestar alto", "Puntaje T de 60 o más: bienestar claramente por encima del centro.",
     "T ≥ 60"),
])

c.seccion("Recorrido", "De la encuesta a esta app")
hitos = [
    ("Encuesta", "Aplicación de la escala, con marcos muestrales de 2025 por seccional y "
                 "modalidad.", ""),
    ("Análisis original", "Consultorio de Estadística USTA (febrero de 2026): depuración, "
                          "factores de expansión, psicometría, puntajes T e inferencia.", ""),
    ("Entrega 1", "Agente «Nada sin fuente»: responde sobre la encuesta y 20 documentos "
                  "académicos, siempre con fuente; sin fuente, se abstiene.", ""),
    ("Entrega 2", "Estimaciones con el diseño muestral, reproducción del estudio anterior, DIF, "
                  "brecha ajustada, modelo multinivel y esta app.", ""),
    ("Entrega final", "Encuesta viva: respuestas nuevas que actualizan los resultados.", "futuro"),
]
st.markdown('<div class="linea">' + "".join(
    f'<div class="hito {clase}"><b>{t}</b><br><span>{d}</span></div>' for t, d, clase in hitos)
    + "</div>", unsafe_allow_html=True)
st.caption("Proyecto del curso Consultoría e Investigación (USTA): Juliana Fiquitiva y "
           "Luisa Martínez U.")
c.pie_pagina()
