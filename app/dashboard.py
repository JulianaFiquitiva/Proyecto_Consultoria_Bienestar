# -*- coding: utf-8 -*-
"""
Dashboard del Índice de Bienestar Estudiantil USTA (entrega 2).

    streamlit run app/dashboard.py

Lee resultados/entrega2/hallazgos_entrega2.json (generado con
`python -m agent.reports.entrega2`). Si no existe, lo calcula desde data/.
Solo muestra agregados; los grupos con menos de 30 casos no se muestran.
No usa OpenAI ni internet, y no escribe en data/.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.reports import entrega2  # noqa: E402

AZUL = "#1B3A5C"
AZUL_CLARO = "#4A90D9"
NARANJA = "#E8913A"
GRIS = "#8A94A0"
ESTADO_COLOR = {"VALIDADO": "#2E7D32", "DIFERENCIA MENOR": "#B26A00",
                "DIFERENCIA IMPORTANTE": "#C62828", "—": GRIS}

st.set_page_config(page_title="Índice de Bienestar USTA", page_icon="📊",
                   layout="wide")


@st.cache_data(show_spinner=False)
def _hallazgos():
    guardados = entrega2.cargar()
    if guardados is not None:
        return guardados, "precalculados"
    from agent.data.loader import DataLoader
    loader = DataLoader()
    return entrega2.calcular(loader.load_prepared(), loader.load_irt()), "calculados ahora"


def _layout(fig, alto=380, **kw):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13), **kw)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def _forest(etiquetas, medias, ic, titulo="", referencia=None, color=AZUL, alto=None):
    """Puntos con intervalo de confianza, uno por fila."""
    fig = go.Figure(go.Scatter(
        x=medias, y=etiquetas, mode="markers",
        marker=dict(size=11, color=color),
        error_x=dict(type="data", symmetric=False,
                     array=[s - m for m, (_, s) in zip(medias, ic)],
                     arrayminus=[m - i for m, (i, _) in zip(medias, ic)],
                     thickness=2, width=6, color=color),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    if referencia is not None:
        fig.add_vline(x=referencia, line_dash="dash", line_color=GRIS)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, alto or max(220, 60 * len(etiquetas)), title=titulo)


def _ic(e, dec=1):
    return f"IC 95 %: {e['ic'][0]:.{dec}f} – {e['ic'][1]:.{dec}f}"


# ── Encabezado ─────────────────────────────────────────────────────────
try:
    H, origen = _hallazgos()
except FileNotFoundError:
    st.error("No hay hallazgos precalculados ni datos en data/. Ejecute "
             "`python -m agent.reports.entrega2` en un equipo con los datos.")
    st.stop()

R = H["resumen"]
st.title("Índice de Bienestar Estudiantil USTA")
st.caption(f"Escala de Ryff · n = {R['n']:,} estudiantes que representan a "
           f"N = {R['N']:,} · Estimaciones poblacionales con el diseño muestral "
           f"estratificado · Resultados {origen} ({H['generado'][:10]})")

tab_res, tab_comp, tab_dim, tab_hall, tab_met = st.tabs(
    ["Resumen", "Comparaciones", "Dimensiones", "Hallazgos entrega 2", "Metodología"])

# ── Resumen ────────────────────────────────────────────────────────────
with tab_res:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("T global poblacional", f"{R['T_global']['media']:.1f}", help=_ic(R["T_global"], 2))
    c1.caption(_ic(R["T_global"], 2))
    c2.metric("En riesgo (T < 40)", f"{R['riesgo']['media'] * 100:.1f} %")
    c2.caption(f"IC 95 %: {R['riesgo']['ic'][0] * 100:.1f} – {R['riesgo']['ic'][1] * 100:.1f} %")
    c3.metric("Bienestar alto (T ≥ 60)", f"{R['alto']['media'] * 100:.1f} %")
    c3.caption(f"IC 95 %: {R['alto']['ic'][0] * 100:.1f} – {R['alto']['ic'][1] * 100:.1f} %")
    c4.metric("Efecto de diseño", f"{R['T_global']['deff']:.2f}")
    c4.caption(f"n efectivo ≈ {R['n'] / R['T_global']['deff']:,.0f}")

    bordes = R["histograma_bordes"]
    centros = [(a + b) / 2 for a, b in zip(bordes[:-1], bordes[1:])]
    colores = [NARANJA if c < 40 else (AZUL_CLARO if c >= 60 else AZUL) for c in centros]
    fig = go.Figure(go.Bar(x=centros, y=R["histograma"], marker_color=colores, width=3.6,
                           hovertemplate="T ≈ %{x}: %{y:,.0f} estudiantes<extra></extra>"))
    for x, texto in ((40, "Riesgo"), (60, "Alto")):
        fig.add_vline(x=x, line_dash="dot", line_color=GRIS,
                      annotation_text=texto, annotation_position="top")
    _layout(fig, 360, title="Distribución poblacional del T-score global",
            xaxis_title="T-score global", yaxis_title="Estudiantes (población estimada)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"La media muestral sin ponderar es {R['media_sin_ponderar']:.2f}: "
               "ponderar corrige la sobre- o subrepresentación de cada estrato.")

# ── Comparaciones ──────────────────────────────────────────────────────
with tab_comp:
    C = H["comparaciones"]
    variable = st.selectbox("Comparar por", list(C), format_func=lambda k: C[k]["etiqueta"])
    datos = C[variable]
    grupos = datos["grupos"]
    st.plotly_chart(_forest(list(grupos), [g["media"] for g in grupos.values()],
                            [g["ic"] for g in grupos.values()],
                            "T-score global poblacional (IC 95 %)",
                            referencia=R["T_global"]["media"]),
                    width="stretch")
    st.dataframe(pd.DataFrame([{"Grupo": k, "n": g["n"], "N estimado": g["N"],
                                "T": g["media"], "IC inf.": g["ic"][0], "IC sup.": g["ic"][1]}
                               for k, g in grupos.items()]),
                 hide_index=True, width="stretch")
    if datos["excluidos"]["categorias"]:
        st.info(f"No se muestran {datos['excluidos']['categorias']} categoría(s) con menos "
                f"de {H['min_n_grupo']} casos ({datos['excluidos']['casos']} estudiantes): "
                "la estimación sería inestable y podría identificar personas.")
    st.subheader("Diferencias por pares")
    st.caption("Prueba de Wald con la varianza del diseño; p ajustado por Bonferroni.")
    pares = pd.DataFrame(datos["pares"])
    if not pares.empty:
        pares = pares[["grupo_1", "grupo_2", "diferencia", "ic_inferior", "ic_superior",
                       "p_valor", "p_ajustado_bonferroni", "significativo"]]
        pares.columns = ["Grupo 1", "Grupo 2", "Diferencia", "IC inf.", "IC sup.",
                         "p", "p ajustado", "Significativa"]
        st.dataframe(pares, hide_index=True, width="stretch")

# ── Dimensiones ────────────────────────────────────────────────────────
with tab_dim:
    D = H["dimensiones"]
    orden = sorted(D, key=lambda k: -D[k]["media"])
    st.plotly_chart(_forest(orden, [D[k]["media"] for k in orden], [D[k]["ic"] for k in orden],
                            "T-score poblacional por dimensión de Ryff (IC 95 %)", referencia=50),
                    width="stretch")
    st.caption("Los intervalos se solapan: las seis dimensiones están muy cerca de la "
               "media normativa (T = 50) y no hay una dimensión claramente más débil.")

# ── Hallazgos entrega 2 ────────────────────────────────────────────────
with tab_hall:
    st.markdown("Cinco preguntas de investigación sobre el estudio anterior.")

    with st.expander("2.0 · ¿Se reproduce el estudio anterior?", expanded=True):
        rep = pd.DataFrame(H["reproducibilidad"])
        st.dataframe(rep.style.map(lambda v: f"color: {ESTADO_COLOR.get(v, 'inherit')}",
                                   subset=[c for c in rep.columns if c.startswith("Estado")]),
                     hide_index=True, width="stretch")
        st.markdown(
            "- Las cifras de referencia se reproducen **al ponderar** con los factores de "
            "expansión. Antes el agente respondía con medias sin ponderar.\n"
            "- El **Informe Ejecutivo** no se reproduce con los datos disponibles "
            "(T = 51,2; 19,5 % en riesgo; 29,6 % alto) y trae tamaños por seccional "
            "distintos a los de la base. Hay que aclarar con la contraparte qué versión "
            "de los datos lo generó.")

    with st.expander("2.1 · ¿Los intervalos de confianza son correctos?"):
        M = H["ic_metodos"]
        st.plotly_chart(_forest([m["metodo"] for m in M], [m["media"] for m in M],
                                [m["ic"] for m in M], "T global según el método (IC 95 %)",
                                color=AZUL_CLARO, alto=260), width="stretch")
        st.markdown(
            f"Ponderar mueve la media de {M[0]['media']:.2f} a {M[2]['media']:.2f}. "
            f"La varianza del diseño (8 estratos, linealización de Taylor) da un error "
            f"estándar de {M[2]['se']:.3f} y un deff de {R['T_global']['deff']:.2f}.")

    with st.expander("2.2 · ¿La escala mide igual a todos los grupos? (DIF)"):
        if "dif" not in H:
            st.warning("Los hallazgos se generaron sin DIF (`--sin-dif`).")
        else:
            col_a, col_b = st.columns(2)
            for columna, clave in ((col_a, "genero"), (col_b, "nivel")):
                r = H["dif"][clave]
                tabla = pd.DataFrame(r["tabla"]).sort_values("delta_r2")
                fig = go.Figure(go.Bar(x=tabla["delta_r2"], y=tabla["item"], orientation="h",
                                       marker_color=AZUL_CLARO,
                                       hovertemplate="%{y}: ΔR² = %{x:.4f}<extra></extra>"))
                fig.add_vline(x=0.035, line_dash="dash", line_color=NARANJA,
                              annotation_text="moderado", annotation_position="top")
                _layout(fig, 640, title=r["comparacion"], xaxis_title="ΔR² de Nagelkerke",
                        xaxis_range=[0, 0.045])
                columna.plotly_chart(fig, width="stretch")
                columna.caption(f"{r['items_significativos_bonferroni']} ítems significativos "
                                f"con Bonferroni; ninguno supera ΔR² = 0,035. {r['conclusion']}")

    with st.expander("2.3 · ¿La brecha pregrado–posgrado se explica por la edad?"):
        B = H["brecha"]
        st.plotly_chart(_forest([m["modelo"] for m in B["modelos"]],
                                [m["brecha"] for m in B["modelos"]],
                                [[m["ic_inferior"], m["ic_superior"]] for m in B["modelos"]],
                                "Brecha posgrado − pregrado en puntos T (IC 95 %)",
                                referencia=0, color=NARANJA, alto=260),
                        width="stretch")
        edad = B["edad_media_ponderada"]
        st.markdown(
            f"La brecha bruta de **{B['brecha_bruta']:.1f}** puntos baja a "
            f"**{B['brecha_ajustada']:.1f}** al controlar edad, género, estrato y seccional: "
            f"los controles explican el {B['porcentaje_explicado_por_controles']:.0f} %. "
            f"El posgrado tiene en promedio {edad['Posgrado']:.0f} años y el pregrado "
            f"{edad['Pregrado']:.0f}. La brecha ajustada sigue siendo significativa.")

    with st.expander("2.4 · ¿Cuánto depende el bienestar del programa?"):
        ML = H["multinivel"]
        filas = []
        for clave, nombre in (("programa", "Programa"), ("nbc", "NBC")):
            if clave in ML:
                filas.append((nombre, "Sin controles", ML[clave]["nulo"]["icc"]))
                filas.append((nombre, "Controlando nivel y edad",
                              ML[clave]["condicional_nivel_edad"]["icc"]))
        df_icc = pd.DataFrame(filas, columns=["Agrupación", "Modelo", "ICC"])
        fig = go.Figure()
        for modelo, color in (("Sin controles", AZUL), ("Controlando nivel y edad", AZUL_CLARO)):
            sub = df_icc[df_icc["Modelo"] == modelo]
            fig.add_bar(x=sub["Agrupación"], y=sub["ICC"] * 100, name=modelo, marker_color=color,
                        text=[f"{v * 100:.1f} %" for v in sub["ICC"]], textposition="outside")
        _layout(fig, 340, title="Porcentaje de la varianza del bienestar entre grupos (ICC)",
                yaxis_title="%", barmode="group", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        st.markdown("Más del 95 % de la variación del bienestar ocurre **dentro** de cada "
                    "programa. La mitad de las diferencias entre programas se explica por "
                    "la edad y el nivel de sus estudiantes.")
        st.caption(ML.get("limitacion", ""))

# ── Metodología ────────────────────────────────────────────────────────
with tab_met:
    st.markdown(f"""
**Población y muestra.** N = {R['N']:,} estudiantes; n = {R['n']:,} respuestas válidas,
en 8 estratos (seccional × pregrado/posgrado) con factores de expansión calibrados por género.

**Medición.** Escala de bienestar psicológico de Ryff (29 ítems, Likert 1–6, 6 dimensiones).
Puntajes por Modelo de Respuesta Graduada: T = 50 + 10·θ.

**Estimación.** Medias y proporciones ponderadas. Errores estándar por linealización de
Taylor con corrección por población finita; deff = V(diseño) / V(MAS).
Comparaciones con pruebas de Wald y corrección de Bonferroni.

**Preguntas de la entrega 2.**
- 2.0: reproducción de las cifras reportadas (tolerancia según el redondeo reportado).
- 2.1: intervalos con la varianza del diseño.
- 2.2: DIF por regresión logística ordinal (Zumbo, 1999); ΔR² de Nagelkerke con los
  umbrales de Jodoin y Gierl (2001).
- 2.3: regresión ponderada con errores estándar de diseño (Binder, 1983), modelos anidados.
- 2.4: modelo multinivel de intercepto aleatorio; prueba de τ² = 0 con mezcla de χ².

**Privacidad.** Solo agregados; no se muestran grupos con menos de {H['min_n_grupo']} casos.

**Limitaciones.** Los pesos calibrados se tratan como fijos (varianza algo conservadora).
El DIF y el modelo multinivel no usan factores de expansión. La edad y el nivel de
formación se solapan poco, así que el ajuste por edad extrapola en los extremos.
""")
