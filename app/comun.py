# -*- coding: utf-8 -*-
"""
Piezas compartidas por las páginas de la app: carga de los hallazgos, el
motor del chat, el estilo y los gráficos.

Colores: la marca (azul USTA) viste la interfaz; los datos usan la paleta
categórica validada (azul, naranja, aqua, en ese orden) y, para los niveles
de bienestar, un divergente rojo-gris-azul con la etiqueta siempre escrita.
"""
import os
from html import escape

import plotly.graph_objects as go
import streamlit as st

from agent.core.chat import MotorChat
from agent.reports import entrega2

# ── Paleta ────────────────────────────────────────────────────────────
MARCA = "#1B3A5C"          # interfaz: títulos, encabezados
MARCA_CLARA = "#E8EEF6"
SERIE = ["#2a78d6", "#eb6834", "#1baf7a"]     # categórica, orden fijo
AZUL, NARANJA, AQUA = SERIE
RIESGO, TIPICO, ALTO = "#e34948", "#c3c2b7", "#2a78d6"
TINTA, TINTA_2, TINTA_MUTED = "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE = "#e1e0d9", "#c3c2b7"
# divergente alrededor de T = 50 (rojo = debajo, azul = encima, gris = neutro)
DIVERGENTE = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]

COORDENADAS = {"Bogotá": (4.711, -74.072), "Tunja": (5.535, -73.367),
               "Bucaramanga": (7.119, -73.122), "Villavicencio": (4.142, -73.626)}


# ── Configuración ─────────────────────────────────────────────────────
def modo() -> str:
    valor = os.environ.get("INDICE_MODO", "auto").strip().lower()
    if valor in ("publico", "completo"):
        return valor
    from agent.core.config import AgentConfig
    cfg = AgentConfig()
    return "completo" if cfg.get_data_path(cfg.irt_data_file).exists() else "publico"


def max_consultas() -> int:
    """Consultas por sesión que usan servicios externos (se lee en cada ejecución)."""
    return int(os.environ.get("CHAT_MAX_CONSULTAS", "20"))


@st.cache_data(show_spinner=False)
def hallazgos(modo_app: str):
    guardados = entrega2.cargar()
    if guardados is not None:
        return guardados
    if modo_app == "publico":
        raise FileNotFoundError(entrega2.SALIDA)
    from agent.data.loader import DataLoader
    loader = DataLoader()
    return entrega2.calcular(loader.load_prepared(), loader.load_irt())


@st.cache_resource(show_spinner="Preparando el agente…")
def motor(modo_app: str) -> MotorChat:
    """Un solo motor por proceso: el agente y el índice se cargan una vez."""
    from agent.core.orchestrator import StatisticalAgent
    agente = StatisticalAgent()
    if modo_app == "completo":
        agente.load_data()
    return MotorChat(hallazgos(modo_app), modo=modo_app, agente=agente)


def H():
    return hallazgos(modo())


# ── Estilo ────────────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
/* Solo texto: los íconos (Material Symbols) conservan su propia fuente */
html, body, .stMarkdown, .stMarkdown p, .stMarkdown li, h1, h2, h3, h4, h5,
label, .stCaption, button p {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.block-container {{ padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1240px; }}
h1, h2, h3 {{ color: {MARCA}; letter-spacing: -0.01em; }}
.hero {{
  background: linear-gradient(135deg, {MARCA} 0%, #24507f 60%, #2a78d6 100%);
  color: #fff; border-radius: 20px; padding: 2.4rem 2.6rem; margin-bottom: 1.6rem;
  box-shadow: 0 10px 30px rgba(27,58,92,.18);
}}
.hero .eyebrow {{ text-transform: uppercase; letter-spacing: .12em; font-size: .78rem;
  font-weight: 600; color: #bcd3ee; }}
.hero h1 {{ color: #fff; font-size: 2.3rem; margin: .35rem 0 .6rem; line-height: 1.15; }}
.hero p {{ color: #dbe7f5; font-size: 1.05rem; max-width: 760px; margin: 0; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 1rem; margin: .4rem 0 1.4rem; }}
.kpi {{ background: #fff; border: 1px solid rgba(11,11,11,.08); border-radius: 16px;
  padding: 1.1rem 1.25rem; box-shadow: 0 2px 10px rgba(27,58,92,.06); }}
.kpi .etq {{ color: {TINTA_2}; font-size: .85rem; font-weight: 500; }}
.kpi .valor {{ color: {MARCA}; font-size: 2rem; font-weight: 700; line-height: 1.2;
  margin-top: .2rem; }}
.kpi .det {{ color: {TINTA_MUTED}; font-size: .8rem; margin-top: .15rem; }}
.kpi.acento {{ border-top: 4px solid {AZUL}; }}
.kpi.riesgo {{ border-top: 4px solid {RIESGO}; }}
.kpi.alto {{ border-top: 4px solid {ALTO}; }}
.tarjetas {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem; margin: .4rem 0 1.2rem; }}
.tarjeta {{ background: #fff; border: 1px solid rgba(11,11,11,.08); border-radius: 16px;
  padding: 1.1rem 1.25rem; }}
.tarjeta h4 {{ margin: 0 0 .35rem; color: {MARCA}; font-size: 1.02rem; }}
.tarjeta p {{ margin: 0; color: {TINTA_2}; font-size: .92rem; line-height: 1.45; }}
.tarjeta .num {{ color: {AZUL}; font-weight: 700; font-size: 1.5rem; }}
.seccion {{ margin: 2rem 0 .6rem; }}
.seccion .eyebrow {{ text-transform: uppercase; letter-spacing: .1em; font-size: .75rem;
  font-weight: 600; color: {AZUL}; }}
.seccion h2 {{ margin: .15rem 0 .3rem; font-size: 1.55rem; }}
.seccion p {{ color: {TINTA_2}; margin: 0; max-width: 820px; }}
.linea {{ border-left: 3px solid {MARCA_CLARA}; margin: .6rem 0 0 .4rem; padding-left: 1.2rem; }}
.hito {{ position: relative; padding: .2rem 0 1.1rem; }}
.hito:before {{ content: ''; position: absolute; left: -1.62rem; top: .45rem; width: 12px;
  height: 12px; border-radius: 50%; background: {AZUL}; border: 3px solid #fff;
  box-shadow: 0 0 0 2px {AZUL}; }}
.hito.futuro:before {{ background: #fff; }}
.hito b {{ color: {MARCA}; }} .hito span {{ color: {TINTA_2}; font-size: .92rem; }}
.pie {{ color: {TINTA_MUTED}; font-size: .8rem; margin-top: 2.5rem; border-top: 1px solid
  {REJILLA}; padding-top: .8rem; }}
div[data-testid="stChatMessage"] {{ border-radius: 14px; }}
</style>
"""


def estilo():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, titulo: str, texto: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">{escape(eyebrow)}</div>'
                f'<h1>{escape(titulo)}</h1><p>{texto}</p></div>', unsafe_allow_html=True)


def kpis(items):
    """items: (etiqueta, valor, detalle, clase)"""
    html = "".join(f'<div class="kpi {c}"><div class="etq">{escape(e)}</div>'
                   f'<div class="valor">{escape(v)}</div><div class="det">{escape(d)}</div></div>'
                   for e, v, d, c in items)
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def tarjetas(items):
    """items: (titulo, texto) o (titulo, texto, numero)"""
    partes = []
    for it in items:
        num = f'<div class="num">{escape(it[2])}</div>' if len(it) > 2 else ""
        partes.append(f'<div class="tarjeta">{num}<h4>{escape(it[0])}</h4><p>{it[1]}</p></div>')
    st.markdown(f'<div class="tarjetas">{"".join(partes)}</div>', unsafe_allow_html=True)


def seccion(eyebrow: str, titulo: str, texto: str = ""):
    p = f"<p>{texto}</p>" if texto else ""
    st.markdown(f'<div class="seccion"><div class="eyebrow">{escape(eyebrow)}</div>'
                f'<h2>{escape(titulo)}</h2>{p}</div>', unsafe_allow_html=True)


def pie_pagina():
    h = H()
    st.markdown(f'<div class="pie">Índice de Bienestar Estudiantil USTA · Estimaciones '
                f'poblacionales con el diseño muestral estratificado · Resultados generados el '
                f'{h["generado"][:10]} · Solo agregados: no se muestran grupos con menos de '
                f'{h["min_n_grupo"]} casos.</div>', unsafe_allow_html=True)


def ic(e, dec=1, pct=False):
    if pct:
        return f"IC 95 %: {e['ic'][0] * 100:.{dec}f} – {e['ic'][1] * 100:.{dec}f} %"
    return f"IC 95 %: {e['ic'][0]:.{dec}f} – {e['ic'][1]:.{dec}f}"


def valor_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ── Gráficos ──────────────────────────────────────────────────────────
def layout(fig, alto=380, **kw):
    kw.setdefault("margin", dict(l=10, r=10, t=50, b=10))
    # Texto explícito: sin él, Plotly en Streamlit dibuja "undefined" como título.
    titulo = kw.pop("title", "") or ""
    fig.update_layout(height=alto,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui, sans-serif", size=13, color=TINTA_2),
                      title=dict(text=titulo, font=dict(size=15, color=TINTA)),
                      hoverlabel=dict(bgcolor="#fff", font_size=13), **kw)
    fig.update_xaxes(gridcolor=REJILLA, linecolor=EJE, zeroline=False)
    fig.update_yaxes(gridcolor=REJILLA, linecolor=EJE, zeroline=False)
    return fig


def mostrar(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def forest(etiquetas, medias, ics, titulo="", referencia=None, color=AZUL, alto=None,
           rango=None):
    """Punto con su intervalo de confianza, uno por fila."""
    fig = go.Figure(go.Scatter(
        x=medias, y=etiquetas, mode="markers",
        marker=dict(size=12, color=color, line=dict(color="#fff", width=2)),
        error_x=dict(type="data", symmetric=False,
                     array=[s - m for m, (_, s) in zip(medias, ics)],
                     arrayminus=[m - i for m, (i, _) in zip(medias, ics)],
                     thickness=2, width=0, color=color),
        customdata=[[i, s] for i, s in ics],
        hovertemplate="<b>%{y}</b><br>%{x:.2f} (IC %{customdata[0]:.2f} – "
                      "%{customdata[1]:.2f})<extra></extra>"))
    if referencia is not None:
        fig.add_vline(x=referencia, line_dash="dot", line_color=TINTA_MUTED, line_width=1)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    if rango:
        fig.update_xaxes(range=rango)
    return layout(fig, alto or max(240, 58 * len(etiquetas) + 80), title=titulo)


def gauge_t(R):
    t = R["T_global"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=t["media"], number=dict(valueformat=".1f",
                                                           font=dict(size=44, color=MARCA)),
        gauge=dict(axis=dict(range=[20, 80], tickvals=[20, 40, 50, 60, 80],
                             tickcolor=EJE),
                   bar=dict(color=MARCA, thickness=0.28),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[20, 40], color="#f6cfcf"),
                          dict(range=[40, 60], color="#ecebe6"),
                          dict(range=[60, 80], color="#cde2fb")],
                   threshold=dict(line=dict(color=TINTA_MUTED, width=2), value=50))))
    return layout(fig, 270, margin=dict(l=30, r=30, t=30, b=20))


def dona_niveles(riesgo, alto, titulo="", alto_px=300):
    tipico = max(0.0, 1 - riesgo - alto)
    fig = go.Figure(go.Pie(
        labels=["En riesgo (T < 40)", "Rango típico", "Bienestar alto (T ≥ 60)"],
        values=[riesgo, tipico, alto], hole=0.62, sort=False, direction="clockwise",
        marker=dict(colors=[RIESGO, TIPICO, ALTO], line=dict(color="#fff", width=2)),
        textinfo="percent", texttemplate="%{percent:.1%}", textfont=dict(color="#fff", size=13),
        hovertemplate="%{label}: %{percent:.1%}<extra></extra>"))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1, x=0.5,
                                                   xanchor="center"))
    return layout(fig, alto_px, title=titulo, margin=dict(l=10, r=10, t=40, b=10))


def histograma(R):
    bordes = R["histograma_bordes"]
    centros = [(a + b) / 2 for a, b in zip(bordes[:-1], bordes[1:])]
    colores = [RIESGO if c < 40 else (ALTO if c >= 60 else TIPICO) for c in centros]
    fig = go.Figure(go.Bar(x=centros, y=R["histograma"], marker_color=colores, width=3.6,
                           marker_line=dict(color="#fff", width=1),
                           hovertemplate="T ≈ %{x}: %{y:,.0f} estudiantes<extra></extra>"))
    for x, texto in ((40, "Riesgo"), (60, "Alto")):
        fig.add_vline(x=x, line_dash="dot", line_color=TINTA_MUTED, line_width=1,
                      annotation_text=texto, annotation_position="top",
                      annotation_font_color=TINTA_2)
    return layout(fig, 340, title="Distribución poblacional del puntaje T",
                  xaxis_title="Puntaje T global", yaxis_title="Estudiantes (población estimada)")


def mapa_seccionales(grupos):
    nombres = [k for k in grupos if k in COORDENADAS]
    if not nombres:
        return None
    lat = [COORDENADAS[k][0] for k in nombres]
    lon = [COORDENADAS[k][1] for k in nombres]
    N = [grupos[k]["N"] for k in nombres]
    fig = go.Figure(go.Scattergeo(
        lat=lat, lon=lon, text=nombres, mode="markers+text", textposition="middle right",
        textfont=dict(color=TINTA, size=13),
        marker=dict(size=[12 + 30 * (v / max(N)) ** 0.5 for v in N], color=AZUL, opacity=.85,
                    line=dict(color="#fff", width=2)),
        customdata=[[grupos[k]["media"], grupos[k]["n"], grupos[k]["N"]] for k in nombres],
        hovertemplate="<b>%{text}</b><br>T = %{customdata[0]:.1f}<br>n = %{customdata[1]:,}"
                      " respuestas<br>N = %{customdata[2]:,} estudiantes<extra></extra>"))
    fig.update_geos(lonaxis_range=[-78.5, -69.5], lataxis_range=[1.5, 10.5],
                    showcountries=True, countrycolor=EJE, showland=True, landcolor="#f3f2ee",
                    showocean=True, oceancolor="#eef4fb", showlakes=False,
                    showcoastlines=True, coastlinecolor=EJE, bgcolor="rgba(0,0,0,0)",
                    projection_type="mercator")
    return layout(fig, 380, margin=dict(l=0, r=0, t=10, b=0))


def peso_grupos(grupos):
    """Barras del porcentaje de la población que representa cada grupo (una sola serie)."""
    total = sum(g["N"] for g in grupos.values()) or 1
    nombres = list(grupos)
    pct = [grupos[k]["N"] / total * 100 for k in nombres]
    fig = go.Figure(go.Bar(x=pct, y=nombres, orientation="h", marker_color=AZUL,
                           text=[f"{v:.0f} %" for v in pct], textposition="outside",
                           customdata=[grupos[k]["N"] for k in nombres],
                           hovertemplate="%{y}: %{customdata:,} estudiantes (%{x:.1f} %)"
                                         "<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(ticksuffix=" %", range=[0, max(pct) * 1.25])
    return layout(fig, max(240, 58 * len(nombres) + 80),
                  title="Peso de cada grupo en la población")


def barras_niveles(grupos, titulo="Niveles de bienestar por grupo"):
    nombres = [k for k, g in grupos.items() if "riesgo" in g]
    if not nombres:
        return None
    r = [grupos[k]["riesgo"]["media"] * 100 for k in nombres]
    a = [grupos[k]["alto"]["media"] * 100 for k in nombres]
    t = [100 - x - y for x, y in zip(r, a)]
    fig = go.Figure()
    for valores, nombre, color, txt in ((r, "En riesgo", RIESGO, "#fff"),
                                        (t, "Rango típico", TIPICO, TINTA),
                                        (a, "Bienestar alto", ALTO, "#fff")):
        fig.add_bar(y=nombres, x=valores, name=nombre, orientation="h", marker_color=color,
                    marker_line=dict(color="#fff", width=2), text=[f"{v:.0f} %" for v in valores],
                    textposition="inside", insidetextfont=dict(color=txt),
                    hovertemplate="%{y} · " + nombre + ": %{x:.1f} %<extra></extra>")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(range=[0, 100], ticksuffix=" %")
    return layout(fig, max(220, 52 * len(nombres) + 110), title=titulo, barmode="stack",
                  legend=dict(orientation="h", y=-0.18, x=0, traceorder="normal"))


def radar(series, titulo=""):
    """series: lista de (nombre, {dimension: media}). Máximo tres, en orden fijo de color."""
    fig = go.Figure()
    for (nombre, valores), color in zip(series[:3], SERIE):
        dims = list(valores) + [list(valores)[0]]
        fig.add_trace(go.Scatterpolar(
            r=[valores[d] for d in dims], theta=dims, name=nombre, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=8),
            fill="toself" if len(series) == 1 else None, fillcolor="rgba(42,120,214,.12)",
            hovertemplate="%{theta}: %{r:.2f}<extra>" + nombre + "</extra>"))
    todos = [v for _, vals in series for v in vals.values()]
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[min(todos) - 1, max(todos) + 0.6],
                                                 gridcolor=REJILLA, tickfont=dict(size=11)),
                                 angularaxis=dict(gridcolor=REJILLA)),
                      showlegend=len(series) > 1,
                      legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
    return layout(fig, 420, title=titulo)


def heatmap_dimensiones(grupos, titulo):
    nombres = [k for k, g in grupos.items() if "dimensiones" in g]
    if not nombres:
        return None
    dims = list(grupos[nombres[0]]["dimensiones"])
    z = [[grupos[g]["dimensiones"][d]["media"] for d in dims] for g in nombres]
    desv = max(abs(v - 50) for fila in z for v in fila) or 1
    fig = go.Figure(go.Heatmap(
        z=z, x=dims, y=nombres, colorscale=DIVERGENTE, zmin=50 - desv, zmax=50 + desv,
        text=[[f"{v:.1f}" for v in fila] for fila in z], texttemplate="%{text}",
        textfont=dict(color=TINTA, size=12), xgap=2, ygap=2,
        colorbar=dict(title="T", thickness=12, outlinewidth=0),
        hovertemplate="%{y} · %{x}: T = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", side="top")
    return layout(fig, max(260, 48 * len(nombres) + 140), title=titulo,
                  margin=dict(l=10, r=10, t=90, b=10))


def cascada_brecha(B):
    m = {x["modelo"].split(":")[0]: x["brecha"] for x in B["modelos"]}
    bruta, m1, m2 = m.get("M0", B["brecha_bruta"]), m.get("M1"), B["brecha_ajustada"]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "total"],
        x=["Brecha bruta", "Efecto de la edad", "Género, estrato<br>y seccional",
           "Brecha ajustada"],
        y=[bruta, m1 - bruta, m2 - m1, 0],
        text=[f"{bruta:+.2f}", f"{m1 - bruta:+.2f}", f"{m2 - m1:+.2f}", f"{m2:+.2f}"],
        textposition="outside", connector=dict(line=dict(color=EJE, width=1)),
        increasing=dict(marker=dict(color=NARANJA)), decreasing=dict(marker=dict(color=AQUA)),
        totals=dict(marker=dict(color=AZUL)),
        hovertemplate="%{x}: %{text} puntos<extra></extra>"))
    fig.update_yaxes(title="Puntos T", range=[0, bruta * 1.25])
    return layout(fig, 360, title="De la brecha bruta a la ajustada (posgrado − pregrado)",
                  showlegend=False)


def coeficientes(B):
    coef = {k: v for k, v in B["modelo_completo"]["coeficientes"].items() if k != "Intercepto"}
    nombres = list(coef)
    return forest(nombres, [coef[k]["coef"] for k in nombres],
                  [[coef[k]["ic_inferior"], coef[k]["ic_superior"]] for k in nombres],
                  "Modelo completo: efecto de cada variable (puntos T, IC 95 %)",
                  referencia=0, color=AZUL)


def dona_varianza(icc, titulo):
    fig = go.Figure(go.Pie(
        labels=["Entre grupos", "Dentro de cada grupo"], values=[icc, 1 - icc], hole=0.65,
        sort=False, marker=dict(colors=[NARANJA, "#dfe6ef"], line=dict(color="#fff", width=2)),
        textinfo="none", hovertemplate="%{label}: %{percent:.1%}<extra></extra>"))
    fig.add_annotation(text=f"<b>{icc * 100:.1f} %</b><br><span style='font-size:11px'>entre "
                            f"grupos</span>", showarrow=False, font=dict(size=20, color=MARCA))
    fig.update_layout(showlegend=False)
    return layout(fig, 250, title=titulo, margin=dict(l=10, r=10, t=40, b=10))


def efectos_programa(bloque, titulo):
    filas = bloque["grupos_mas_altos"] + bloque["grupos_mas_bajos"]
    if not filas:
        return None
    filas = sorted(filas, key=lambda f: f["efecto_estimado"])
    nombres = [f"{f['grupo'].title()[:42]} (n={f['n']})" for f in filas]
    valores = [f["efecto_estimado"] for f in filas]
    fig = go.Figure(go.Bar(
        x=valores, y=nombres, orientation="h",
        marker_color=[AZUL if v >= 0 else NARANJA for v in valores],
        text=[f"{v:+.1f}" for v in valores], textposition="outside",
        hovertemplate="%{y}: %{x:+.2f} puntos<extra></extra>"))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return layout(fig, 60 + 34 * len(filas), title=titulo, xaxis_title="Puntos T sobre el "
                  "promedio (BLUP)", margin=dict(l=10, r=40, t=50, b=10))


def reproducibilidad(filas):
    """Diferencia relativa (%) de cada fuente contra lo reproducido; 0 = coincide."""
    fuentes = [("Referencia entrega 1", AZUL), ("Informe Ejecutivo", NARANJA)]
    fig = go.Figure()
    indicadores = [f["Indicador"] for f in filas]
    for fuente, color in fuentes:
        x = []
        for f in filas:
            ref = f.get(fuente)
            x.append((float(ref) / f["Reproducido"] - 1) * 100
                     if isinstance(ref, (int, float)) and f["Reproducido"] else None)
        fig.add_bar(y=indicadores, x=x, name=fuente, orientation="h", marker_color=color,
                    marker_line=dict(color="#fff", width=1),
                    hovertemplate="%{y}<br>" + fuente + ": %{x:+.1f} % frente a lo reproducido"
                                  "<extra></extra>")
    fig.add_vline(x=0, line_color=TINTA_MUTED, line_width=1)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return layout(fig, 70 + 34 * len(filas), barmode="group", xaxis_ticksuffix=" %",
                  title="Diferencia de cada fuente frente a la cifra reproducida",
                  legend=dict(orientation="h", y=-0.08, x=0))


def dif_barras(r):
    import pandas as pd
    tabla = pd.DataFrame(r["tabla"]).sort_values("delta_r2")
    fig = go.Figure(go.Bar(x=tabla["delta_r2"], y=tabla["item"], orientation="h",
                           marker_color=AZUL,
                           hovertemplate="%{y}: ΔR² = %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0.035, line_dash="dash", line_color=NARANJA, line_width=2,
                  annotation_text="DIF moderado", annotation_position="top",
                  annotation_font_color=TINTA_2)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11))
    return layout(fig, 680, title=r["comparacion"], xaxis_title="ΔR² de Nagelkerke",
                  xaxis_range=[0, 0.045])
