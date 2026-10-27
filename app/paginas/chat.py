# -*- coding: utf-8 -*-
"""Pregúntale al agente: chat con las mismas rutas del agente."""
import streamlit as st

import comun as c
from agent.core.chat import EJEMPLOS

H = c.H()
MODO = c.modo()
RUTA = {"encuesta": ("Encuesta", ":material/bar_chart:"),
        "corpus": ("Corpus documental", ":material/menu_book:"),
        "busqueda": ("Búsqueda bibliográfica", ":material/travel_explore:"),
        "abstencion": ("Abstención", ":material/block:")}

st.session_state.setdefault("chat", [])
st.session_state.setdefault("consultas_externas", 0)


def grafico(spec):
    """Dibuja el gráfico que acompaña a una respuesta sobre la encuesta."""
    if not spec:
        return
    tipo = spec.get("tipo")
    R = H["resumen"]
    fig = None
    if tipo == "niveles":
        fig = c.dona_niveles(R["riesgo"]["media"], R["alto"]["media"], alto_px=280)
    elif tipo == "comparacion" and spec.get("clave") in H["comparaciones"]:
        d = H["comparaciones"][spec["clave"]]
        fig = c.forest(list(d["grupos"]), [g["media"] for g in d["grupos"].values()],
                       [g["ic"] for g in d["grupos"].values()],
                       f"Puntaje T por {d['etiqueta'].lower()} (IC 95 %)",
                       referencia=R["T_global"]["media"])
    elif tipo == "brecha":
        fig = c.cascada_brecha(H["brecha"])
    elif tipo == "dimensiones":
        fig = c.radar([("Población", {k: v["media"] for k, v in H["dimensiones"].items()})])
    elif tipo == "multinivel":
        fig = c.efectos_programa(H["multinivel"]["programa"]["condicional_nivel_edad"],
                                 "Programas con más y menos bienestar (n ≥ 30)")
    elif tipo == "dif" and "dif" in H:
        fig = c.dif_barras(H["dif"][spec["grupo"]])
    elif tipo == "reproducibilidad":
        fig = c.reproducibilidad(H["reproducibilidad"])
    if fig is not None:
        c.mostrar(fig)


def mostrar_respuesta(m):
    if m["preformateado"]:
        st.text(m["texto"])
    else:
        st.markdown(m["texto"])
    grafico(m.get("grafico"))
    if m.get("decision"):
        with st.expander("Cómo eligió el método el agente", icon=":material/account_tree:"):
            st.markdown("\n".join(f"{i}. {paso}" for i, paso in enumerate(m["decision"], 1)))
    etiqueta, icono = RUTA.get(m["ruta"], (m["ruta"], ":material/help:"))
    st.caption(f"{etiqueta} · Fuente: {m['fuente']}"
               + (f" · Método: {m['metodo']}" if m["metodo"] else ""))
    if m["avisos"]:
        with st.expander("Avisos"):
            for aviso in m["avisos"]:
                st.write(f"- {aviso}")


def responder(pregunta):
    motor = c.motor(MODO)
    externa = motor.agente.ruta_de_pregunta(pregunta) in ("corpus", "busqueda")
    if externa and st.session_state["consultas_externas"] >= c.max_consultas():
        return {"texto": (f"Se alcanzó el límite de {c.max_consultas()} consultas al corpus y a "
                          "la búsqueda bibliográfica en esta sesión. Las preguntas sobre los "
                          "resultados de la encuesta siguen disponibles."),
                "ruta": "abstencion", "fuente": "ninguna", "metodo": "",
                "preformateado": False, "avisos": [], "grafico": None, "decision": []}
    r = motor.responder(pregunta)
    if r.usa_servicio_externo:
        st.session_state["consultas_externas"] += 1
    return {"texto": r.texto, "ruta": r.ruta, "fuente": r.fuente, "metodo": r.metodo,
            "preformateado": r.preformateado, "avisos": r.avisos, "grafico": r.grafico,
            "decision": r.decision}


# ── Barra lateral ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Pregúntale al agente")
    st.caption("Ante una pregunta sobre los datos, el agente decide qué análisis hacer, "
               "elige el método estadístico y explica el resultado. Para la literatura "
               "usa el RAG con 20 documentos académicos. Cada respuesta dice de dónde sale; "
               "si no hay fuente, no responde.")
    if MODO == "publico":
        st.caption("Versión pública: las cifras salen de resultados agregados, nunca de los "
                   "datos individuales.")
    st.caption(f"Consultas al corpus en esta sesión: {st.session_state['consultas_externas']} "
               f"de {c.max_consultas()}.")
    if st.session_state["chat"] and st.button("Nueva conversación", icon=":material/refresh:",
                                              width="stretch"):
        st.session_state["chat"] = []
        st.rerun()

# ── Conversación ──────────────────────────────────────────────────────
ejemplo = None
if not st.session_state["chat"]:
    c.hero("Chat", "Pregúntale al agente",
           "Pregunte sobre los datos y el agente decide qué método estadístico usar, lo "
           "aplica y explica el resultado con cifras y un gráfico. Abra «Cómo eligió el "
           "método» en cada respuesta para ver el paso a paso. Para la literatura "
           "académica, empiece con <b>corpus</b>.")
    cols = st.columns(2)
    for i, texto in enumerate(EJEMPLOS):
        if cols[i % 2].button(texto, key=f"ejemplo_{i}", width="stretch",
                              icon=":material/chat_bubble:"):
            ejemplo = texto

for m in st.session_state["chat"]:
    if m["rol"] == "user":
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(m["texto"])
    else:
        with st.chat_message("assistant", avatar=":material/analytics:"):
            mostrar_respuesta(m)

pregunta = st.chat_input("Escriba su pregunta sobre el estudio…") or ejemplo
if pregunta:
    st.session_state["chat"].append({"rol": "user", "texto": pregunta})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(pregunta)
    with st.chat_message("assistant", avatar=":material/analytics:"):
        with st.spinner("Buscando la respuesta…"):
            respuesta = responder(pregunta)
    st.session_state["chat"].append({"rol": "assistant", **respuesta})
    st.rerun()
