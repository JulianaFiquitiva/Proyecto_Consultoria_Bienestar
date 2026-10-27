# -*- coding: utf-8 -*-
"""
App del Índice de Bienestar Estudiantil USTA: contexto del estudio,
resultados, gráficas y chat con el agente.

    streamlit run app/dashboard.py

Lee resultados/entrega2/hallazgos_entrega2.json (generado con
`python -m agent.reports.entrega2`). Solo muestra agregados; los grupos con
menos de 30 casos no se muestran. No escribe en data/.

Las páginas están en app/paginas/; el estilo y los gráficos, en app/comun.py.

Variables de entorno (todas opcionales):
- INDICE_MODO: "publico" (solo el JSON de agregados, para publicar la app),
  "completo" (además el agente con los Excel de data/) o "auto" (por
  defecto: completo si están los datos).
- APP_PASSWORD: si se define, la app pide esa contraseña.
- CHAT_MAX_CONSULTAS: consultas por sesión que usan servicios externos
  (OpenAI para el corpus, Crossref/PubMed). Por defecto 20.
"""
import hmac
import os
import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent))     # paquete agent
sys.path.insert(0, str(APP))            # comun.py, para las páginas

import comun as c  # noqa: E402

st.set_page_config(page_title="Índice de Bienestar USTA", page_icon=":material/insights:",
                   layout="wide")
c.estilo()


def _pedir_contrasena():
    """Si APP_PASSWORD está definida, nada se muestra sin ella."""
    clave = os.environ.get("APP_PASSWORD", "")
    if not clave or st.session_state.get("autenticado"):
        return
    c.hero("Universidad Santo Tomás", "Índice de Bienestar Estudiantil",
           "Ingrese la contraseña para ver los resultados.")
    intento = st.text_input("Contraseña", type="password")
    if st.button("Entrar", type="primary"):
        if hmac.compare_digest(intento.encode(), clave.encode()):
            st.session_state["autenticado"] = True
            st.rerun()
        st.error("Contraseña incorrecta.")
    st.stop()


_pedir_contrasena()

try:
    c.H()
except FileNotFoundError:
    st.error("No hay hallazgos precalculados ni datos en data/. Ejecute "
             "`python -m agent.reports.entrega2` en un equipo con los datos.")
    st.stop()

PAGINAS = [
    st.Page(APP / "paginas" / "inicio.py", title="Inicio", icon=":material/home:", default=True),
    st.Page(APP / "paginas" / "estudio.py", title="El estudio", icon=":material/school:"),
    st.Page(APP / "paginas" / "resultados.py", title="Resultados", icon=":material/insights:"),
    st.Page(APP / "paginas" / "comparaciones.py", title="Comparaciones",
            icon=":material/compare_arrows:"),
    st.Page(APP / "paginas" / "dimensiones.py", title="Dimensiones", icon=":material/radar:"),
    st.Page(APP / "paginas" / "hallazgos.py", title="Hallazgos", icon=":material/lab_research:"),
    st.Page(APP / "paginas" / "chat.py", title="Pregúntale al agente", icon=":material/forum:"),
    st.Page(APP / "paginas" / "metodologia.py", title="Metodología", icon=":material/menu_book:"),
]
st.navigation(PAGINAS, position="top").run()
