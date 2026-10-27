# -*- coding: utf-8 -*-
"""
Motor del chat de la app ("Pregúntale al agente").

Responde con la MISMA cascada de rutas del agente (encuesta / corpus /
busqueda / abstencion), pero las preguntas sobre la encuesta se contestan
con los resultados AGREGADOS de resultados/entrega2/hallazgos_entrega2.json:

- modo "publico": solo agregados. No necesita los Excel de data/, asi que
  sirve para publicar la app. Lo que no esta precalculado se abstiene.
- modo "completo": primero los agregados y, si la pregunta pide un analisis
  que no esta precalculado (correlacion, regresion...), el agente completo
  con los datos de data/.

Las respuestas sobre la encuesta no llaman al LLM: se redactan con
plantillas a partir de las cifras, asi que no cuestan ni inventan numeros.
Solo el corpus (RAG) llama al LLM.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.core.query_planner import classify_scope

MODOS = ("publico", "completo")

EJEMPLOS = [
    "¿Cuál es el nivel general de bienestar?",
    "¿Hay diferencias entre hombres y mujeres?",
    "¿Cómo varía el bienestar por seccional?",
    "¿La brecha de posgrado se mantiene controlando la edad?",
    "¿La escala mide igual a hombres y mujeres?",
    "¿Cuánto depende el bienestar del programa?",
    "corpus ¿Qué escala usó el estudio con estudiantes de psicología de Bucaramanga?",
]


@dataclass
class RespuestaChat:
    texto: str                      # markdown (o texto plano si preformateado)
    ruta: str                       # encuesta | corpus | busqueda | abstencion
    fuente: str                     # de donde salen las cifras o el texto
    metodo: str = ""
    usa_servicio_externo: bool = False   # OpenAI, Crossref o PubMed
    preformateado: bool = False
    avisos: List[str] = field(default_factory=list)
    citas: List[Dict[str, Any]] = field(default_factory=list)
    # Gráfico sugerido para la app, p. ej. {"tipo": "comparacion", "clave": "Género"}
    grafico: Optional[Dict[str, str]] = None


def _norm(texto: str) -> str:
    """Minusculas y sin tildes, para comparar nombres de grupos y columnas."""
    sin_tildes = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in sin_tildes if not unicodedata.combining(c)).lower().strip()


def _num(x) -> str:
    """Cifra de la tabla de reproducibilidad (puede venir como texto, p. ej. '—')."""
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def _pct(x: float) -> str:
    return f"{x * 100:.1f} %"


def _ic(ic, dec=1, pct=False) -> str:
    if pct:
        return f"IC 95 %: {ic[0] * 100:.{dec}f} – {ic[1] * 100:.{dec}f} %"
    return f"IC 95 %: {ic[0]:.{dec}f} – {ic[1]:.{dec}f}"


class MotorChat:
    """Responde preguntas con los hallazgos agregados y, si hace falta, el agente."""

    def __init__(self, hallazgos: Dict[str, Any], modo: str = "publico", agente=None):
        if modo not in MODOS:
            raise ValueError(f"modo debe ser uno de {MODOS}")
        self.H = hallazgos
        self.modo = modo
        if agente is None:
            from agent.core.orchestrator import StatisticalAgent
            agente = StatisticalAgent()       # sin datos: solo enruta y consulta el corpus
        self.agente = agente
        self.fuente_agregados = (f"Resultados agregados de la entrega 2 "
                                 f"(generados el {hallazgos.get('generado', '')[:10]})")

    # ── Entrada ──────────────────────────────────────────────────────────
    def responder(self, pregunta: str) -> RespuestaChat:
        pregunta = (pregunta or "").strip()
        if not pregunta:
            return self._abstencion("ambigua", "La pregunta está vacía.")

        from agent.core.orchestrator import pregunta_sobre_corpus
        ruta = self.agente.ruta_de_pregunta(pregunta)
        if ruta == "corpus":
            return self._corpus(pregunta_sobre_corpus(pregunta) or pregunta)
        if ruta == "busqueda":
            return self._busqueda(pregunta)
        if ruta == "abstencion":
            scope, motivo = classify_scope(pregunta)
            if scope == "ambigua":
                # En el chat todo se pregunta sobre el estudio de bienestar:
                # "¿hay diferencias entre hombres y mujeres?" no nombra el
                # bienestar, pero pide un analisis concreto. Solo se responde
                # si con ese contexto las reglas reconocen un analisis.
                plan = self.agente._fallback_plan(f"bienestar {pregunta}")
                if plan.function_to_call != "abstencion":
                    return self._encuesta(pregunta, plan)
            return self._abstencion(scope, motivo)
        return self._encuesta(pregunta)

    # ── Encuesta: agregados ──────────────────────────────────────────────
    def _encuesta(self, pregunta: str, plan=None) -> RespuestaChat:
        plan = plan or self.agente._fallback_plan(pregunta)
        f = plan.function_to_call
        p = plan.parameters or {}

        texto, grafico = None, None
        if f == "descriptive_overview":
            texto, grafico = self._txt_resumen(), {"tipo": "niveles"}
        elif f in ("design_comparison", "group_comparison", "multi_group_comparison",
                   "smart_comparison"):
            clave = self._clave_comparacion(p.get("group_col", ""))
            if clave:
                texto = self._txt_comparacion(clave, pregunta)
                grafico = {"tipo": "comparacion", "clave": clave}
        elif f == "adjusted_gap":
            texto, grafico = self._txt_brecha(), {"tipo": "brecha"}
        elif f == "dif_analysis" and "dif" in self.H:
            grupo = p.get("group", "genero")
            texto = self._txt_dif(grupo)
            grafico = {"tipo": "dif", "grupo": grupo if grupo in self.H["dif"]
                       else next(iter(self.H["dif"]))}
        elif f == "multilevel_icc":
            texto, grafico = self._txt_multinivel(), {"tipo": "multinivel"}
        elif f == "dimension_profile":
            texto, grafico = self._txt_dimensiones(), {"tipo": "dimensiones"}
        elif f in ("validate_results", "full_validation"):
            texto, grafico = self._txt_reproducibilidad(), {"tipo": "reproducibilidad"}

        if texto is not None:
            return RespuestaChat(texto=texto, ruta="encuesta", fuente=self.fuente_agregados,
                                 metodo=plan.description, grafico=grafico)

        # No esta precalculado
        if self.modo == "completo":
            return self._agente_completo(pregunta)
        if f == "abstencion":
            return self._abstencion(p.get("scope", "ambigua"), p.get("motivo", ""))
        return RespuestaChat(
            texto=("Esa consulta necesita un análisis que no está precalculado en la "
                   f"versión pública ({plan.description.lower()}). Aquí solo hay "
                   "resultados agregados, para no publicar los datos individuales.\n\n"
                   "Puedo responder sobre: bienestar general, comparaciones por "
                   "seccional, modalidad, género, estrato o nivel de formación, "
                   "dimensiones de Ryff, la brecha posgrado–pregrado, el DIF de la "
                   "escala, el efecto del programa y la reproducibilidad del estudio."),
            ruta="abstencion", fuente="ninguna", metodo="no disponible en modo público")

    def _agente_completo(self, pregunta: str) -> RespuestaChat:
        r = self.agente.ask(pregunta)
        return RespuestaChat(
            texto=r.interpretation, ruta="encuesta",
            fuente=str(r.traceability.get("dataset", "data/")),
            metodo=str(r.traceability.get("method") or r.plan.description),
            usa_servicio_externo=bool(self.agente.llm.is_configured),
            preformateado=True, avisos=list(r.warnings))

    def _clave_comparacion(self, group_col: str) -> Optional[str]:
        objetivo = _norm(group_col)
        if not objetivo:
            return None
        # "Estrato socioeconomico" (sin tilde) debe encontrar "Estrato socioeconómico"
        for clave in self.H.get("comparaciones", {}):
            if _norm(clave).split()[0] == objetivo.split()[0]:
                return clave
        return None

    # ── Plantillas ───────────────────────────────────────────────────────
    def _txt_resumen(self) -> str:
        R = self.H["resumen"]
        t = R["T_global"]
        return (
            f"**Bienestar general de la población estudiantil** "
            f"(N = {R['N']:,} estudiantes; n = {R['n']:,} respuestas)\n\n"
            f"- T-score medio: **{t['media']:.1f}** ({_ic(t['ic'], 2)})\n"
            f"- En riesgo (T < 40): **{_pct(R['riesgo']['media'])}** "
            f"({_ic(R['riesgo']['ic'], pct=True)})\n"
            f"- Bienestar alto (T ≥ 60): **{_pct(R['alto']['media'])}** "
            f"({_ic(R['alto']['ic'], pct=True)})\n\n"
            f"El T-score tiene centro 50 y desviación 10: la población está muy cerca "
            f"del centro de la escala. Son estimaciones poblacionales con los factores "
            f"de expansión (sin ponderar, la media muestral sería "
            f"{R['media_sin_ponderar']:.2f})."
        )

    def _txt_comparacion(self, clave: str, pregunta: str) -> str:
        C = self.H["comparaciones"][clave]
        q = _norm(pregunta)
        grupos = sorted(C["grupos"].items(), key=lambda kv: -kv[1]["media"])
        lineas = [f"**Bienestar por {C['etiqueta'].lower()}** (T-score poblacional)\n"]
        for nombre, g in grupos:
            marca = "**" if _norm(nombre) in q else ""
            lineas.append(f"- {marca}{nombre}{marca}: {g['media']:.1f} "
                          f"({_ic(g['ic'])}; n = {g['n']:,})")
        sig = [p for p in C["pares"] if p["significativo"]]
        lineas.append("")
        if not C["pares"]:
            lineas.append("No hay suficientes grupos con 30 casos o más para comparar.")
        elif not sig:
            lineas.append("Ninguna diferencia entre grupos es estadísticamente significativa "
                          "con la varianza del diseño y la corrección de Bonferroni.")
        else:
            lineas.append("Diferencias significativas (Wald con varianza del diseño, "
                          "Bonferroni):")
            for p in sig:
                # Siempre "mayor − menor", para leer la diferencia en positivo
                alto, bajo = ((p["grupo_1"], p["grupo_2"]) if p["diferencia"] > 0
                              else (p["grupo_2"], p["grupo_1"]))
                pa = p["p_ajustado_bonferroni"]
                lineas.append(f"- {alto} − {bajo}: +{abs(p['diferencia']):.2f} puntos "
                              f"(p ajustado {'< 0.001' if pa < 0.001 else f'= {pa:.3f}'})")
            if max(abs(p["diferencia"]) for p in sig) < 3:
                lineas.append("\nSon diferencias pequeñas: menos de un tercio de desviación "
                              "estándar.")
        exc = C.get("excluidos", {})
        if exc.get("categorias"):
            lineas.append(f"\n_No se muestran {exc['categorias']} categoría(s) con menos de "
                          f"{self.H.get('min_n_grupo', 30)} casos._")
        return "\n".join(lineas)

    def _txt_brecha(self) -> str:
        B = self.H["brecha"]
        lineas = ["**Brecha de bienestar posgrado − pregrado**\n"]
        for m in B["modelos"]:
            lineas.append(f"- {m['modelo']}: **{m['brecha']:+.2f}** puntos "
                          f"(IC 95 %: {m['ic_inferior']:.2f} – {m['ic_superior']:.2f})")
        edad = B.get("edad_media_ponderada", {})
        lineas.append(
            f"\nLa brecha bruta de {B['brecha_bruta']:.1f} puntos baja a "
            f"{B['brecha_ajustada']:.1f} al controlar edad, género, estrato y seccional: "
            f"los controles explican el {B['porcentaje_explicado_por_controles']:.0f} %.")
        if edad:
            lineas.append(f"Los estudiantes de posgrado tienen en promedio "
                          f"{edad.get('Posgrado', 0):.0f} años y los de pregrado "
                          f"{edad.get('Pregrado', 0):.0f}.")
        lineas.append("La brecha ajustada sigue siendo significativa. Es una asociación, "
                      "no un efecto causal.")
        return "\n".join(lineas)

    def _txt_dif(self, grupo: str) -> str:
        r = self.H["dif"].get(grupo) or next(iter(self.H["dif"].values()))
        top = r["tabla"][:3]
        lineas = [f"**¿La escala mide igual? DIF por {r['comparacion']}**\n",
                  f"- Ítems con DIF significativo (Bonferroni): "
                  f"{r['items_significativos_bonferroni']} de {r['n_items']}",
                  f"- ΔR² de Nagelkerke máximo: **{r['delta_r2_maximo']:.4f}** "
                  f"(el DIF moderado empieza en 0,035)",
                  "", r["conclusion"], "", "Ítems con mayor ΔR²:"]
        for fila in top:
            lineas.append(f"- {fila['item']}: ΔR² = {fila['delta_r2']:.4f} "
                          f"({fila['magnitud']})")
        return "\n".join(lineas)

    def _txt_multinivel(self) -> str:
        ML = self.H["multinivel"]
        lineas = ["**¿Cuánto depende el bienestar del programa?** (correlación "
                  "intraclase, ICC)\n"]
        for clave, nombre in (("programa", "Programa"), ("nbc", "Núcleo básico (NBC)")):
            if clave in ML:
                nulo, cond = ML[clave]["nulo"], ML[clave]["condicional_nivel_edad"]
                lineas.append(f"- {nombre} ({nulo['n_grupos']} grupos): "
                              f"**{_pct(nulo['icc'])}** de la varianza sin controles; "
                              f"**{_pct(cond['icc'])}** controlando nivel y edad")
        prog = ML.get("programa", {}).get("nulo", {})
        lineas.append("\nMás del 95 % de la variación del bienestar ocurre dentro de cada "
                      "programa.")
        if prog:
            lineas.append(f"{prog['grupos_con_menos_de_10']} de {prog['n_grupos']} programas "
                          f"tienen menos de 10 respuestas, así que no se publica un ranking "
                          f"de programas.")
        if ML.get("limitacion"):
            lineas.append(f"\n_{ML['limitacion']}_")
        return "\n".join(lineas)

    def _txt_dimensiones(self) -> str:
        D = self.H["dimensiones"]
        orden = sorted(D, key=lambda k: -D[k]["media"])
        lineas = ["**Bienestar por dimensión de Ryff** (T-score poblacional)\n"]
        for k in orden:
            lineas.append(f"- {k}: {D[k]['media']:.1f} ({_ic(D[k]['ic'])})")
        lineas.append(f"\nLa más alta es {orden[0]} y la más baja {orden[-1]}, pero los "
                      f"intervalos se solapan: las seis dimensiones están muy cerca de 50.")
        return "\n".join(lineas)

    def _txt_reproducibilidad(self) -> str:
        filas = self.H["reproducibilidad"]
        col = "Estado vs. Referencia entrega 1"
        validadas = sum(1 for f in filas if f.get(col) == "VALIDADO")
        lineas = ["**¿Se reproduce el estudio anterior?**\n",
                  f"Con los factores de expansión se reproducen **{validadas} de "
                  f"{len(filas)}** cifras de referencia:\n"]
        for f in filas:
            estado = f.get(col, "—")
            lineas.append(f"- {f['Indicador']}: {_num(f['Reproducido'])} "
                          f"(referencia {_num(f['Referencia entrega 1'])}) · {estado.lower()}")
        lineas.append("\nEl Informe Ejecutivo del estudio anterior reporta cifras distintas "
                      "(T = 51,2) y tamaños de muestra por seccional que no coinciden con "
                      "la base: parece venir de otra versión de los datos.")
        return "\n".join(lineas)

    # ── Otras rutas ──────────────────────────────────────────────────────
    @staticmethod
    def corpus_disponible() -> bool:
        """Sin el indice (rag_store/ no se versiona) el corpus se abstendria
        siempre, como si los documentos no dijeran nada: mejor avisarlo."""
        from agent.rag.embedder import TFIDF_VECTORIZER
        from agent.rag.vectorstore import STORE_DIR
        return TFIDF_VECTORIZER.exists() and (STORE_DIR / "chroma.sqlite3").exists()

    def _corpus(self, pregunta: str) -> RespuestaChat:
        if not self.corpus_disponible():
            return RespuestaChat(
                texto=("El corpus de documentos académicos no está disponible en esta "
                       "versión de la app, así que no puedo responder con citas. Las "
                       "preguntas sobre los resultados de la encuesta sí funcionan."),
                ruta="corpus", fuente="corpus documental (no instalado)", metodo="RAG")
        r = self.agente.ask_corpus(pregunta)
        res = r.result or {}
        if not r.success:
            return RespuestaChat(
                texto=("No pude consultar el corpus de documentos en este momento. "
                       f"Detalle: {r.error}"),
                ruta="corpus", fuente="corpus documental (no disponible)",
                metodo="RAG", avisos=list(r.warnings))
        citas = res.get("fuentes", [])
        texto = res.get("respuesta", "")
        if citas:
            texto += "\n\n**Fuentes:**\n" + "\n".join(
                f"{i}. {c['archivo']}, p. {c['pagina']}" for i, c in enumerate(citas, 1))
        return RespuestaChat(
            texto=texto, ruta="corpus",
            fuente="Corpus de 20 documentos académicos (RAG)",
            metodo=r.traceability.get("method", "rag"),
            usa_servicio_externo=bool(res.get("llamada_llm")),
            avisos=list(r.warnings), citas=citas)

    def _busqueda(self, pregunta: str) -> RespuestaChat:
        try:
            res = self.agente._literature_search({"query": pregunta, "max_results": 5})
        except Exception as exc:          # sin red, API caida...
            return RespuestaChat(texto=f"No pude consultar Crossref ni PubMed: {exc}",
                                 ruta="busqueda", fuente="Crossref / PubMed",
                                 usa_servicio_externo=True)
        papers = res.get("papers", [])
        if not papers:
            texto = "No encontré artículos en Crossref ni en PubMed para esa búsqueda."
        else:
            lineas = [f"Encontré {len(papers)} artículos (solo metadatos; no se leyó el "
                      f"texto completo):\n"]
            for p in papers:
                anio = f" ({p['anio']})" if p.get("anio") else ""
                enlace = f" · https://doi.org/{p['doi']}" if p.get("doi") else ""
                lineas.append(f"- {p.get('titulo', 'Sin título')}{anio}{enlace}")
            texto = "\n".join(lineas)
        return RespuestaChat(texto=texto, ruta="busqueda", fuente="Crossref / PubMed",
                             metodo="búsqueda bibliográfica", usa_servicio_externo=True,
                             avisos=res.get("advertencias", [])[:1])

    def _abstencion(self, scope: str, motivo: str) -> RespuestaChat:
        if scope == "fuera_de_dominio":
            texto = ("Esa pregunta está fuera del estudio, así que no la respondo. "
                     "Solo respondo sobre la encuesta de bienestar estudiantil de la "
                     "USTA y sobre los documentos académicos del corpus.")
        else:
            texto = ("No reconozco esa pregunta como una consulta sobre el estudio, así "
                     "que prefiero no responder antes que adivinar. Si es sobre la "
                     "encuesta, mencione una variable (género, seccional, estrato, nivel, "
                     "dimensiones, programa) y lo que quiere saber (resumen, comparación, "
                     "brecha...).")
        if motivo:
            texto += f"\n\n_Motivo: {motivo}_"
        texto += "\n\nEjemplos: " + " · ".join(f"«{e}»" for e in EJEMPLOS[:4])
        return RespuestaChat(texto=texto, ruta="abstencion", fuente="ninguna",
                             metodo="abstención")
