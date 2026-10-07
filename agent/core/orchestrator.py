"""
Orquestador del Agente Estadístico - Version Completa.

Conecta: LLM + StatisticalEngine + QueryPlanner + Validadores + Reportes.
"""
import json
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# scipy reparte DLLs compilados y Smart App Control puede bloquear uno de
# ellos (p. ej. scipy.sparse.csgraph._tools) sin aviso. Como scipy aqui solo
# se usa en dos tests auxiliares de supuestos, NO se le deja impedir que el
# agente arranque: si el import falla, el RAG y el resto siguen vivos y esos
# dos supuestos salen como "NO EVALUADO".
try:
    from scipy import stats as sp_stats
    SCIPY_ERROR: Optional[str] = None
except BaseException as exc:            # incluye el SystemExit de SAC
    sp_stats = None
    SCIPY_ERROR = f"{type(exc).__name__}: {exc}"

from agent.api_client import LLMClient, LLMResponse
from agent.core.config import (
    AgentConfig,
    DIMENSION_MAP,
    DIMENSION_ORDER,
    ITEM_LABELS,
    NEGATIVE_ITEMS,
    N_ITEMS,
    REPORTED_RESULTS,
)
from agent.core.query_planner import classify_scope


class AnalysisType(Enum):
    DESCRIPTIVE = "descriptivo"
    COMPARISON = "comparacion"
    CORRELATION = "correlacion"
    REGRESSION = "regresion"
    IRT = "irt"
    VALIDATION = "validacion"
    OVERVIEW = "resumen"
    PROFILE = "perfil"
    SEARCH = "busqueda_web"
    CORPUS = "corpus_literatura"
    OUT_OF_DOMAIN = "fuera_de_dominio"
    AMBIGUOUS = "ambigua"
    UNKNOWN = "desconocido"


@dataclass
class AnalysisPlan:
    analysis_type: AnalysisType
    description: str
    function_to_call: str
    parameters: Dict[str, Any]
    variables_needed: List[str]
    confidence: float


@dataclass
class AgentResponse:
    question: str
    plan: AnalysisPlan
    result: Any
    interpretation: str
    warnings: List[str]
    visualization_hint: Optional[str]
    traceability: Dict[str, Any]
    success: bool
    error: Optional[str] = None


SYSTEM_PROMPT = """Eres un agente estadistico especializado en el Indice de Bienestar Estudiantil 
de la Universidad Santo Tomas. Tu tarea es interpretar preguntas del usuario y generar 
un plan de analisis estadistico.

CONTEXTO:
- Escala de Bienestar de Ryff: 29 items, Likert 1-6
- 6 dimensiones: Autoaceptacion, Crecimiento personal, Proposito de vida, 
  Dominio del entorno, Relaciones positivas, Autonomia
- Muestra: n=1,813 estudiantes validos de N~29,950
- Diseno: muestreo estratificado por seccional x modalidad
- T-scores: T = 50 + 10 * theta (prior IRT N(0,1))

VARIABLES DISPONIBLES (nombres exactos de columnas):
- Género (Mujer/Hombre/No binario/Prefiero no decirlo)
- Edad (numerica, 18-61)
- Estrato socioeconomico (1-6)
- Nivel de formación (Pregrado/Especialización/Maestría/Técnico)
- Modalidad (Presencial/Virtual/Híbrida/A distancia)
- Semestre (1-10)
- Sede o seccional (Sede Principal Bogotá/Seccional Tunja/Seccional Villavicencio/Seccional Bucaramanga/Campus Medellín)
- seccional (Bogotá/Tunja/Villavicencio/Bucaramanga) - version simplificada
- modalidad_exp (Pregrado/Posgrado) - version simplificada
- Programa de formación (~114 programas)
- Núcleo básico de conocimiento (~29 NBC)
- factor_expansion (factores de expansion)
- 29 items de la escala (columnas con texto largo de Ryff)
- 6 dimensiones de bienestar
- T_Global (T-score global)
- T_[Dimensión] (T-scores por dimensión)

IMPORTANTE: Usa los nombres EXACTOS de las columnas listadas arriba. No inventes nombres.

VARIABLES NO DISPONIBLES (reportar si se preguntan):
- Ingreso economico
- Satisfaccion universitaria explícita
- Rendimiento academico (notas)
- Salud mental clinica

RESPONDE SIEMPRE EN JSON con esta estructura:
{
    "analysis_type": "descriptivo|comparacion|correlacion|regresion|irt|validacion|resumen|perfil",
    "description": "Descripcion breve del analisis",
    "function_to_call": "nombre_de_la_funcion",
    "parameters": {
        "value_col": "nombre_columna_objetivo",
        "group_col": "nombre_columna_agrupacion (si aplica)",
        "extra": "parametros adicionales"
    },
    "variables_needed": ["lista", "de", "variables"],
    "confidence": 0.95
}

FUNCIONES DISPONIBLES:
- "descriptive_overview": Resumen general del bienestar
- "group_comparison": Comparar T_Global entre dos grupos
- "multi_group_comparison": Comparar entre mas de 2 grupos
- "smart_comparison": Comparacion con seleccion automatica de metodo
- "correlation_analysis": Correlacion entre variables
- "regression_analysis": Modelo predictivo
- "irt_analysis": Analisis de la escala
- "dimension_profile": Perfil por dimensiones
- "validate_results": Comparar con resultados anteriores
- "full_validation": Validacion completa (psicometrica + IRT + resultados)
"""


# Prefijo explicito con el que el usuario puede forzar la consulta al corpus
# documental local: "corpus: <pregunta>", "fuentes: <pregunta>", "corpus <pregunta>".
_RE_PREFIJO_CORPUS = re.compile(
    r"^\s*(?:"
    r"corpus(?:\s|[:\-–])"                       # corpus ... / corpus: ...
    r"|(?:fuentes|literatura)\s*[:\-–]"           # fuentes: ... (separador obligatorio)
    r")\s*(?P<preg>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# Pregunta que pide lo que DICEN los documentos del corpus (no "donde encontrar
# mas literatura", que sigue siendo busqueda web). Se detecta en el texto libre
# para que "que dicen los documentos sobre X" no acabe en un analisis estadistico
# sobre la encuesta ni en la abstencion de "fuera de dominio".
_RE_PREGUNTA_CORPUS = re.compile(
    r"(?:"
    r"\bdocumentos?\b|"
    r"\bqu[eé]\s+(?:dicen|encuentran)\b|"
    r"\b(?:el|los|las)\s+corpus\b|\bdel\s+corpus\b|\ben\s+el\s+corpus\b|"
    r"\b(?:estos|esos|los)\s+art[ií]culos\s+"
    r"(?:dicen|reportan|encuentran|mencionan|aportan)\b|"
    # "literatura" como FUENTE de lo que se afirma, no como peticion de
    # buscar mas literatura. El verbo tiene que ser de CONTENIDO: mientras
    # aqui solo entran "que dice la literatura" o "segun la literatura",
    # "dame referencias/literatura sobre X" no coincide y sigue yendo a
    # _is_search_request (busqueda web). Se evalua ANTES que esa funcion,
    # por eso el patron tiene que ser preciso.
    r"\bqu[eé]\s+(?:dice|dicen|describe[n]?)\s+(?:la\s+)?literatura\b|"
    r"\b(?:seg[uú]n)\s+(?:la\s+)?literatura\b|"
    r"\b(?:en|dentro\s+de)\s+(?:la\s+)?literatura\s+"
    r"(?:se\s+)?(?:dice|describe|reporta|se[nñ]ala|encuentra)\b|"
    r"\bliteratura\s+(?:dice|describe|reporta|se[nñ]ala|indica|menciona)\b"
    r")",
    re.IGNORECASE,
)


def pregunta_sobre_corpus(question: str) -> Optional[str]:
    """Pregunta LIMPIA si el usuario forzo el corpus con prefijo; None si no."""
    m = _RE_PREFIJO_CORPUS.match(question or "")
    if not m:
        return None
    limpia = (m.group("preg") or "").strip()
    return limpia or None


class StatisticalAgent:
    """
    Agente estadistico que interpreta preguntas y ejecuta analisis.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[AgentConfig] = None,
    ):
        self.config = config or AgentConfig()
        self.llm = llm_client or LLMClient()
        self._data: Optional[pd.DataFrame] = None
        self._irt_data: Optional[pd.DataFrame] = None
        self._grm_params: Optional[pd.DataFrame] = None
        self._functions: Dict[str, Callable] = {}
        self._register_functions()
        # Indice RAG: se crea solo si alguien consulta el corpus (carga pypdf
        # no, pero si chroma + el vectorizador) y se reutiliza entre consultas.
        self._rag_retriever = None

    def load_data(self, data_path: Optional[str] = None):
        """Carga el dataset para analisis."""
        from agent.data.loader import DataLoader
        loader = DataLoader(self.config)
        
        if data_path:
            self._data = pd.read_excel(data_path)
        else:
            try:
                self._data = loader.load_prepared()
            except Exception:
                self._data = None

        try:
            self._irt_data = loader.load_irt()
        except Exception:
            self._irt_data = None

        try:
            self._grm_params = loader.load_grm_params()
        except Exception:
            self._grm_params = None

    def _register_functions(self):
        """Registra las funciones de analisis disponibles."""
        self._functions = {
            "descriptive_overview": self._descriptive_overview,
            "group_comparison": self._group_comparison,
            "multi_group_comparison": self._multi_group_comparison,
            "smart_comparison": self._smart_comparison,
            "correlation_analysis": self._correlation_analysis,
            "regression_analysis": self._regression_analysis,
            "dimension_profile": self._dimension_profile,
            "validate_results": self._validate_results,
            "full_validation": self._full_validation,
            "literature_search": self._literature_search,
        }

    def ask(self, question: str) -> AgentResponse:
        """Procesa una pregunta del usuario."""
        # Prefijo explicito de corpus ("corpus: ...", "fuentes: ...") -> RAG.
        # Se decide ANTES del planificador: una pregunta sobre la literatura
        # indexada no debe acabar en la abstencion de "fuera de dominio".
        pregunta_corpus = pregunta_sobre_corpus(question)
        if pregunta_corpus is not None:
            return self.ask_corpus(pregunta_corpus)

        traceability = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            # Se completa con el dataset REALMENTE usado en la respuesta.
            # Si no se usa dataset (busqueda web, abstencion) queda "ninguno".
            "dataset": "ninguno",
            "columns_used": [],
            "filters_applied": [],
            "method": None,
            "n_records": None,
        }

        plan = self._generate_plan(question)

        # Pregunta por lo que dicen los documentos -> RAG con cita.
        # Se resuelve aqui: no usa el dataset de la encuesta ni la funcion
        # que el planificador hubiera elegido para una variable.
        if plan.analysis_type == AnalysisType.CORPUS:
            return self.ask_corpus(plan.parameters.get("query") or question)

        if plan.analysis_type == AnalysisType.UNKNOWN:
            return AgentResponse(
                question=question, plan=plan, result=None,
                interpretation="No pude interpretar la pregunta. Podrias reformularla?",
                warnings=["Analisis no identificado"], visualization_hint=None,
                traceability=traceability, success=False,
                error="No se pudo generar un plan de analisis",
            )

        # ABSTENCION: texto fijo. NO llama al LLM ni ejecuta calculos,
        # y NO usa ningun dataset.
        if plan.function_to_call == "abstencion":
            traceability["method"] = "abstencion"
            traceability["dataset"] = "ninguno"
            traceability["columns_used"] = []
            return AgentResponse(
                question=question, plan=plan,
                result={"scope": plan.parameters.get("scope"),
                        "motivo": plan.parameters.get("motivo")},
                interpretation=self._abstention_text(plan.parameters),
                warnings=[], visualization_hint=None,
                traceability=traceability, success=True,
            )

        # Analisis que NO necesitan el dataset local
        needs_data = plan.analysis_type not in (
            AnalysisType.SEARCH, AnalysisType.OUT_OF_DOMAIN, AnalysisType.AMBIGUOUS
        )

        if needs_data and self._data is None and self._irt_data is None:
            return AgentResponse(
                question=question, plan=plan, result=None,
                interpretation="No hay datos cargados. Ejecuta load_data() primero.",
                warnings=["Datos no disponibles"], visualization_hint=None,
                traceability=traceability, success=False, error="Datos no cargados",
            )

        try:
            func = self._functions.get(plan.function_to_call)
            if func is None:
                raise ValueError(f"Funcion no encontrada: {plan.function_to_call}")

            result = func(plan.parameters)
            traceability["method"] = plan.function_to_call
            traceability["columns_used"] = plan.variables_needed

            # Registrar SOLO el dataset realmente usado en esta respuesta
            ds_name, ds_n = self._dataset_used_by(plan.function_to_call)
            traceability["dataset"] = ds_name
            traceability["n_records"] = ds_n

            interpretation = self._interpret_result(question, plan, result)

            return AgentResponse(
                question=question, plan=plan, result=result,
                interpretation=interpretation, warnings=[],
                visualization_hint=self._suggest_visualization(plan),
                traceability=traceability, success=True,
            )

        except Exception as e:
            return AgentResponse(
                question=question, plan=plan, result=None,
                interpretation=f"Error al ejecutar el analisis: {str(e)}",
                warnings=[f"Error: {str(e)}"], visualization_hint=None,
                traceability=traceability, success=False, error=str(e),
            )

    def ask_corpus(self, question: str) -> AgentResponse:
        """
        Pregunta al corpus documental local (RAG) con cita obligatoria.

        NO usa el dataset de la encuesta: busca en el indice de PDF de
        docs/rag_corpus y cita archivo + pagina. Si el mejor puntaje no
        supera el umbral calibrado, devuelve la abstinencia fija sin llamar
        al LLM (dos capas de abstinencia, igual que en agent/rag/cli.py).

        Tambien se llega aqui con el prefijo "corpus: <pregunta>" a traves
        de ask().
        """
        from agent.rag.rag_engine import ask_with_sources, umbral

        plan = self._corpus_plan(question)
        traceability = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "dataset": "docs/rag_corpus (corpus documental indexado)",
            "columns_used": [],
            "filters_applied": [],
            "method": f"rag (umbral {umbral():.4f})",
            "n_records": None,
        }

        # Un solo indice cargado y reutilizado entre consultas.
        if self._rag_retriever is None:
            try:
                from agent.rag.retriever import Retriever
                self._rag_retriever = Retriever()
            except Exception as exc:
                return AgentResponse(
                    question=question, plan=plan, result=None,
                    interpretation=f"No se pudo cargar el indice RAG: {exc}",
                    warnings=[f"Indice RAG no disponible: {exc}"],
                    visualization_hint=None, traceability=traceability,
                    success=False, error=str(exc),
                )

        try:
            resp = ask_with_sources(question, retriever=self._rag_retriever)
        except Exception as exc:
            return AgentResponse(
                question=question, plan=plan, result=None,
                interpretation=f"Error al consultar el corpus: {exc}",
                warnings=[f"Error RAG: {exc}"], visualization_hint=None,
                traceability=traceability, success=False, error=str(exc),
            )

        fuentes = [
            {"archivo": f.archivo, "pagina": f.pagina,
             "puntaje": round(f.puntaje, 4)}
            for f in resp.fuentes
        ]
        traceability["n_records"] = len(resp.fragmentos)

        lineas = [resp.respuesta, "", "--- FUENTES CITADAS ---"]
        if resp.fuentes:
            for i, f in enumerate(resp.fuentes, 1):
                lineas.append(f"  {i}. [{f.archivo}, p. {f.pagina}]  "
                              f"(puntaje {f.puntaje:.4f})")
        else:
            lineas.append("  (ninguna: hubo abstinencia)")

        warnings: List[str] = []
        if resp.abstencion:
            if resp.puntaje_max < umbral():
                warnings.append(
                    "Abstinencia dura: el mejor fragmento (%.4f) no supera el "
                    "umbral calibrado (%.4f); no se llamo al LLM."
                    % (resp.puntaje_max, umbral())
                )
            elif resp.error:
                # El fallo es de la llamada (cuota, red, modelo), no del
                # corpus: decirlo, para que no se lea como "el corpus no
                # tiene esta informacion".
                warnings.append(
                    "No se pudo responder: fallo la llamada al LLM, no el "
                    "corpus. El mejor fragmento si supera el umbral "
                    "(%.4f >= %.4f)."
                    % (resp.puntaje_max, umbral())
                )
            else:
                warnings.append(
                    "Abstinencia: el LLM no encontro la respuesta en los "
                    "fragmentos entregados (puntaje maximo %.4f)."
                    % resp.puntaje_max
                )
        if resp.error:
            warnings.append(f"Detalle: {resp.error}")

        return AgentResponse(
            question=question, plan=plan,
            result={
                "respuesta": resp.respuesta,
                "fuentes": fuentes,
                "puntaje_max": round(resp.puntaje_max, 4),
                "umbral": umbral(),
                "abstencion": resp.abstencion,
                "backend": resp.backend,
                "llamada_llm": resp.llamada_llm,
            },
            interpretation="\n".join(lineas),
            warnings=warnings, visualization_hint=None,
            traceability=traceability, success=True,
            error=resp.error,
        )

    def _dataset_used_by(self, function_to_call: str) -> Tuple[str, Optional[int]]:
        """
        Devuelve (nombre_dataset, n_registros) que usa REALMENTE esa funcion.

        - abstencion / literature_search -> no usan dataset
        - validate / full_validation     -> usan ambos datasets
        - el resto                       -> usan el dataset IRT si existe
        """
        NO_DATASET = ("abstencion", "literature_search")
        BOTH = ("validate_results", "full_validation")

        if function_to_call in NO_DATASET:
            return "ninguno", None

        names: List[str] = []
        primary: Optional[pd.DataFrame] = None

        if function_to_call in BOTH:
            if self._data is not None:
                names.append(self.config.prepared_data_file)
                primary = self._data
            if self._irt_data is not None:
                names.append(self.config.irt_data_file)
                primary = self._irt_data
        else:
            if self._irt_data is not None:
                names.append(self.config.irt_data_file)
                primary = self._irt_data
            elif self._data is not None:
                names.append(self.config.prepared_data_file)
                primary = self._data

        if not names:
            return "ninguno", None
        return " + ".join(names), (len(primary) if primary is not None else None)

    def _generate_plan(self, question: str) -> AnalysisPlan:
        """Genera un plan de analisis usando LLM o fallback por reglas."""
        # Corpus documental local: se resuelve con el indice RAG y con cita
        # de archivo y pagina. Va PRIMERO: sin esto, "que dicen los documentos
        # sobre X" caeria en la busqueda web o en la abstinencia de ambito.
        if _RE_PREGUNTA_CORPUS.search(question):
            return self._corpus_plan(question)

        # Busqueda web explicita: se detecta ANTES del LLM para no malinterpretarla
        if self._is_search_request(question):
            return self._search_plan(question)

        # Fuera de dominio o ambigua: ABSTENCION.
        # Se decide ANTES de llamar al LLM, asi no redacta sobre el dataset.
        scope, motivo = classify_scope(question)
        if scope != "en_dominio":
            return self._abstain_plan(scope, motivo)

        if not self.llm.is_configured:
            return self._fallback_plan(question)

        messages = [{"role": "user", "content": question}]
        response = self.llm.chat(messages, system_prompt=SYSTEM_PROMPT)

        if not response.success:
            return self._fallback_plan(question)

        try:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                plan_dict = json.loads(content[start:end])
                return AnalysisPlan(
                    analysis_type=AnalysisType(plan_dict.get("analysis_type", "unknown")),
                    description=plan_dict.get("description", ""),
                    function_to_call=plan_dict.get("function_to_call", "descriptive_overview"),
                    parameters=plan_dict.get("parameters", {}),
                    variables_needed=plan_dict.get("variables_needed", []),
                    confidence=plan_dict.get("confidence", 0.5),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        return self._fallback_plan(question)

    # ── Patrones de deteccion de busqueda ────────────────────────────────
    # Verbos de busqueda: "busca X", "buscar X", "búscame X"
    _RE_SEARCH_VERB = re.compile(
        r"\b(?:busca(?:r|me|s)?|b[uú]sca(?:r|me|s)?|b[uú]squeda)\b"
    )

    # Terminos academicos / de fuente: "papers", "artículos", "referencias"...
    _RE_ACADEMIC_TERM = re.compile(
        r"\b(?:papers?|art[ií]culos?|literatura|bibliograf[ií]a|"
        r"referencias?|citaciones?|citas?|publicaciones?|"
        r"doi|revistas?|scholar|crossref|pubmed|fuentes)\b"
    )

    # Peticion explicita de apoyo documental o de fuentes en la web
    _RE_REF_REQUEST = re.compile(
        r"(?:en\s+la\s+web|en\s+internet|"
        r"sustento\s+te[oó]rico|apoyos?\s+te[oó]ricos?|marco\s+te[oó]rico|"
        r"fuentes\s+(?:sobre|acerca|hay|del\s+tema)|"
        r"m[eé]\s+(?:fuentes|referencias|citas|papers|art[ií]culos|literatura)|"
        r"dame\s+(?:fuentes|referencias|citas|papers|art[ií]culos|literatura)|"
        r"autores\s+(?:sobre|de)\s+)",
        re.IGNORECASE,
    )

    # Fuentes del PROPIO estudio: NO son busqueda web
    _RE_LOCAL_SOURCES = re.compile(
        r"(?:fuentes?\s+(?:del\s+(?:estudio|proyecto|an[aá]lisis|paper|documento)|"
        r"usad[oa]s?\s+por|usa|tiene)|"
        r"de\s+d[oó]nde\s+(?:salen|provienen|se\s+obtienen)|"
        r"qu[eé]\s+datos\s+(?:se\s+us[oó]|hay))",
        re.IGNORECASE,
    )

    def _is_search_request(self, question: str) -> bool:
        """
        Detecta si el usuario pide buscar en la web / referencias del tema.

        Reglas:
        1. Verbo de busqueda explicito -> busqueda
        2. Termino academico (papers, referencias, artículos...) -> busqueda
        3. Peticion de apoyo documental ("dame referencias", "marco teórico") -> busqueda
        4. EXCEPCION: si pregunta por las fuentes DEL propio estudio, no es
           busqueda web (se responde con documentacion local del proyecto)
        """
        q = question.lower().strip()

        if self._RE_LOCAL_SOURCES.search(q):
            return False

        if self._RE_SEARCH_VERB.search(q):
            return True
        if self._RE_REF_REQUEST.search(q):
            return True
        if self._RE_ACADEMIC_TERM.search(q):
            return True

        return False

    def _search_plan(self, question: str) -> AnalysisPlan:
        """Plan para una busqueda web de literatura academica."""
        return AnalysisPlan(
            analysis_type=AnalysisType.SEARCH,
            description="Busqueda de literatura academica en Crossref y PubMed",
            function_to_call="literature_search",
            parameters={"query": question, "max_results": 5},
            variables_needed=[],
            confidence=1.0,
        )

    def _corpus_plan(self, question: str) -> AnalysisPlan:
        """Plan para consultar el corpus documental local (RAG con cita)."""
        return AnalysisPlan(
            analysis_type=AnalysisType.CORPUS,
            description="Consulta al corpus documental local (RAG con cita)",
            function_to_call="corpus_answer",
            parameters={"query": question},
            variables_needed=[],
            confidence=1.0,
        )

    def _abstain_plan(self, scope: str, motivo: str) -> AnalysisPlan:
        """
        Plan de ABSTENCION: fuera de dominio o ambigua.

        No ejecuta calculos ni llama al LLM para redactar.
        """
        es_fuera = scope == "fuera_de_dominio"
        return AnalysisPlan(
            analysis_type=(
                AnalysisType.OUT_OF_DOMAIN if es_fuera else AnalysisType.AMBIGUOUS
            ),
            description=(
                "Pregunta fuera del dominio del dataset"
                if es_fuera
                else "Pregunta ambigua, no se reconoce el analisis"
            ),
            function_to_call="abstencion",
            parameters={"scope": scope, "motivo": motivo},
            variables_needed=[],
            confidence=1.0,
        )

    def _abstention_text(self, params: Dict) -> str:
        """
        Texto FIJO de abstencion. No usa LLM ni calculos estadisticos.
        """
        scope = params.get("scope", "fuera_de_dominio")
        motivo = params.get("motivo", "La pregunta no corresponde al dataset.")

        ejemplos = (
            "  1. ¿Cual es la media global de bienestar (T-score)?\n"
            "  2. ¿Hay diferencia de bienestar entre hombres y mujeres?\n"
            "  3. ¿Que dimension de Ryff tiene el puntaje mas bajo?\n"
            "  4. ¿Que factores predicen el bienestar?"
        )

        if scope == "fuera_de_dominio":
            return (
                "[FUERA DE DOMINIO - NO SE EJECUTO NINGUN ANALISIS]\n"
                "\n"
                "No puedo responder esta pregunta con el agente estadistico.\n"
                "\n"
                f"Motivo: {motivo}\n"
                "\n"
                "El dataset contiene SOLO respuestas de la encuesta de bienestar\n"
                "(1,813 estudiantes validos de ~29,950). No contiene literatura,\n"
                "bibliografia, directorios de fuentes ni datos de otros paises.\n"
                "\n"
                "Que SI puedo responder (ejemplos con el dataset):\n"
                f"{ejemplos}\n"
                "\n"
                'Para buscar literatura externa use: "dame referencias sobre <tema>".\n'
                "\n"
                "Si lo que busca es lo que DICEN los articulos ya indexados\n"
                '(20 PDF del corpus local), use: "corpus <su pregunta>".\n'
                "Ese camino SI cita archivo y pagina, y se abstiene solo si\n"
                "ningun fragmento supera el umbral calibrado."
            )

        # ambigua
        return (
            "[PREGUNTA AMBIGUA - NO SE EJECUTO NINGUN ANALISIS]\n"
            "\n"
            "No reconozco con claridad que analisis pedir, asi que no ejecute\n"
            "ningun calculo ni genere una respuesta sobre el dataset.\n"
            "\n"
            f"Motivo: {motivo}\n"
            "\n"
            "Para que pueda responder, incluya en la pregunta:\n"
            "  - una variable del dataset: T-score, dimensiones de Ryff, genero,\n"
            "    edad, estrato, sede/seccional, modalidad, programa o semestre\n"
            "  - y el analisis: resumen, comparacion, correlacion o regresion\n"
            "\n"
            "Ejemplos validos:\n"
            f"{ejemplos}"
        )

    def _literature_search(self, params: Dict) -> Dict[str, Any]:
        """
        Busca literatura academica en la web (Crossref + PubMed).

        Solo usa APIs academicas gratuitas. NO es navegacion web general.
        """
        from agent.literature.searcher import LiteratureSearcher

        query = params.get("query", "")
        max_results = int(params.get("max_results", 5))

        searcher = LiteratureSearcher()
        results = searcher.auto_search(query, max_results=max_results)

        papers = [
            {
                "titulo": r.title,
                "autores": r.authors,
                "anio": r.year,
                "fuente": r.source,
                "doi": r.doi,
                "url": r.url,
                "resumen": r.abstract,
            }
            for r in results
        ]

        if not papers:
            return {
                "query": query,
                "n_resultados": 0,
                "papers": [],
                "bases_consultadas": ["Crossref", "PubMed"],
                "advertencias": [
                    "No se encontraron papers. La busqueda usa APIs "
                    "academicas gratuitas (Crossref/PubMed), no la web general."
                ],
            }

        return {
            "query": query,
            "n_resultados": len(papers),
            "papers": papers,
            "bases_consultadas": ["Crossref", "PubMed"],
            "metodo_seleccionado": "Busqueda bibliografica (Crossref, fallback PubMed)",
            "razon": (
                "Se consultan bases academicas indexadas con DOI verificable. "
                "Crossref es la fuente primaria por cobertura y velocidad; "
                "PubMed solo se usa si Crossref no retorna resultados."
            ),
            "advertencias": [
                "El DOI confirma la IDENTIDAD de la fuente (existe y es localizable), "
                "NO garantiza que sea pares revisada ni que el contenido sustente "
                "una afirmacion concreta. Eso requiere leer el documento.",
                "La busqueda cubre solo bases indexadas (Crossref/PubMed); "
                "no es navegacion web general ni incluye todo el contenido de internet.",
                "No se descargaron ni se leyeron los textos completos: solo metadatos "
                "(titulo, autores, ano, fuente, DOI).",
            ],
            "supuestos_verificados": [
                {
                    "nombre": "Identificabilidad de la fuente",
                    "passed": bool(papers[0].get("doi") or papers[0].get("url")),
                    "interpretacion": (
                        "Toda fuente tiene DOI o URL verificable"
                        if (papers[0].get("doi") or papers[0].get("url"))
                        else "Alguna fuente sin DOI/URL: no verificable"
                    ),
                },
                {
                    "nombre": "Procedencia academica",
                    "passed": True,
                    "interpretacion": (
                        "Resultados provenientes de bases indexadas "
                        "(Crossref/PubMed), no de un generador de texto"
                    ),
                },
                {
                    "nombre": "Sustento del contenido",
                    "passed": False,
                    "interpretacion": (
                        "NO verificado: solo se tienen metadatos, no el texto "
                        "completo. No se puede afirmar que el paper diga algo "
                        "concreto sin leerlo."
                    ),
                },
            ],
        }

    def _find_column(self, candidates: List[str]) -> str:
        """Busca la primera columna que exista en los datos."""
        df = self._irt_data if self._irt_data is not None else self._data
        if df is None:
            return candidates[0]
        for c in candidates:
            if c in df.columns:
                return c
        return candidates[0]

    def _fallback_plan(self, question: str) -> AnalysisPlan:
        """Plan de respaldo basado en reglas simples."""
        q = question.lower()

        # Busqueda web explicita (red de seguridad si se llega por esta via)
        if self._is_search_request(question):
            return self._search_plan(question)

        # Fuera de dominio / ambigua (red de seguridad)
        scope, motivo = classify_scope(question)
        if scope != "en_dominio":
            return self._abstain_plan(scope, motivo)

        if any(w in q for w in ["resumen", "general", "global", "descripcion", "como va"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.OVERVIEW,
                description="Resumen general del bienestar",
                function_to_call="descriptive_overview",
                parameters={}, variables_needed=["T_Global"], confidence=0.7,
            )
        elif any(w in q for w in ["valida", "reproduce", "compara con", "correcto"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.VALIDATION,
                description="Validacion de resultados",
                function_to_call="validate_results",
                parameters={}, variables_needed=["T_Global"], confidence=0.8,
            )
        elif any(w in q for w in ["validacion completa", "todo", "integral"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.VALIDATION,
                description="Validacion completa del pipeline",
                function_to_call="full_validation",
                parameters={}, variables_needed=["T_Global"], confidence=0.9,
            )
        elif any(w in q for w in ["genero", "hombre", "mujer", "sexo", "masculino", "femenino"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.COMPARISON,
                description="Comparacion por genero",
                function_to_call="smart_comparison",
                parameters={"value_col": "T_Global", "group_col": "Género"},
                variables_needed=["T_Global", "Género"], confidence=0.8,
            )
        elif any(w in q for w in ["seccional", "sede", "bogota", "tunja", "bucaramanga", "villavicencio"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.COMPARISON,
                description="Comparacion por seccional",
                function_to_call="smart_comparison",
                parameters={"value_col": "T_Global", "group_col": "seccional"},
                variables_needed=["T_Global", "seccional"], confidence=0.8,
            )
        elif any(w in q for w in ["modalidad", "pregrado", "posgrado"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.COMPARISON,
                description="Comparacion por modalidad",
                function_to_call="smart_comparison",
                parameters={"value_col": "T_Global", "group_col": "modalidad_exp"},
                variables_needed=["T_Global", "modalidad_exp"], confidence=0.8,
            )
        elif any(w in q for w in ["regresion", "predice", "predicen", "factor", "modelo"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.REGRESSION,
                description="Analisis de regresion",
                function_to_call="regression_analysis",
                parameters={"value_col": "T_Global", "group_col": "Edad"},
                variables_needed=["T_Global", "Edad"], confidence=0.7,
            )
        elif any(w in q for w in ["estrato", "socioeconomico"]):
            # Buscar la columna correcta
            df_cols = self._irt_data.columns if self._irt_data is not None else (self._data.columns if self._data is not None else [])
            estrato_col = None
            for col in df_cols:
                if 'estrato' in col.lower() and 'socioecon' in col.lower():
                    estrato_col = col
                    break
            if not estrato_col:
                estrato_col = "Estrato socioeconomico"
            return AnalysisPlan(
                analysis_type=AnalysisType.COMPARISON,
                description="Comparacion por estrato",
                function_to_call="multi_group_comparison",
                parameters={"value_col": "T_Global", "group_col": estrato_col},
                variables_needed=["T_Global", estrato_col], confidence=0.8,
            )
        elif any(w in q for w in ["edad"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.CORRELATION,
                description="Correlacion edad-bienestar",
                function_to_call="correlation_analysis",
                parameters={"value_col": "T_Global", "group_col": "Edad"},
                variables_needed=["T_Global", "Edad"], confidence=0.7,
            )
        elif any(w in q for w in ["dimension", "perfil", "autoaceptacion", "autonomia"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.PROFILE,
                description="Perfil por dimensiones",
                function_to_call="dimension_profile",
                parameters={},
                variables_needed=["T_Global"] + [f"T_{d}" for d in DIMENSION_ORDER],
                confidence=0.7,
            )
        elif any(w in q for w in ["regresion", "predice", "factor", "modelo"]):
            return AnalysisPlan(
                analysis_type=AnalysisType.REGRESSION,
                description="Analisis de regresion",
                function_to_call="regression_analysis",
                parameters={"value_col": "T_Global", "group_col": "Edad"},
                variables_needed=["T_Global", "Edad"], confidence=0.7,
            )
        else:
            # La rama por defecto YA NO es descriptive_overview.
            # descriptive_overview solo se ejecuta si la pregunta lo pide
            # de forma explicita (rama "resumen/general/global/descripcion").
            return self._abstain_plan(
                "ambigua",
                "La pregunta toca variables del dataset pero no pide de forma "
                "explicita un analisis (resumen, comparacion, correlacion o "
                "regresion).",
            )

    def _descriptive_overview(self, params: Dict) -> Dict[str, Any]:
        """Resumen general del bienestar."""
        df = self._irt_data if self._irt_data is not None else self._data
        if df is None or "T_Global" not in df.columns:
            return {"error": "T_Global no disponible"}

        t = df["T_Global"]
        return {
            "n": len(t),
            "media": round(float(t.mean()), 2),
            "mediana": round(float(t.median()), 2),
            "desv_std": round(float(t.std()), 2),
            "min": round(float(t.min()), 2),
            "max": round(float(t.max()), 2),
            "pct_bajo": round(float((t < 40).mean() * 100), 1),
            "pct_alto": round(float((t >= 60).mean() * 100), 1),
            "clasificacion": self._classify_t(float(t.mean())),
        }

    def _group_comparison(self, params: Dict) -> Dict[str, Any]:
        """Compara dos grupos."""
        df = self._irt_data if self._irt_data is not None else self._data
        if df is None:
            return {"error": "Datos no disponibles"}

        value_col = params.get("value_col", "T_Global")
        group_col = params.get("group_col")

        if group_col not in df.columns:
            return {"error": f"Variable '{group_col}' no encontrada"}

        from agent.statistics.inference import InferenceEngine
        engine = InferenceEngine()
        test = engine.compare_two_groups(df, value_col, group_col)

        result = {}
        for g in df[group_col].unique():
            subset = df[df[group_col] == g][value_col].dropna()
            result[str(g)] = {
                "n": int(len(subset)),
                "media": round(float(subset.mean()), 2),
                "desv_std": round(float(subset.std()), 2),
            }

        result["prueba_estadistica"] = {
            "test": test.test_name,
            "estadistico": round(test.statistic, 4),
            "p_valor": round(test.p_value, 4),
            "tamano_efecto": round(test.effect_size, 4) if test.effect_size else None,
            "significativo": test.significant,
        }

        return result

    def _multi_group_comparison(self, params: Dict) -> Dict[str, Any]:
        """Compara mas de 2 grupos."""
        df = self._irt_data if self._irt_data is not None else self._data
        if df is None:
            return {"error": "Datos no disponibles"}

        value_col = params.get("value_col", "T_Global")
        group_col = params.get("group_col")

        if group_col not in df.columns:
            return {"error": f"Variable '{group_col}' no encontrada"}

        from agent.statistics.inference import InferenceEngine
        engine = InferenceEngine()
        test = engine.compare_multiple_groups(df, value_col, group_col)

        group_stats = {}
        for g in df[group_col].unique():
            subset = df[df[group_col] == g][value_col].dropna()
            group_stats[str(g)] = {
                "n": int(len(subset)),
                "media": round(float(subset.mean()), 2),
                "desv_std": round(float(subset.std()), 2),
            }

        return {
            "estadisticos_por_grupo": group_stats,
            "prueba_estadistica": {
                "test": test.test_name,
                "estadistico": round(test.statistic, 4),
                "p_valor": round(test.p_value, 4),
                "tamano_efecto": round(test.effect_size, 4) if test.effect_size else None,
                "significativo": test.significant,
            },
        }

    def _smart_comparison(self, params: Dict) -> Dict[str, Any]:
        """Comparacion con seleccion automatica de metodo."""
        df = self._irt_data if self._irt_data is not None else self._data
        if df is None:
            return {"error": "Datos no disponibles"}

        value_col = params.get("value_col", "T_Global")
        group_col = params.get("group_col")

        if group_col not in df.columns:
            return {"error": f"Variable '{group_col}' no encontrada"}

        from agent.statistics.statistical_engine import StatisticalEngine
        engine = StatisticalEngine()
        
        plan, test_result = engine.run_comparison(
            df, value_col, group_col, "factor_expansion"
        )

        group_stats = {}
        for g in df[group_col].unique():
            subset = df[df[group_col] == g][value_col].dropna()
            group_stats[str(g)] = {
                "n": int(len(subset)),
                "media": round(float(subset.mean()), 2),
                "desv_std": round(float(subset.std()), 2),
            }

        return {
            "metodo_seleccionado": plan.test_name,
            "razon": plan.reason,
            "alternativa": plan.alternative,
            "supuestos_verificados": [
                {"nombre": a.name, "passed": a.passed, "interpretacion": a.interpretation}
                for a in plan.assumptions_checked
            ],
            "estadisticos_por_grupo": group_stats,
            "prueba_estadistica": {
                "test": test_result.test_name,
                "estadistico": round(test_result.statistic, 4),
                "p_valor": round(test_result.p_value, 4),
                "tamano_efecto": round(test_result.effect_size, 4) if test_result.effect_size else None,
                "significativo": test_result.significant,
                "interpretacion": test_result.interpretation,
            } if test_result else None,
            "advertencias": plan.warnings,
        }

    def _correlation_analysis(self, params: Dict) -> Dict[str, Any]:
        """Analisis de correlacion."""
        df = self._irt_data if self._irt_data is not None else self._data
        if df is None:
            return {"error": "Datos no disponibles"}

        from agent.statistics.statistical_engine import StatisticalEngine
        engine = StatisticalEngine()
        
        value_col = params.get("value_col", "T_Global")
        group_col = params.get("group_col", "Edad")

        if group_col not in df.columns:
            return {"error": f"Variable '{group_col}' no encontrada"}

        from scipy.stats import pearsonr, spearmanr
        mask = df[[value_col, group_col]].dropna().index
        x = df.loc[mask, value_col]
        y = df.loc[mask, group_col]
        
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)

        return {
            "variables": [value_col, group_col],
            "n": len(mask),
            "pearson": {"r": round(r_p, 4), "p": round(p_p, 4)},
            "spearman": {"rho": round(r_s, 4), "p": round(p_s, 4)},
        }

    def _regression_analysis(self, params: Dict) -> Dict[str, Any]:
        """Analisis de regresion con verificacion de supuestos."""
        df = self._irt_data if self._irt_data is not None else self._data
        if df is None:
            return {"error": "Datos no disponibles"}

        from agent.statistics.regression import RegressionEngine
        engine = RegressionEngine()

        x_cols = []
        for col in df.columns:
            if col.lower() in ["edad", "estrato socioeconomico", "estrato socioeconómico"]:
                x_cols.append(col)

        if not x_cols:
            return {"error": "No hay variables predictoras numericas disponibles"}

        # Seleccionar metodo verificando supuestos
        metodo = engine.auto_select_method(df, "T_Global", x_cols, "factor_expansion")

        # Verificar supuestos
        supuestos = []
        
        # Verificar datos faltantes
        data = df[["T_Global"] + x_cols].dropna()
        n_total = len(df)
        n_valid = len(data)
        pct_missing = ((n_total - n_valid) / n_total) * 100
        
        supuestos.append({
            "nombre": "Datos faltantes",
            "test": "Verificacion de completitud",
            "estadistico": round(pct_missing, 2),
            "p_value": None,
            "passed": pct_missing < 10,
            "interpretacion": f"{pct_missing:.1f}% datos faltantes ({n_total - n_valid} de {n_total})",
        })

        if metodo == "ols":
            # Normalidad de residuos
            y = data["T_Global"].values
            X = data[x_cols].values
            X_with_const = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            residuals = y - X_with_const @ beta

            if sp_stats is None:
                motivo = f"NO EVALUADO: scipy no se pudo cargar ({SCIPY_ERROR})"
                supuestos.append({
                    "nombre": "Normalidad de residuos",
                    "test": "Shapiro-Wilk",
                    "estadistico": None, "p_value": None, "passed": None,
                    "interpretacion": motivo,
                })
                supuestos.append({
                    "nombre": "Homogeneidad de varianzas",
                    "test": "Breusch-Pagan",
                    "estadistico": None, "p_value": None, "passed": None,
                    "interpretacion": motivo,
                })
            else:
                stat, p = sp_stats.shapiro(
                    residuals[:min(5000, len(residuals))])
                supuestos.append({
                    "nombre": "Normalidad de residuos",
                    "test": "Shapiro-Wilk",
                    "estadistico": round(stat, 4),
                    "p_value": round(p, 4),
                    "passed": p > 0.05,
                    "interpretacion": "Residuos normales" if p > 0.05 else "Residuos NO normales",
                })

                # Homogeneidad de varianzas
                abs_resid = np.abs(residuals)
                stat_bp, p_bp = sp_stats.pearsonr(y, abs_resid)
                supuestos.append({
                    "nombre": "Homogeneidad de varianzas",
                    "test": "Breusch-Pagan",
                    "estadistico": round(stat_bp, 4),
                    "p_value": round(p_bp, 4),
                    "passed": p_bp > 0.05,
                    "interpretacion": "Varianzas homogeneas" if p_bp > 0.05 else "Varianzas NO homogeneas",
                })

        # Ejecutar regresion
        if metodo == "ols":
            result = engine.ols(df, "T_Global", x_cols)
        elif metodo == "wls":
            result = engine.wls(df, "T_Global", x_cols, "factor_expansion")
        else:
            result = engine.ols(df, "T_Global", x_cols)

        # Formatear coeficientes de forma legible
        coef_dict = {}
        if hasattr(result, 'coefficients') and hasattr(result.coefficients, 'to_dict'):
            raw_coef = result.coefficients.to_dict()
            # El resultado viene como {metrica: {variable: valor}}
            if "coef" in raw_coef:
                for var, val in raw_coef["coef"].items():
                    p_val = raw_coef.get("p", {}).get(var, "N/A")
                    se = raw_coef.get("std_err", {}).get(var, "N/A")
                    coef_dict[var] = {
                        "beta": round(val, 4),
                        "p_value": round(p_val, 4) if isinstance(p_val, (int, float)) else p_val,
                        "std_error": round(se, 4) if isinstance(se, (int, float)) else se,
                    }

        return {
            "metodo_seleccionado": metodo.upper(),
            "razon": "Variable dependiente continua" if metodo == "ols" else "Pesos de diseno muestral",
            "supuestos_verificados": supuestos,
            "modelo": result.model_type,
            "n_obs": len(data),
            "r_cuadrado": round(result.r_squared, 4) if result.r_squared else None,
            "r_cuadrado_ajustado": round(result.adj_r_squared, 4) if hasattr(result, 'adj_r_squared') else None,
            "f_estadistico": round(result.f_statistic, 4) if hasattr(result, 'f_statistic') else None,
            "f_p_value": round(result.f_p_value, 4) if hasattr(result, 'f_p_value') else None,
            "coeficientes": coef_dict,
            "advertencias": [],
        }

    def _dimension_profile(self, params: Dict) -> Dict[str, Any]:
        """Perfil por dimensiones."""
        df = self._irt_data if self._irt_data is not None else self._data
        if df is None:
            return {"error": "Datos no disponibles"}

        profile = {}
        for dim in DIMENSION_ORDER:
            col = f"T_{dim}"
            if col in df.columns:
                t = df[col]
                profile[dim] = {
                    "media": round(float(t.mean()), 2),
                    "desv_std": round(float(t.std()), 2),
                    "pct_bajo": round(float((t < 40).mean() * 100), 1),
                    "pct_alto": round(float((t >= 60).mean() * 100), 1),
                }

        if profile:
            min_dim = min(profile, key=lambda x: profile[x]["media"])
            max_dim = max(profile, key=lambda x: profile[x]["media"])
            profile["_resumen"] = {
                "dimension_mas_baja": min_dim,
                "dimension_mas_alta": max_dim,
            }

        return profile

    def _validate_results(self, params: Dict) -> Dict[str, Any]:
        """Valida contra resultados reportados."""
        from agent.validation.result_validator import ResultValidator
        validator = ResultValidator()

        rows = validator.validate_all(
            df_prepared=self._data,
            df_irt=self._irt_data,
        )

        table = validator.generate_comparison_table(rows)
        summary = validator.summary(rows)

        return {
            "tabla": table.to_dict(orient="records"),
            "resumen": summary,
        }

    def _full_validation(self, params: Dict) -> Dict[str, Any]:
        """Validacion completa del pipeline."""
        results = {}

        # 1. Validacion de datos
        from agent.data.validator import DataValidator
        data_validator = DataValidator()
        
        if self._data is not None:
            data_report = data_validator.validate_prepared(self._data)
            results["validacion_datos"] = {
                "passed": data_report.passed,
                "checks": data_report.n_checks,
                "passed_checks": data_report.n_passed,
                "failed_checks": data_report.n_failed,
                "resumen": data_report.summary(),
            }

        # 2. Validacion psicometrica
        from agent.validation.psychometric_validator import PsychometricValidator
        psych_validator = PsychometricValidator()
        
        if self._data is not None:
            from agent.data.loader import DataLoader
            loader = DataLoader(self.config)
            try:
                item_cols = loader.get_item_columns(self._data)
                psych_results = psych_validator.validate_all(
                    self._data, item_cols, "factor_expansion"
                )
                results["validacion_psicometrica"] = {
                    "checks": len(psych_results),
                    "passed": sum(1 for r in psych_results if r.passed),
                    "failed": sum(1 for r in psych_results if not r.passed),
                    "resumen": psych_validator.summary(psych_results),
                }
            except Exception as e:
                results["validacion_psicometrica"] = {"error": str(e)}

        # 3. Validacion IRT
        from agent.validation.irt_validator import IRTValidator
        irt_validator = IRTValidator()
        
        if self._irt_data is not None:
            from agent.data.loader import DataLoader
            loader = DataLoader(self.config)
            try:
                item_cols = loader.get_item_columns(self._data) if self._data is not None else []
                irt_results = irt_validator.validate_all(
                    self._irt_data, item_cols, self._grm_params
                )
                results["validacion_irt"] = {
                    "checks": len(irt_results),
                    "passed": sum(1 for r in irt_results if r.passed),
                    "failed": sum(1 for r in irt_results if not r.passed),
                    "resumen": irt_validator.summary(irt_results),
                }
            except Exception as e:
                results["validacion_irt"] = {"error": str(e)}

        # 4. Validacion de pesos
        from agent.validation.survey_weights import SurveyWeightValidator
        weight_validator = SurveyWeightValidator()
        
        if self._data is not None:
            weight_results = weight_validator.validate(self._data)
            results["validacion_pesos"] = {
                "checks": len(weight_results),
                "passed": sum(1 for r in weight_results if r.passed),
                "failed": sum(1 for r in weight_results if not r.passed),
            }

        # 5. Validacion de resultados
        from agent.validation.result_validator import ResultValidator
        result_validator = ResultValidator()
        rows = result_validator.validate_all(
            df_prepared=self._data,
            df_irt=self._irt_data,
        )
        results["validacion_resultados"] = {
            "tabla": result_validator.generate_comparison_table(rows).to_dict(orient="records"),
            "resumen": result_validator.summary(rows),
        }

        return results

    def _classify_t(self, t_score: float) -> str:
        if t_score < 40:
            return "Muy bajo"
        elif t_score < 45:
            return "Bajo"
        elif t_score < 55:
            return "Medio"
        elif t_score < 60:
            return "Alto"
        else:
            return "Muy alto"

    def _interpret_result(self, question: str, plan: AnalysisPlan, result: Dict) -> str:
        """Genera interpretacion del resultado."""
        # SIEMPRE generar resumen estadistico primero
        stats_summary = self._generate_stats_summary(plan, result)
        
        if not self.llm.is_configured:
            return stats_summary

        # Convertir result a JSON manejando numpy types
        def default_serializer(obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        result_json = json.dumps(result, default=default_serializer, ensure_ascii=False)

        if plan.analysis_type == AnalysisType.SEARCH:
            prompt = f"""Pregunta original: {question}

Resultado de la busqueda (solo METADATOS, no se leyeron los articulos):
{result_json}

Genera una interpretacion en espanol. REGLAS ESTRICTAS ANTE LA ALUCINACION:
- NO afirmes que un articulo ha sido revisado por pares solo por tener DOI.
- NO afirmes que un articulo dice, demuestra o concluye algo concreto:
  no se descargo ni se leyo ningun texto completo, solo metadatos.
- NO inventes hallazgos, conclusiones, numeros o porcentajes.
- NO inventes autores, anos, revistas ni DOIs que no esten en el JSON.
- Describe unicamente: cuantos resultados hay, de que tratan segun su titulo,
  y de donde vienen (base de datos).
- Indica explicitamente que verificar cada fuente exige leer el documento."""
            system_prompt = (
                "Eres un verificador de fuentes riguroso. Describes solo lo que "
                "esta en los metadatos y marcas todo lo no verificado como tal."
            )
        else:
            prompt = f"""Pregunta original: {question}

Tipo de analisis: {plan.description}

Resultado obtenido:
{result_json}

Genera una interpretacion clara y concisa en espanol, dirigida a un usuario no estadistico.
Incluye:
1. Que significa en terminos practicos
2. Limitaciones si las hay"""
            system_prompt = "Eres un estadistico que explica resultados de forma simple y precisa."

        messages = [{"role": "user", "content": prompt}]
        response = self.llm.chat(messages, system_prompt=system_prompt)

        # SIEMPRE devolver el resumen estadistico + interpretacion del LLM
        if response.success:
            return stats_summary + "\n\n--- Interpretacion en lenguaje simple ---\n" + response.content
        return stats_summary

    def _generate_stats_summary(self, plan: AnalysisPlan, result: Dict) -> str:
        """Genera resumen estadistico detallado SIEMPRE."""
        interp = []
        interp.append(f"[METODO: {plan.description}]")
        interp.append("")

        # Supuestos verificados
        supuestos = result.get("supuestos_verificados", [])
        if supuestos:
            interp.append("--- Supuestos verificados ---")
            for s in supuestos:
                _p = s.get("passed")
                estado = ("NO EVALUADO" if _p is None
                          else ("CUMPLE" if _p else "NO CUMPLE"))
                interp.append(f"  [{estado}] {s.get('nombre')}: {s.get('interpretacion')}")
            interp.append("")

        # Metodo seleccionado
        metodo_seleccionado = result.get("metodo_seleccionado")
        razon = result.get("razon")
        if metodo_seleccionado:
            interp.append(f"Prueba seleccionada: {metodo_seleccionado}")
            if razon:
                interp.append(f"Razon: {razon}")
            interp.append("")

        # Advertencias
        advertencias = result.get("advertencias", [])
        if advertencias:
            interp.append("--- Advertencias ---")
            for adv in advertencias:
                interp.append(f"  - {adv}")
            interp.append("")

        # Resultados especificos segun tipo
        if plan.analysis_type == AnalysisType.SEARCH:
            interp.append("--- Fuentes encontradas en la web ---")
            interp.append(f"Consulta: {result.get('query', 'N/A')}")
            interp.append(f"Bases consultadas: {', '.join(result.get('bases_consultadas', []))}")
            interp.append(f"Resultados: {result.get('n_resultados', 0)}")
            interp.append("")
            for i, p in enumerate(result.get("papers", []), 1):
                anio = f" ({p.get('anio')})" if p.get("anio") else ""
                interp.append(f"{i}. {p.get('titulo', 'Sin titulo')}{anio}")
                interp.append(f"   Autores: {p.get('autores', 'N/A')}")
                interp.append(f"   Fuente: {p.get('fuente', 'N/A')}")
                interp.append(f"   DOI: {p.get('doi') or 'N/A'}")
                if p.get("url"):
                    interp.append(f"   URL: {p.get('url')}")
                interp.append("")

        elif plan.analysis_type == AnalysisType.OVERVIEW:
            interp.append("--- Resultados ---")
            interp.append(f"Media T-score: {result.get('media', 'N/A')}")
            interp.append(f"Clasificacion: {result.get('clasificacion', 'N/A')}")
            interp.append(f"En riesgo (T<40): {result.get('pct_bajo', 0)}%")
            interp.append(f"Bienestar alto (T>=60): {result.get('pct_alto', 0)}%")

        elif plan.analysis_type == AnalysisType.COMPARISON:
            interp.append("--- Resultados ---")
            prueba = result.get("prueba_estadistica", {})
            interp.append(f"Prueba: {prueba.get('test', 'N/A')}")
            interp.append(f"Estadistico: {prueba.get('estadistico', 'N/A')}")
            interp.append(f"p-value: {prueba.get('p_valor', 'N/A')}")
            interp.append(f"Tamano del efecto: {prueba.get('tamano_efecto', 'N/A')}")
            interp.append(f"Significativo: {'Si' if str(prueba.get('significativo', '')).lower() == 'true' else 'No'}")
            interp.append("")
            interp.append("--- Datos por grupo ---")
            estadisticos_grupo = result.get("estadisticos_por_grupo", {})
            if estadisticos_grupo:
                for key, val in estadisticos_grupo.items():
                    interp.append(f"  - {key}: n={val.get('n', 'N/A')}, media={val.get('media', 'N/A')}, SD={val.get('desv_std', 'N/A')}")

        elif "r_cuadrado" in result:
            interp.append("--- Resultados del modelo ---")
            interp.append(f"Tipo: {result.get('metodo_seleccionado', 'N/A')}")
            interp.append(f"Observaciones: {result.get('n_obs', 'N/A')}")
            interp.append(f"R-cuadrado: {result.get('r_cuadrado', 'N/A')}")
            if result.get('r_cuadrado_ajustado'):
                interp.append(f"R-cuadrado ajustado: {result.get('r_cuadrado_ajustado')}")
            if result.get('f_estadistico'):
                interp.append(f"F-estadistico: {result.get('f_estadistico')}")
                interp.append(f"F p-value: {result.get('f_p_value')}")
            interp.append("")
            interp.append("--- Coeficientes ---")
            coef = result.get("coeficientes", {})
            for var, vals in coef.items():
                if isinstance(vals, dict):
                    interp.append(f"  - {var}: beta={vals.get('beta')}, p={vals.get('p_value')}, SE={vals.get('std_error')}")

        elif plan.analysis_type == AnalysisType.PROFILE:
            interp.append("--- Perfil de dimensiones ---")
            perfil = result
            if "_resumen" in perfil:
                res = perfil["_resumen"]
                interp.append(f"Dimension mas alta: {res['dimension_mas_alta']} (T={perfil[res['dimension_mas_alta']]['media']})")
                interp.append(f"Dimension mas baja: {res['dimension_mas_baja']} (T={perfil[res['dimension_mas_baja']]['media']})")

        else:
            interp.append("Analisis completado.")

        return "\n".join(interp)

    def _basic_interpretation(self, plan: AnalysisPlan, result: Dict) -> str:
        """Interpretacion basica sin LLM."""
        if "error" in result:
            return f"Error: {result['error']}"

        interp = []
        interp.append(f"[METODO: {plan.description}]")
        interp.append("")

        # SIEMPRE mostrar supuestos verificados si existen
        supuestos = result.get("supuestos_verificados", [])
        if supuestos:
            interp.append("--- Supuestos verificados ---")
            for s in supuestos:
                _p = s.get("passed")
                estado = ("NO EVALUADO" if _p is None
                          else ("CUMPLE" if _p else "NO CUMPLE"))
                interp.append(f"  [{estado}] {s.get('nombre')}: {s.get('interpretacion')}")
            interp.append("")

        # SIEMPRE mostrar metodo seleccionado y por que
        metodo_seleccionado = result.get("metodo_seleccionado")
        razon = result.get("razon")
        if metodo_seleccionado:
            interp.append(f"Prueba seleccionada: {metodo_seleccionado}")
            if razon:
                interp.append(f"Razon: {razon}")
            interp.append("")

        # Mostrar advertencias
        advertencias = result.get("advertencias", [])
        if advertencias:
            interp.append("--- Advertencias ---")
            for adv in advertencias:
                interp.append(f"  - {adv}")
            interp.append("")

        if plan.analysis_type == AnalysisType.OVERVIEW:
            interp.append(
                f"El bienestar global tiene una media de T={result.get('media', 'N/A')} "
                f"({result.get('clasificacion', 'N/A')}). "
                f"El {result.get('pct_bajo', 0)}% esta en riesgo (T<40) y "
                f"el {result.get('pct_alto', 0)}% tiene bienestar alto (T>=60)."
            )

        elif plan.analysis_type == AnalysisType.COMPARISON:
            prueba = result.get("prueba_estadistica", {})
            interp.append("--- Resultados ---")
            interp.append(f"Prueba: {prueba.get('test', 'N/A')}")
            interp.append(f"Estadistico: {prueba.get('estadistico', 'N/A')}")
            interp.append(f"p-value: {prueba.get('p_valor', 'N/A')}")
            interp.append(f"Tamano del efecto: {prueba.get('tamano_efecto', 'N/A')}")
            interp.append(f"Significativo: {'Si' if str(prueba.get('significativo', '')).lower() == 'true' else 'No'}")
            interp.append("")

            # Mostrar grupos
            interp.append("--- Datos por grupo ---")
            estadisticos_grupo = result.get("estadisticos_por_grupo", {})
            if estadisticos_grupo:
                for key, val in estadisticos_grupo.items():
                    interp.append(f"  - {key}: n={val.get('n', 'N/A')}, media={val.get('media', 'N/A')}, SD={val.get('desv_std', 'N/A')}")
            else:
                # Fallback: buscar en el resultado directamente
                for key, val in result.items():
                    if isinstance(val, dict) and "n" in val:
                        interp.append(f"  - {key}: n={val['n']}, media={val['media']}, SD={val['desv_std']}")

        elif plan.analysis_type == AnalysisType.PROFILE:
            perfil = result
            if "_resumen" in perfil:
                res = perfil["_resumen"]
                interp.append("--- Perfil de dimensiones ---")
                interp.append(
                    f"Dimension mas alta: {res['dimension_mas_alta']} "
                    f"(T={perfil[res['dimension_mas_alta']]['media']}). "
                    f"Dimension mas baja: {res['dimension_mas_baja']} "
                    f"(T={perfil[res['dimension_mas_baja']]['media']})."
                )
            else:
                interp.append("Analisis completado.")

        elif plan.analysis_type == AnalysisType.VALIDATION:
            interp.append(result.get("resumen", "Validacion completada."))
        elif "r_cuadrado" in result:
            # Regresion
            interp.append("--- Resultados del modelo ---")
            interp.append(f"Tipo: {result.get('metodo_seleccionado', 'N/A')}")
            interp.append(f"Observaciones: {result.get('n_obs', 'N/A')}")
            interp.append(f"R-cuadrado: {result.get('r_cuadrado', 'N/A')}")
            if result.get('r_cuadrado_ajustado'):
                interp.append(f"R-cuadrado ajustado: {result.get('r_cuadrado_ajustado')}")
            if result.get('f_estadistico'):
                interp.append(f"F-estadistico: {result.get('f_estadistico')}")
                interp.append(f"F p-value: {result.get('f_p_value')}")
            interp.append("")
            interp.append("--- Coeficientes ---")
            coef = result.get("coeficientes", {})
            for var, vals in coef.items():
                if isinstance(vals, dict):
                    interp.append(f"  - {var}: beta={vals.get('beta')}, p={vals.get('p_value')}, SE={vals.get('std_error')}")
                else:
                    interp.append(f"  - {var}: {vals}")
        else:
            interp.append("Analisis completado.")

        return "\n".join(interp)

    def _suggest_visualization(self, plan: AnalysisPlan) -> Optional[str]:
        if plan.analysis_type == AnalysisType.OVERVIEW:
            return "histograma"
        elif plan.analysis_type == AnalysisType.COMPARISON:
            return "barras_con_ic"
        elif plan.analysis_type == AnalysisType.PROFILE:
            return "barras_horizontales"
        return None


def create_agent(
    provider: str = "openai",
    data_path: Optional[str] = None,
    **kwargs,
) -> StatisticalAgent:
    """Factory function para crear el agente."""
    from agent.api_client import create_client
    client = create_client(provider, **kwargs)
    agent = StatisticalAgent(llm_client=client)
    agent.load_data(data_path)
    return agent
