"""
Query Planner - Interpreta la intencion del usuario y planifica el analisis.

Responsabilidades:
1. Interpretar preguntas en lenguaje natural
2. Identificar variables objetivo y de agrupacion
3. Comprobar que las variables existan en el dataset
4. Comprobar tamanos de muestra
5. Decidir si usar pesos
6. Ejecutar el analisis
7. Calcular incertidumbre
8. Interpretar y advertir limitaciones
"""
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.core.config import (
    DIMENSION_MAP,
    DIMENSION_ORDER,
    ITEM_LABELS,
    NEGATIVE_ITEMS,
    N_ITEMS,
)


@dataclass
class QueryPlan:
    analysis_type: str
    description: str
    value_variable: Optional[str]
    group_variable: Optional[str]
    extra_variables: List[str]
    weights_variable: Optional[str]
    filters: Dict[str, Any]
    method: str
    confidence_level: float
    warnings: List[str]
    variables_found: Dict[str, bool]


@dataclass
class QueryResult:
    plan: QueryResult
    data: Any
    interpretation: str
    warnings: List[str]
    traceability: Dict[str, Any]
    success: bool
    error: Optional[str] = None


# Mapeo de palabras clave a variables
KEYWORD_TO_VARIABLE = {
    "genero": "Genero",
    "sexo": "Genero",
    "hombre": "Genero",
    "mujer": "Genero",
    "masculino": "Genero",
    "femenino": "Genero",
    "edad": "Edad",
    "mayor": "Edad",
    "joven": "Edad",
    "estrato": "Estrato socioeconomico",
    "socioeconomico": "Estrato socioeconomico",
    "nivel": "Nivel de formacion",
    "pregrado": "Nivel de formacion",
    "posgrado": "Nivel de formacion",
    "especializacion": "Nivel de formacion",
    "maestria": "Nivel de formacion",
    "modalidad": "Modalidad",
    "presencial": "Modalidad",
    "virtual": "Modalidad",
    "hibrido": "Modalidad",
    "semestre": "Semestre",
    "seccional": "seccional",
    "sede": "seccional",
    "bogota": "seccional",
    "bucaramanga": "seccional",
    "tunja": "seccional",
    "villavicencio": "seccional",
    "programa": "Programa de formacion",
    "carrera": "Programa de formacion",
    "nbc": "Nucleo basico de conocimiento",
    "nucleo": "Nucleo basico de conocimiento",
    "bienestar": "T_Global",
    "felicidad": "T_Global",
    "indice": "T_Global",
    "global": "T_Global",
    "autoaceptacion": "T_Autoaceptacion",
    "crecimiento": "T_Crecimiento personal",
    "proposito": "T_Proposito de vida",
    "dominio": "T_Dominio del entorno",
    "relaciones": "T_Relaciones positivas",
    "autonomia": "T_Autonomia",
    "riesgo": "T_Global",
    "alto": "T_Global",
    "bajo": "T_Global",
}

KEYWORDS_ANALYSIS_TYPE = {
    "resumen": "overview",
    "general": "overview",
    "descripci": "descriptivo",
    "describe": "descriptivo",
    "cuantos": "descriptivo",
    "cuantas": "descriptivo",
    "compara": "comparacion",
    "diferencia": "comparacion",
    "difieren": "comparacion",
    "distinto": "comparacion",
    "correlacion": "correlacion",
    "relacion": "correlacion",
    "asocia": "correlacion",
    "predice": "regresion",
    "regresion": "regresion",
    "factor": "regresion",
    "valida": "validacion",
    "reproduce": "validacion",
    "compara con": "validacion",
    "perfil": "perfil",
    "dimension": "perfil",
    "por que": "perfil",
}


# ── Clasificacion de ambito: dominio vs fuera de dominio ────────────────
#
# Categoria "fuera_de_dominio": la pregunta NO pide un analisis sobre las
# variables del dataset. El agente se abstiene en vez de inventar.
#
# Cada regla lleva el motivo que se le muestra al usuario.

OUT_OF_DOMAIN_RULES: List[Tuple[str, str]] = [
    # 1. Pregunta por otros estudios / publicaciones existentes
    (
        r"(?:mas|m[aá]s|otros?|existentes?|nuevos?)\s+"
        r"(?:estudios|papers?|art[ií]culos|investigaciones|publicaciones)",
        "Pregunta por otros estudios o publicaciones existentes. El dataset "
        "solo contiene respuestas de una encuesta, no un registro de estudios.",
    ),
    (
        r"(?:estudios|papers?|art[ií]culos|investigaciones)\s+relacionados?",
        "Pregunta por estudios relacionados. El dataset no indexa literatura "
        "cientifica; solo tiene respuestas de la encuesta de bienestar.",
    ),
    (
        r"(?:existen?|hay)\s+(?:mas|m[aá]s|otros?)\s+estudios",
        "Pregunta cuantos otros estudios existen. Eso no esta medido en el "
        "dataset, que solo contiene respuestas de la encuesta.",
    ),
    # 2. Pregunta donde consultar / donde encontrar informacion externa
    (
        r"d[oó]nde\s+(?:puedo|se\s+puede|podemos|pueden)?\s*"
        r"(?:consultar|encontrar|averiguar)",
        "Pregunta donde consultar informacion externa. El dataset no contiene "
        "bibliografia ni directorios de fuentes.",
    ),
    (
        r"(?:puedo|podemos)\s+consultar\s+(?:mas|m[aá]s)",
        "Pregunta donde consultar mas informacion. El dataset no contiene "
        "bibliografia ni directorios de fuentes.",
    ),
    (
        r"consultar\s+(?:mas|m[aá]s)\s+sobre\s+temas?",
        "Pregunta donde consultar mas temas. El dataset no cataloga temas "
        "ni documentos externos.",
    ),
    # 3. Pregunta por otros paises / regiones
    (
        r"otros?\s+pa[ií]ses",
        "Pregunta por otros paises. El dataset solo cubre sedes de "
        "Universidad Santo Tomas en Colombia.",
    ),
    (
        r"(?:en|de|los)\s+otros?\s+pa[ií]ses",
        "Pregunta por otros paises. El dataset solo cubre sedes de "
        "Universidad Santo Tomas en Colombia.",
    ),
    (
        r"latinoam[eé]rica",
        "Pregunta por la region de Latinoamerica. El dataset solo cubre "
        "sedes de Universidad Santo Tomas en Colombia.",
    ),
    # 4. Variables explicitamente ausentes del dataset
    (
        r"\b(?:ingreso|notas?|calificaci[oó]n(?:es)?|rendimiento\s+acad[eé]mico|"
        r"satisfacci[oó]n\s+universitaria|clima|pol[ií]tica|deporte|"
        r"m[uú]sica|comida|noviazgo|religi[oó]n)\b",
        "Pregunta por una variable que no existe en el instrumento de medicion.",
    ),
]

# Senales de que la pregunta SI toca variables del dataset
DOMAIN_TERMS_RE = re.compile(
    r"\b(?:t[_\s-]?score|t_global|tscore|puntuaci[oó]n|puntaje|"
    r"dimension(?:es)?|escala|ryff|i[té]tems?|genero|sexo|hombre|mujer|"
    r"edad|estrato|sede|seccional|modalidad|programa|carrera|semestre|"
    r"n[uú]cleo|nbc|bienestar|felicidad|riesgo|autoaceptaci[oó]n|autonom[ií]a|"
    r"crecimiento|prop[oó]sito|dominio|relaciones\s+positivas|"
    r"alpha|cronbach|pregrado|posgrado|especializaci[oó]n|maestr[ií]a|"
    r"estudiantes?|muestra|encuesta|respuestas|participantes|registros|"
    r"dataset|correlaci[oó]n|regresi[oó]n|coeficiente|media|mediana|"
    r"desviaci[oó]n|promedio|significativo|p[\s-]?valor|universidad\s+santo\s+tom[aá]s)\b",
    re.IGNORECASE,
)

# Senales de que se pide un ANALISIS (no solo mencionar una palabra)
ANALYSIS_INTENT_RE = re.compile(
    r"\b(?:cu[aá]ntos?|cu[aá]ntas?|cu[aá]l|cu[aá]les|diferencia|"
    r"compara(?:r)?|difieren|correlaci[oó]n|relaci[oó]n|predice|predicen|"
    r"regresi[oó]n|resumen|general|perfil|valida|reproduce|"
    r"media|promedio|distribuci[oó]n|clasificaci[oó]n|riesgo|"
    r"significativo|analiza|muestra|estado|nivel)\b",
    re.IGNORECASE,
)

# Palabras que por si solas NO bastan para considerar el tema "en dominio"
# (aparecen en casi cualquier pregunta y no anclan una variable)
WEAK_ONLY_RE = re.compile(
    r"^(?:que|cual|cuales|como|donde|cuando|quien|por\s+que|"
    r"los|las|del|de|la|el|y|o|en|con|para|por|sobre|estos|estas|"
    r"mas|muchos|existe|existen|hay|puedo|quiero|necesito)\b",
    re.IGNORECASE,
)


def classify_scope(question: str) -> Tuple[str, str]:
    """
    Clasifica el ambito de una pregunta.

    Returns:
        (categoria, motivo) donde categoria es una de:
        - "fuera_de_dominio": el dataset no puede responderla -> ABSTENCION
        - "en_dominio": toca variables del dataset -> se puede analizar
        - "ambigua": no se reconoce con claridad -> ABSTENCION (reformular)

    Orden de decision:
    1. Si coincide con una regla de exclusion -> fuera_de_dominio
    2. Si menciona variables del dataset Y pide un analisis -> en_dominio
    3. Si solo menciona variables pero no pide analisis -> en_dominio
    4. En otro caso -> ambigua
    """
    q = question.lower().strip()

    # 1) Reglas de exclusion (tienen prioridad sobre cualquier palabra de dominio)
    for pattern, motivo in OUT_OF_DOMAIN_RULES:
        if re.search(pattern, q, re.IGNORECASE):
            return "fuera_de_dominio", motivo

    # 2) Ancla de variable del dataset
    has_domain = bool(DOMAIN_TERMS_RE.search(question))
    # 3) Intencion de analisis
    has_intent = bool(ANALYSIS_INTENT_RE.search(question))

    if has_domain:
        return "en_dominio", ""

    if has_intent:
        # Pide analisis pero no se anclo ninguna variable -> ambigua
        return (
            "ambigua",
            "La pregunta pide un analisis pero no se reconoce que variable "
            "del dataset usar.",
        )

    return (
        "ambigua",
        "No se reconoce con claridad de que trata la pregunta ni que "
        "variable del dataset usar.",
    )


class QueryPlanner:
    """
    Planifica y ejecuta analisis estadisticos basado en preguntas
    en lenguaje natural.
    """
    
    def __init__(self, data: pd.DataFrame, irt_data: Optional[pd.DataFrame] = None):
        self.data = data
        self.irt_data = irt_data
        self._available_columns = set(data.columns)
        if irt_data is not None:
            self._available_columns.update(irt_data.columns)
    
    def plan_query(self, question: str) -> QueryPlan:
        """
        Genera un plan de analisis a partir de una pregunta.
        """
        question_lower = question.lower()
        warnings = []
        variables_found = {}

        # 0. Clasificar el AMBITO antes de elegir analisis.
        #    Fuera de dominio o ambigua -> ABSTENCION (no se ejecuta nada).
        scope, motivo = classify_scope(question)
        if scope != "en_dominio":
            return QueryPlan(
                analysis_type=scope,          # "fuera_de_dominio" | "ambigua"
                description=motivo,
                value_variable=None,
                group_variable=None,
                extra_variables=[],
                weights_variable=None,
                filters={},
                method="abstencion",
                confidence_level=0.0,
                warnings=[motivo],
                variables_found={},
            )

        # 1. Detectar tipo de analisis (None = no reconocido todavia)
        analysis_type: Optional[str] = None
        for keyword, atype in KEYWORDS_ANALYSIS_TYPE.items():
            if keyword in question_lower:
                analysis_type = atype
                break

        # 1b. En dominio pero sin analisis reconocible -> AMBIGUA
        #     (la rama por defecto YA NO es descriptive_overview)
        if analysis_type is None:
            return QueryPlan(
                analysis_type="ambigua",
                description=(
                    "La pregunta toca variables del dataset pero no se "
                    "reconoce que analisis pedir. Reformular la pregunta."
                ),
                value_variable=None,
                group_variable=None,
                extra_variables=[],
                weights_variable=None,
                filters={},
                method="abstencion",
                confidence_level=0.0,
                warnings=["Analisis no reconocido"],
                variables_found={},
            )
        
        # 2. Detectar variable objetivo
        value_variable = None
        for keyword, var in KEYWORD_TO_VARIABLE.items():
            if keyword in question_lower:
                if var in self._available_columns:
                    value_variable = var
                    variables_found[var] = True
                    break
        
        # Si no se encontro variable, usar T_Global por defecto
        if value_variable is None:
            if "T_Global" in self._available_columns:
                value_variable = "T_Global"
                variables_found["T_Global"] = True
        
        # 3. Detectar variable de agrupacion
        group_variable = None
        for keyword, var in KEYWORD_TO_VARIABLE.items():
            if keyword in question_lower and var != value_variable:
                if var in self._available_columns:
                    group_variable = var
                    variables_found[var] = True
                    break
        
        # 4. Verificar variables
        if value_variable and value_variable not in self._available_columns:
            warnings.append(f"Variable '{value_variable}' no encontrada en el dataset")
            variables_found[value_variable] = False
        
        if group_variable and group_variable not in self._available_columns:
            warnings.append(f"Variable '{group_variable}' no encontrada en el dataset")
            variables_found[group_variable] = False
        
        # 5. Decidir si usar pesos
        weights_variable = None
        if "factor_expansion" in self._available_columns:
            weights_variable = "factor_expansion"
        
        # 6. Detectar filtros
        filters = {}
        for keyword, var in KEYWORD_TO_VARIABLE.items():
            if keyword in question_lower and var in self._available_columns:
                # Buscar valores especificos
                if var == "seccional":
                    for sede in ["bogota", "bucaramanga", "tunja", "villavicencio"]:
                        if sede in question_lower:
                            filters[var] = sede
                elif var == "Nivel de formacion":
                    for nivel in ["pregrado", "posgrado", "especializacion", "maestria"]:
                        if nivel in question_lower:
                            filters[var] = nivel
        
        # 7. Seleccionar metodo
        if analysis_type == "comparacion":
            if group_variable:
                n_groups = self.data[group_variable].nunique() if group_variable in self.data.columns else 0
                method = "multi_group" if n_groups > 2 else "two_group"
            else:
                method = "two_group"
        elif analysis_type == "correlacion":
            method = "correlation"
        elif analysis_type == "regresion":
            method = "regression"
        else:
            method = "descriptive"
        
        # 8. Verificar tamanos de muestra
        if group_variable and group_variable in self.data.columns:
            group_sizes = self.data[group_variable].value_counts()
            for g, n in group_sizes.items():
                if n < 30:
                    warnings.append(f"Grupo '{g}' tiene solo {n} casos, resultados poco confiables para inferencia")
        
        return QueryPlan(
            analysis_type=analysis_type,
            description=question,
            value_variable=value_variable,
            group_variable=group_variable,
            extra_variables=[],
            weights_variable=weights_variable,
            filters=filters,
            method=method,
            confidence_level=0.95,
            warnings=warnings,
            variables_found=variables_found,
        )
    
    def execute_plan(self, plan: QueryPlan) -> Dict[str, Any]:
        """
        Ejecuta el plan de analisis y retorna resultados.
        """
        from agent.statistics.statistical_engine import StatisticalEngine
        engine = StatisticalEngine()
        
        df = self.data.copy()
        if self.irt_data is not None and plan.value_variable and "T_" in plan.value_variable:
            df = self.irt_data
        
        # Aplicar filtros
        for var, val in plan.filters.items():
            if var in df.columns:
                if isinstance(val, str):
                    mask = df[var].str.contains(val, case=False, na=False)
                    df = df[mask]
        
        traceability = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": plan.analysis_type,
            "value_variable": plan.value_variable,
            "group_variable": plan.group_variable,
            "weights_variable": plan.weights_variable,
            "filters": plan.filters,
            "n_records": len(df),
            "method": plan.method,
        }
        
        result = {"traceability": traceability, "warnings": plan.warnings}
        
        try:
            if plan.analysis_type == "overview" or plan.analysis_type == "descriptivo":
                if plan.group_variable:
                    result["descriptive"] = engine.descriptive_summary(
                        df, plan.value_variable, plan.weights_variable, plan.group_variable
                    )
                else:
                    result["descriptive"] = engine.descriptive_summary(
                        df, plan.value_variable, plan.weights_variable
                    )
            
            elif plan.analysis_type == "comparacion":
                if plan.value_variable and plan.group_variable:
                    analysis_plan, test_result = engine.run_comparison(
                        df, plan.value_variable, plan.group_variable, plan.weights_variable
                    )
                    result["analysis_plan"] = {
                        "method": analysis_plan.method,
                        "test_name": analysis_plan.test_name,
                        "reason": analysis_plan.reason,
                        "alternative": analysis_plan.alternative,
                    }
                    result["test_result"] = {
                        "test_name": test_result.test_name,
                        "statistic": test_result.statistic,
                        "p_value": test_result.p_value,
                        "effect_size": test_result.effect_size,
                        "effect_size_name": test_result.effect_size_name,
                        "significant": test_result.significant,
                        "interpretation": test_result.interpretation,
                        "ci_lower": test_result.ci_lower,
                        "ci_upper": test_result.ci_upper,
                    } if test_result else None
                    result["group_comparison"] = engine.descriptive_summary(
                        df, plan.value_variable, plan.weights_variable, plan.group_variable
                    )
            
            elif plan.analysis_type == "correlacion":
                if plan.value_variable and plan.group_variable:
                    from scipy.stats import pearsonr, spearmanr
                    mask = df[[plan.value_variable, plan.group_variable]].dropna().index
                    x = df.loc[mask, plan.value_variable]
                    y = df.loc[mask, plan.group_variable]
                    r_pearson, p_pearson = pearsonr(x, y)
                    r_spearman, p_spearman = spearmanr(x, y)
                    result["correlation"] = {
                        "pearson": {"r": round(r_pearson, 4), "p": round(p_pearson, 4)},
                        "spearman": {"rho": round(r_spearman, 4), "p": round(p_spearman, 4)},
                        "n": len(mask),
                    }
            
            elif plan.analysis_type == "regresion":
                x_cols = [c for c in [plan.group_variable] if c and c in df.columns]
                if x_cols and plan.value_variable:
                    analysis_plan, reg_result = engine.run_regression(
                        df, plan.value_variable, x_cols, plan.weights_variable
                    )
                    result["regression"] = {
                        "model_type": reg_result.model_type,
                        "r_squared": reg_result.r_squared,
                        "adj_r_squared": reg_result.adj_r_squared,
                        "coefficients": reg_result.coefficients.to_dict(),
                        "interpretation": reg_result.interpretation,
                    }
            
            elif plan.analysis_type == "perfil":
                from agent.psychometrics.scale import PsychometricScale
                scale = PsychometricScale()
                from agent.data.loader import DataLoader
                loader = DataLoader()
                item_cols = loader.get_item_columns(df)
                report = scale.full_analysis(df, item_cols, plan.weights_variable)
                result["perfil"] = {
                    "alpha_global": report.alpha_global,
                    "alpha_dimensions": report.alpha_dimensions,
                    "dimension_means": report.dimension_means,
                    "warnings": report.warnings,
                }
            
            elif plan.analysis_type == "validacion":
                from agent.validation.result_validator import ResultValidator
                validator = ResultValidator()
                rows = validator.validate_all(df_prepared=df, df_irt=self.irt_data)
                result["validacion"] = {
                    "tabla": validator.generate_comparison_table(rows).to_dict(orient="records"),
                    "resumen": validator.summary(rows),
                }
        
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
        
        result["success"] = "error" not in result
        return result
    
    def interpret_result(self, question: str, result: Dict) -> str:
        """
        Genera una interpretacion en lenguaje natural del resultado.
        """
        parts = []
        
        if result.get("error"):
            return f"Error al ejecutar el analisis: {result['error']}"
        
        if "descriptive" in result:
            desc = result["descriptive"]
            if "por_grupo" in desc:
                parts.append("Resumen por grupo:")
                for g, stats in desc["por_grupo"].items():
                    parts.append(f"- {g}: media={stats['media']}, DE={stats['desv_std']}, n={stats['n']}")
            else:
                parts.append(f"Resumen: n={desc['n']}, media={desc['media']}, DE={desc['desv_std']}, mediana={desc['mediana']}")
        
        elif "test_result" in result and result["test_result"]:
            tr = result["test_result"]
            sig = "significativa" if tr["significant"] else "no significativa"
            parts.append(f"La diferencia es {sig} (p={tr['p_value']:.4f}).")
            parts.append(f"Tamano del efecto: {tr['effect_size_name']}={tr['effect_size']:.3f}")
            if tr.get("ci_lower") is not None:
                parts.append(f"IC 95% de la diferencia: [{tr['ci_lower']:.2f}, {tr['ci_upper']:.2f}]")
        
        elif "correlation" in result:
            corr = result["correlation"]
            r = corr["pearson"]["r"]
            p = corr["pearson"]["p"]
            strength = "fuerte" if abs(r) > 0.7 else "moderada" if abs(r) > 0.4 else "debil"
            sig = "estadisticamente significativa" if p < 0.05 else "no estadisticamente significativa"
            parts.append(f"Correlacion {strength}: r={r:.3f}, p={p:.4f} ({sig})")
        
        elif "perfil" in result:
            perfil = result["perfil"]
            parts.append(f"Alpha de Cronbach global: {perfil['alpha_global']:.3f}")
            parts.append("Medias por dimension:")
            for dim, mean in perfil["dimension_means"].items():
                parts.append(f"- {dim}: {mean:.2f}")
        
        elif "validacion" in result:
            parts.append(result["validacion"]["resumen"])
        
        if result.get("warnings"):
            parts.append("\nAdvertencias:")
            for w in result["warnings"]:
                parts.append(f"- {w}")
        
        return "\n".join(parts) if parts else "Analisis completado. Ver resultados detallados."
