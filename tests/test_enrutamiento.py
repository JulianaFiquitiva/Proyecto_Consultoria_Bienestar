# -*- coding: utf-8 -*-
"""
Reglas de enrutamiento (sin LLM, sin API, sin dataset).

Cada clase cubre UNA regla de la cascada StatisticalAgent.ruta_de_pregunta():

    1. prefijo explicito "corpus:" / "fuentes:"        -> corpus
    2. pide lo que DICEN los documentos del corpus      -> corpus
    3. pide buscar fuentes en la web (verbo Y fuente)   -> busqueda
    4. nombra un documento concreto del inventario      -> corpus
    5. classify_scope decide entre encuesta y abstinencia

El inventario del punto 4 se lee al arrancar desde
docs/corpus_inventario_20.csv; los terminos nunca estan escritos a mano.

Ademas se comprueba la cascada completa contra el banco DEV
(tests/banco_enrutamiento_dev.json), que trae la ruta esperada.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.core.orchestrator import (           # noqa: E402
    AnalysisType,
    StatisticalAgent,
)
from agent.core.query_planner import (          # noqa: E402
    _TERMINOS_DOCUMENTO,
    _TERMINOS_GENERICOS,
    _TERMINOS_MUESTRA,
    classify_scope,
    pregunta_sobre_documento_indexado,
    referencia_a_estudio,
)

RAIZ = Path(__file__).parent.parent
BANCO_DEV = Path(__file__).parent / "banco_enrutamiento_dev.json"


@pytest.fixture(scope="module")
def agente():
    """Un solo agente para todo el modulo: ruta_de_pregunta no usa datos
    ni LLM, asi que load_data() no hace falta."""
    return StatisticalAgent()


def ruta(ag, pregunta):
    return ag.ruta_de_pregunta(pregunta)


# ── Regla 3: peticion de busqueda = verbo + fuente ───────────────────────

class TestBusquedaRequiereVerboYFuente:
    @pytest.mark.parametrize("pregunta", [
        "busca artículos sobre ansiedad y bienestar",
        "dame referencias sobre bienestar estudiantil",
        "dame literatura sobre bienestar estudiantil",
        "necesito bibliografía sobre deserción universitaria",
        "encuentra papers sobre autonomía y bienestar",
        "recomiéndame lecturas sobre bienestar estudiantil",
        "quiero tesis sobre clima institucional",
        "localiza publicaciones sobre motivación académica",
    ])
    def test_verbo_mas_fuente_es_busqueda(self, agente, pregunta):
        assert ruta(agente, pregunta) == "busqueda", pregunta

    @pytest.mark.parametrize("pregunta", [
        # solo fuente, sin pedir que se busque: nombra una obra o un tema
        "Según el artículo de Daza Corredor, ¿cuánto explica?",
        "las referencias bibliográficas del trabajo son escasas",
        "los papers sobre bienestar son variados",
    ])
    def test_fuente_sin_verbo_no_es_busqueda(self, agente, pregunta):
        assert ruta(agente, pregunta) != "busqueda", pregunta

    @pytest.mark.parametrize("pregunta", [
        "¿cuál es la media global de bienestar?",
        "quiero saber la diferencia entre hombres y mujeres",
        "necesito la muestra de la encuesta",
    ])
    def test_verbo_sin_fuente_no_es_busqueda(self, agente, pregunta):
        assert ruta(agente, pregunta) != "busqueda", pregunta

    @pytest.mark.parametrize("pregunta", [
        # accion contada como ocurrida: es un dato, no una orden
        "¿en qué bases de datos se buscaron los artículos?",
        "¿qué buscó la revisión en los artículos?",
    ])
    def test_verbo_en_pasado_no_es_busqueda(self, agente, pregunta):
        assert ruta(agente, pregunta) != "busqueda", pregunta

    @pytest.mark.parametrize("pregunta", [
        "quiero apoyo sobre el marco teórico del bienestar",
        "necesito apoyo en internet sobre bienestar estudiantil",
    ])
    def test_peticion_explicita_sin_verbo_de_buscar(self, agente, pregunta):
        assert ruta(agente, pregunta) == "busqueda", pregunta

    @pytest.mark.parametrize("pregunta", [
        "¿de dónde salen las fuentes del estudio?",
        "¿qué datos se usó el estudio?",
    ])
    def test_fuentes_del_propio_estudio_no_es_busqueda(self, agente, pregunta):
        assert ruta(agente, pregunta) != "busqueda", pregunta

    def test_latinoamerica_no_bloquea_una_peticion_de_busqueda(self, agente):
        """La region es fuera de dominio para la encuesta, pero una peticion
        de busqueda se evalua ANTES y no puede quedar bloqueada."""
        assert ruta(agente, "recomiéndame lecturas sobre bienestar en "
                            "Latinoamérica") == "busqueda"


# ── Regla 4: documento concreto del corpus ───────────────────────────────

class TestDocumentoIndexado:
    def test_el_inventario_se_ha_cargado(self):
        assert len(_TERMINOS_DOCUMENTO) > 50
        assert _TERMINOS_MUESTRA

    @pytest.mark.parametrize("pregunta", [
        "¿Qué muestra usó el estudio de Herrera-López y colaboradores?",
        "Según el artículo de Daza Corredor, ¿qué se concluye?",
        "En la revisión de Baik, Ryan y Lynch-Wells, ¿qué se incluyó?",
        "¿Qué instrumento usó Orejarena Silva con estudiantes de psicología?",
        "¿Qué concluye el estudio de la Universidad de Guayaquil?",
    ])
    def test_aparece_un_termino_de_obra(self, agente, pregunta):
        assert ruta(agente, pregunta) == "corpus", pregunta

    @pytest.mark.parametrize("pregunta", [
        # la palabra de la ficha aparece en minuscula en la pregunta
        "¿Qué tipo de investigación utilizó el estudio sobre mindfulness?",
        "¿Qué R² dio la regresión en el estudio de la Universidad del "
        "Magdalena?",
    ])
    def test_el_termino_no_necesita_mayuscula_en_la_pregunta(
            self, agente, pregunta):
        assert ruta(agente, pregunta) == "corpus", pregunta

    @pytest.mark.parametrize("texto", [
        "¿Qué instrumento usó el estudio de Quién Sabe y colaboradores?",
        "El estudio de García Martínez reporta n = 200.",
    ])
    def test_referencia_a_estudio_por_nombre_no_indexado(self, texto):
        assert referencia_a_estudio(texto) is True, texto

    @pytest.mark.parametrize("texto", [
        "¿Qué muestra usó el estudio?",
        "el estudio de bienestar en la USTA",
    ])
    def test_estudio_sin_nombre_no_cuenta(self, texto):
        assert referencia_a_estudio(texto) is False, texto

    @pytest.mark.parametrize("pregunta", [
        "¿Qué dimensión de la escala de Ryff tiene el puntaje más bajo?",
        "¿Cómo difiere el bienestar en Bucaramanga?",
        "¿Qué muestra tiene la escala?",
    ])
    def test_palabras_genericas_no_nombran_una_obra(self, agente, pregunta):
        assert pregunta_sobre_documento_indexado(pregunta) is False, pregunta

    def test_los_terminos_genericos_no_estan_en_los_conjuntos(self):
        solapes = (_TERMINOS_GENERICOS & _TERMINOS_DOCUMENTO) | \
            (_TERMINOS_GENERICOS & _TERMINOS_MUESTRA)
        assert not solapes


# ── Regla 4/5: la encuesta tiene prioridad con sus anclas ────────────────

class TestPrioridadDeLaEncuesta:
    @pytest.mark.parametrize("pregunta", [
        "¿Cuál es la media global de bienestar (T-score) de la USTA?",
        "¿Hay diferencia de bienestar entre hombres y mujeres en la encuesta?",
        "¿Cómo difiere el bienestar entre seccionales de la USTA?",
        "¿Qué factores predicen el bienestar en el dataset?",
        "¿Cuál es la media de los datos de la encuesta?",
    ])
    def test_anclas_de_la_encuesta_mandan(self, agente, pregunta):
        assert ruta(agente, pregunta) == "encuesta", pregunta

    def test_un_termino_de_obra_manda_sobre_las_anclas(self, agente):
        """Si ademas se nombra una obra indexada, va al corpus: el punto 1
        de la regla se evalua antes que las anclas."""
        pregunta = ("Según el artículo de Daza Corredor, ¿cómo queda la "
                    "encuesta de la USTA?")
        assert ruta(agente, pregunta) == "corpus", pregunta


# ── Regla 5: fuera de dominio y pregunta ambigua -> abstinencia ──────────

class TestAbstinencia:
    @pytest.mark.parametrize("pregunta", [
        "Existen más estudios realcionados con el bienestar estudiantil?",
        "en donde puedo consultar más sobre temas de bienestar estudaintil?",
        "en otros paises de latinoamerica hay más estudios relacionados?",
        "¿Cuál es la capital de Francia?",
        "¿Qué tiempo hará mañana en Bogotá?",
    ])
    def test_abstiene(self, agente, pregunta):
        assert ruta(agente, pregunta) == "abstencion", pregunta

    @pytest.mark.parametrize("pregunta", [
        "¿Cuál es la capital de Francia?",
        "en otros paises de latinoamerica hay más estudios relacionados?",
    ])
    def test_generate_plan_tambien_abstiene(self, agente, pregunta):
        plan = agente._generate_plan(pregunta)
        assert plan.function_to_call == "abstencion"
        assert plan.analysis_type in (AnalysisType.OUT_OF_DOMAIN,
                                      AnalysisType.AMBIGUOUS)


# ── La cascada de ruta_de_pregunta es la de _generate_plan ───────────────

class TestLaCascadaEsUnica:
    @pytest.mark.parametrize("pregunta,esperada,tipo", [
        ("¿Qué dicen los documentos sobre el estrés académico?",
         "corpus", AnalysisType.CORPUS),
        ("¿Qué muestra usó el estudio de Herrera-López?",
         "corpus", AnalysisType.CORPUS),
        ("dame referencias sobre bienestar estudiantil",
         "busqueda", AnalysisType.SEARCH),
        ("Existen más estudios realcionados con el bienestar estudiantil?",
         "abstencion", AnalysisType.OUT_OF_DOMAIN),
    ])
    def test_generate_plan_sigue_la_misma_ruta(
            self, agente, pregunta, esperada, tipo):
        assert ruta(agente, pregunta) == esperada
        # Estas tres rutas se resuelven ANTES del LLM, asi que el plan no
        # puede depender de que la API este configurada.
        assert agente._generate_plan(pregunta).analysis_type == tipo


# ── Banco DEV: las 24 preguntas con su ruta esperada ─────────────────────

class TestBancoDev:
    @pytest.fixture(scope="module")
    def banco(self):
        return json.loads(BANCO_DEV.read_text(encoding="utf-8"))

    def test_el_banco_tiene_24_preguntas(self, banco):
        assert len(banco) == 24
        assert {f["ruta_esperada"] for f in banco} == {
            "corpus", "busqueda", "encuesta", "abstencion"}

    def test_las_24_enrutan_como_se_espera(self, agente, banco):
        fallos = [
            (f["id"], f["ruta_esperada"], ruta(agente, f["pregunta"]))
            for f in banco
            if ruta(agente, f["pregunta"]) != f["ruta_esperada"]
        ]
        assert not fallos, fallos

    def test_generate_plan_coincide_en_las_que_no_llaman_al_llm(
            self, agente, banco):
        """corpus / busqueda / abstinencia se deciden sin LLM: el plan tiene
        que reflejar la misma ruta. Las de encuesta quedan fuera porque ahi
        si entraria el LLM."""
        sin_llm = {"corpus", "busqueda", "abstencion"}
        esperado = {
            "corpus": AnalysisType.CORPUS,
            "busqueda": AnalysisType.SEARCH,
            "abstencion": None,
        }
        fallos = []
        for f in banco:
            if f["ruta_esperada"] not in sin_llm:
                continue
            plan = agente._generate_plan(f["pregunta"])
            want = esperado[f["ruta_esperada"]]
            if want is None:
                ok = plan.function_to_call == "abstencion"
            else:
                ok = plan.analysis_type == want
            if not ok:
                fallos.append((f["id"], f["ruta_esperada"],
                               plan.analysis_type, plan.function_to_call))
        assert not fallos, fallos


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
