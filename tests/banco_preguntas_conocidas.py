"""
Banco de Preguntas con Respuesta Conocida (18 preguntas)

Para la actividad "Nada sin fuente" - Evaluación de fidelidad del agente.

Cada pregunta tiene:
- Pregunta text
- Respuesta correcta (verificable)
- Fuente (archivo o ubicación)
- Categoría
- Dificultad
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Question:
    """Pregunta con respuesta conocida."""
    
    id: int
    question: str
    correct_answer: str
    source: str
    category: str  # methodology, data, statistics, results
    difficulty: str  # easy, medium, hard
    expected_type: str  # numeric, text, boolean, list
    tolerance: Optional[float] = None  # For numeric answers
    verification_code: Optional[str] = None  # Python code to verify


# ── Banco de preguntas ─────────────────────────────────────────────────

KNOWN_QUESTIONS: List[Question] = [
    # ════════════════════════════════════════════════════════════════════
    # CATEGORÍA 1: METODOLOGÍA (5 preguntas)
    # ════════════════════════════════════════════════════════════════════
    
    Question(
        id=1,
        question="¿Qué instrumento psicométrico se utilizó para medir el bienestar?",
        correct_answer="Escala de Bienestar Psicológico de Ryff (PWI-SF)",
        source="config.py: DIMENSION_MAP, N_ITEMS, NEGATIVE_ITEMS",
        category="methodology",
        difficulty="easy",
        expected_type="text",
    ),
    
    Question(
        id=2,
        question="¿Cuántos ítems tiene la escala utilizada en el estudio?",
        correct_answer="29",
        source="config.py: N_ITEMS = 29",
        category="methodology",
        difficulty="easy",
        expected_type="numeric",
        tolerance=0,
    ),
    
    Question(
        id=3,
        question="¿Cuántas dimensiones mide la escala de Ryff?",
        correct_answer="6",
        source="config.py: N_DIMENSIONS = 6",
        category="methodology",
        difficulty="easy",
        expected_type="numeric",
        tolerance=0,
    ),
    
    Question(
        id=4,
        question="¿Qué tipo de diseño muestral se utilizó para seleccionar los participantes?",
        correct_answer="Muestreo estratificado por seccional, sede, programa y estrato socioeconómico, con factores de expansión calibrados por raking",
        source="data/loader.py, validation/survey_weights.py",
        category="methodology",
        difficulty="medium",
        expected_type="text",
    ),
    
    Question(
        id=5,
        question="¿Cuál es la población total estimada de estudiantes de la universidad?",
        correct_answer="29,950",
        source="config.py: REPORTED_RESULTS['N_population']",
        category="methodology",
        difficulty="easy",
        expected_type="numeric",
        tolerance=0,
    ),
    
    # ════════════════════════════════════════════════════════════════════
    # CATEGORÍA 2: DATOS (5 preguntas)
    # ════════════════════════════════════════════════════════════════════
    
    Question(
        id=6,
        question="¿Cuántas respuestas válidas se obtuvieron después de la limpieza de datos?",
        correct_answer="1,813",
        source="config.py: REPORTED_RESULTS['n_valid']",
        category="data",
        difficulty="easy",
        expected_type="numeric",
        tolerance=0,
    ),
    
    Question(
        id=7,
        question="¿Qué rango de valores tiene la escala Likert utilizada?",
        correct_answer="1 a 6",
        source="config.py: N_RESPONSE_CATEGORIES = 6",
        category="data",
        difficulty="easy",
        expected_type="text",
    ),
    
    Question(
        id=8,
        question="¿Cuántos ítems tienen redacción negativa y requieren inversión?",
        correct_answer="10",
        source="config.py: NEGATIVE_ITEMS = [2, 4, 5, 8, 9, 13, 19, 22, 23, 26]",
        category="data",
        difficulty="easy",
        expected_type="numeric",
        tolerance=0,
    ),
    
    Question(
        id=9,
        question="¿Cómo se calcula el valor invertido para un ítem con redacción negativa?",
        correct_answer="Valor invertido = 7 - valor original",
        source="data/loader.py: invert_negative_items()",
        category="data",
        difficulty="medium",
        expected_type="text",
    ),
    
    Question(
        id=10,
        question="¿Cuáles son las 6 dimensiones del bienestar psicológico que mide la escala?",
        correct_answer="Autoaceptación, Relaciones positivas, Autonomía, Dominio del entorno, Propósito de vida, Crecimiento personal",
        source="config.py: DIMENSION_MAP.keys()",
        category="data",
        difficulty="medium",
        expected_type="list",
    ),
    
    # ════════════════════════════════════════════════════════════════════
    # CATEGORÍA 3: ESTADÍSTICA (5 preguntas)
    # ════════════════════════════════════════════════════════════════════
    
    Question(
        id=11,
        question="¿Qué modelo de Teoría de Respuesta al Ítem (IRT) se utilizó para estandarizar las puntuaciones?",
        correct_answer="Modelo de Respuesta Graduada (Graded Response Model - GRM)",
        source="psychometrics/irt.py",
        category="statistics",
        difficulty="medium",
        expected_type="text",
    ),
    
    Question(
        id=12,
        question="¿Cómo se calculan las puntuaciones T a partir de los estimadores theta de IRT?",
        correct_answer="T = 50 + 10 * theta",
        source="config.py: T_SCORE_MEAN, T_SCORE_SD; psychometrics/irt.py",
        category="statistics",
        difficulty="medium",
        expected_type="text",
    ),
    
    Question(
        id=13,
        question="¿Qué método de selección automática se usa para elegir entre OLS, WLS o logística en regresión?",
        correct_answer="WLS si hay pesos de expansión, OLS si no hay pesos, logística si la variable dependiente es binaria",
        source="statistics/regression.py: auto_select_method()",
        category="statistics",
        difficulty="hard",
        expected_type="text",
    ),
    
    Question(
        id=14,
        question="¿Qué prueba estadística se utiliza para comparar más de 2 grupos cuando los datos no son normales?",
        correct_answer="Kruskal-Wallis",
        source="statistics/inference.py: compare_multiple_groups()",
        category="statistics",
        difficulty="medium",
        expected_type="text",
    ),
    
    Question(
        id=15,
        question="¿Qué prueba se usa para verificar la normalidad de los residuos en un modelo de regresión?",
        correct_answer="Shapiro-Wilk",
        source="statistics/regression.py: _check_ols_assumptions()",
        category="statistics",
        difficulty="medium",
        expected_type="text",
    ),
    
    # ════════════════════════════════════════════════════════════════════
    # CATEGORÍA 4: RESULTADOS (3 preguntas)
    # ════════════════════════════════════════════════════════════════════
    
    Question(
        id=16,
        question="¿Cuál es la media global de bienestar en puntuaciones T reportada por el estudio?",
        correct_answer="50.7",
        source="config.py: REPORTED_RESULTS['T_global_mean']",
        category="results",
        difficulty="easy",
        expected_type="numeric",
        tolerance=0.1,
    ),
    
    Question(
        id=17,
        question="¿Cuál es el alpha de Cronbach reportado para la escala completa?",
        correct_answer="0.909",
        source="config.py: REPORTED_RESULTS['alpha_cronbach']",
        category="results",
        difficulty="easy",
        expected_type="numeric",
        tolerance=0.001,
    ),
    
    Question(
        id=18,
        question="¿Qué proporción de estudiantes se encuentra en riesgo (T < 40)?",
        correct_answer="14.8%",
        source="config.py: REPORTED_RESULTS['risk_proportion']",
        category="results",
        difficulty="medium",
        expected_type="numeric",
        tolerance=0.001,
    ),
]


def get_questions_by_category(category: str) -> List[Question]:
    """Retorna preguntas filtradas por categoría."""
    return [q for q in KNOWN_QUESTIONS if q.category == category]


def get_questions_by_difficulty(difficulty: str) -> List[Question]:
    """Retorna preguntas filtradas por dificultad."""
    return [q for q in KNOWN_QUESTIONS if q.difficulty == difficulty]


def get_question_by_id(question_id: int) -> Optional[Question]:
    """Retorna una pregunta por su ID."""
    for q in KNOWN_QUESTIONS:
        if q.id == question_id:
            return q
    return None


def get_all_categories() -> List[str]:
    """Retorna todas las categorías disponibles."""
    return list(set(q.category for q in KNOWN_QUESTIONS))


def get_all_difficulties() -> List[str]:
    """Retorna todas las dificultades disponibles."""
    return list(set(q.difficulty for q in KNOWN_QUESTIONS))


def verify_answer(question_id: int, user_answer: str) -> Dict[str, Any]:
    """
    Verifica si la respuesta del usuario es correcta.
    
    Returns:
        dict con keys: is_correct, correct_answer, similarity, message
    """
    question = get_question_by_id(question_id)
    if not question:
        return {
            "is_correct": False,
            "correct_answer": None,
            "similarity": 0.0,
            "message": f"Pregunta con ID {question_id} no encontrada",
        }
    
    # Para respuestas numéricas
    if question.expected_type == "numeric":
        try:
            user_val = float(user_answer.replace(",", "").replace("%", ""))
            correct_val = float(question.correct_answer.replace(",", "").replace("%", ""))
            
            if question.tolerance is not None:
                is_correct = abs(user_val - correct_val) <= question.tolerance
                similarity = 1.0 - (abs(user_val - correct_val) / correct_val) if correct_val != 0 else 0
            else:
                is_correct = user_val == correct_val
                similarity = 1.0 if is_correct else 0.0
            
            return {
                "is_correct": is_correct,
                "correct_answer": question.correct_answer,
                "similarity": similarity,
                "message": "Correcta" if is_correct else f"Incorrecta. Respuesta: {question.correct_answer}",
            }
        except ValueError:
            return {
                "is_correct": False,
                "correct_answer": question.correct_answer,
                "similarity": 0.0,
                "message": "No se pudo interpretar como número",
            }
    
    # Para respuestas de texto (búsqueda de palabras clave)
    import re
    
    def clean_words(text):
        """Limpia y normaliza palabras para comparación."""
        text = text.lower()
        # Remover puntuación
        text = re.sub(r'[,\.\;\:\-\(\)\[\]]', ' ', text)
        # Dividir por espacios y filtrar vacías
        return set(w.strip() for w in text.split() if w.strip() and len(w.strip()) > 1)
    
    user_words = clean_words(user_answer)
    correct_words = clean_words(question.correct_answer)
    
    if len(correct_words) == 0:
        similarity = 0.0
    else:
        intersection = user_words.intersection(correct_words)
        similarity = len(intersection) / len(correct_words)
    
    is_correct = similarity >= 0.5  # 50% de palabras clave presentes
    
    return {
        "is_correct": is_correct,
        "correct_answer": question.correct_answer,
        "similarity": similarity,
        "message": "Correcta" if is_correct else f"Incorrecta. Respuesta: {question.correct_answer}",
    }


def get_statistics() -> Dict[str, Any]:
    """Retorna estadísticas del banco de preguntas."""
    return {
        "total_questions": len(KNOWN_QUESTIONS),
        "by_category": {
            cat: len(get_questions_by_category(cat))
            for cat in get_all_categories()
        },
        "by_difficulty": {
            diff: len(get_questions_by_difficulty(diff))
            for diff in get_all_difficulties()
        },
        "question_ids": [q.id for q in KNOWN_QUESTIONS],
    }


if __name__ == "__main__":
    # Imprimir estadísticas
    stats = get_statistics()
    print("=== BANCO DE PREGUNTAS CON RESPUESTA CONOCIDA ===")
    print(f"Total: {stats['total_questions']} preguntas")
    print(f"Por categoría: {stats['by_category']}")
    print(f"Por dificultad: {stats['by_difficulty']}")
    print()
    
    # Imprimir todas las preguntas
    for q in KNOWN_QUESTIONS:
        print(f"--- Pregunta {q.id} ({q.category}, {q.difficulty}) ---")
        print(f"P: {q.question}")
        print(f"R: {q.correct_answer}")
        print(f"Fuente: {q.source}")
        print()
