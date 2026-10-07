"""
Test para verificar que las respuestas del banco de preguntas son correctas.

Verifica que las respuestas sean consistentes con los valores reales del proyecto.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from agent.core.config import (
    DIMENSION_MAP,
    N_ITEMS,
    N_DIMENSIONS,
    NEGATIVE_ITEMS,
    REPORTED_RESULTS,
)
from tests.banco_preguntas_conocidas import (
    KNOWN_QUESTIONS,
    get_question_by_id,
    verify_answer,
    get_statistics,
)


class TestQuestionBank:
    """Tests para el banco de preguntas."""
    
    def test_total_questions(self):
        """Verifica que hay 18 preguntas (rango 15-20)."""
        stats = get_statistics()
        assert 15 <= stats["total_questions"] <= 20
    
    def test_categories_exist(self):
        """Verifica que existen las 4 categorías."""
        stats = get_statistics()
        expected_cats = {"methodology", "data", "statistics", "results"}
        assert set(stats["by_category"].keys()) == expected_cats
    
    def test_difficulties_exist(self):
        """Verifica que existen las 3 dificultades."""
        stats = get_statistics()
        expected_diffs = {"easy", "medium", "hard"}
        assert set(stats["by_difficulty"].keys()) == expected_diffs
    
    def test_all_questions_have_ids(self):
        """Verifica que todas las preguntas tienen ID único."""
        ids = [q.id for q in KNOWN_QUESTIONS]
        assert len(ids) == len(set(ids))
    
    def test_all_questions_have_sources(self):
        """Verifica que todas las preguntas tienen fuente."""
        for q in KNOWN_QUESTIONS:
            assert q.source, f"Pregunta {q.id} no tiene fuente"


class TestAnswersAgainstConfig:
    """Verifica que las respuestas son correctas contra config.py."""
    
    def test_question_2_items_count(self):
        q = get_question_by_id(2)
        assert q.correct_answer == str(N_ITEMS)
    
    def test_question_3_dimensions_count(self):
        q = get_question_by_id(3)
        assert q.correct_answer == str(N_DIMENSIONS)
    
    def test_question_5_population(self):
        q = get_question_by_id(5)
        assert q.correct_answer.replace(",", "") == str(REPORTED_RESULTS["N_population"])
    
    def test_question_6_valid_responses(self):
        q = get_question_by_id(6)
        assert q.correct_answer.replace(",", "") == str(REPORTED_RESULTS["n_valid"])
    
    def test_question_8_negative_items(self):
        q = get_question_by_id(8)
        assert q.correct_answer == str(len(NEGATIVE_ITEMS))
    
    def test_question_16_t_global_mean(self):
        q = get_question_by_id(16)
        assert float(q.correct_answer) == REPORTED_RESULTS["T_global_mean"]
    
    def test_question_17_alpha_cronbach(self):
        q = get_question_by_id(17)
        assert float(q.correct_answer) == REPORTED_RESULTS["alpha_cronbach"]
    
    def test_question_18_risk_proportion(self):
        q = get_question_by_id(18)
        assert float(q.correct_answer.replace("%", "")) / 100 == pytest.approx(REPORTED_RESULTS["risk_proportion"])


class TestVerifyAnswer:
    """Tests para la función verify_answer."""
    
    def test_correct_numeric_answer(self):
        result = verify_answer(2, "29")
        assert result["is_correct"] is True
    
    def test_incorrect_numeric_answer(self):
        result = verify_answer(2, "30")
        assert result["is_correct"] is False
    
    def test_correct_text_answer(self):
        result = verify_answer(14, "Kruskal-Wallis")
        assert result["is_correct"] is True
    
    def test_partially_correct_text_answer(self):
        result = verify_answer(10, "Autoaceptación y Autonomía")
        # Should be partially correct (has some keywords)
        assert result["similarity"] > 0
    
    def test_nonexistent_question(self):
        result = verify_answer(999, "test")
        assert result["is_correct"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
