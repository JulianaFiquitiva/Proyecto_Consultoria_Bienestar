"""
Pruebas completas del pipeline del Agente de Bienestar Estudiantil.

Verifica:
- Carga de datasets
- Numero de registros
- Numero de items
- Inversion de items
- Calculo de dimensiones
- Factores de expansion
- T-scores
- Resultados globales
- Resultados por grupos
- Validacion completa
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.core.config import (
    DIMENSION_MAP,
    DIMENSION_ORDER,
    NEGATIVE_ITEMS,
    N_ITEMS,
    REPORTED_RESULTS,
)
from agent.data.loader import DataLoader
from agent.data.profiler import DataProfiler
from agent.data.validator import DataValidator
from agent.psychometrics.scale import PsychometricScale
from agent.psychometrics.irt import IRTAnalyzer
from agent.statistics.descriptive import DescriptiveStats
from agent.statistics.inference import InferenceEngine
from agent.statistics.regression import RegressionEngine
from agent.statistics.statistical_engine import StatisticalEngine
from agent.validation.survey_weights import SurveyWeightValidator
from agent.validation.result_validator import ResultValidator
from agent.validation.psychometric_validator import PsychometricValidator
from agent.validation.irt_validator import IRTValidator
from agent.core.orchestrator import StatisticalAgent


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Dataset de ejemplo para pruebas (usa nombres reales de columnas)."""
    np.random.seed(42)
    n = 200
    data = {
        "seccional": np.random.choice(["Bogota", "Tunja", "Bucaramanga"], n),
        "Programa de formación": np.random.choice(["Ing. Sistemas", "Derecho"], n),
        "Núcleo básico de conocimiento": np.random.choice(["Ingenieria", "Derecho"], n),
    }
    for i in range(1, 30):
        data[f"{i}. Item de la escala"] = np.random.randint(1, 7, n)
    data["Métrica Global"] = sum(data[f"{i}. Item de la escala"] for i in range(1, 30))
    data["factor_expansion"] = np.random.uniform(10, 30, n)
    data["T_Global"] = np.random.normal(50, 10, n)
    return pd.DataFrame(data)


@pytest.fixture
def item_cols():
    """Lista de columnas de items."""
    return [f"{i}. Item de la escala" for i in range(1, 30)]


# ── Pruebas de configuracion ─────────────────────────────────────────────

class TestConfig:
    def test_n_items(self):
        assert N_ITEMS == 29

    def test_negative_items_count(self):
        assert len(NEGATIVE_ITEMS) == 10

    def test_dimensions_count(self):
        assert len(DIMENSION_MAP) == 6

    def test_all_items_in_dimensions(self):
        all_items = set()
        for items in DIMENSION_MAP.values():
            all_items.update(items)
        assert all_items == set(range(1, 30))

    def test_dimension_order(self):
        assert len(DIMENSION_ORDER) == 6
        assert DIMENSION_ORDER[0] == "Autoaceptación"

    def test_reported_results_keys(self):
        required = ["n_valid", "N_population", "T_global_mean", "alpha_cronbach"]
        for key in required:
            assert key in REPORTED_RESULTS


# ── Pruebas de carga de datos ────────────────────────────────────────────

class TestDataLoader:
    def test_get_item_columns(self, sample_data, item_cols):
        loader = DataLoader()
        found = loader.get_item_columns(sample_data)
        assert len(found) == 29

    def test_invert_negative_items(self, sample_data, item_cols):
        loader = DataLoader()
        original = sample_data.copy()
        inverted = loader.invert_negative_items(sample_data, item_cols)

        for item_num in NEGATIVE_ITEMS:
            col = item_cols[item_num - 1]
            assert inverted[col].mean() == pytest.approx(
                7 - original[col].mean(), rel=1e-10
            )

    def test_compute_dimensions(self, sample_data, item_cols):
        loader = DataLoader()
        result = loader.compute_dimensions(sample_data, item_cols)

        for dim in DIMENSION_ORDER:
            assert f"Dim_{dim}" in result.columns

        assert "Métrica Global" in result.columns

        for dim in DIMENSION_ORDER:
            col = f"Dim_{dim}"
            assert result[col].mean() >= 1
            assert result[col].mean() <= 6

    def test_get_weight_column(self, sample_data):
        loader = DataLoader()
        col = loader.get_weight_column(sample_data)
        assert col == "factor_expansion"
        loader = DataLoader()
        original = sample_data.copy()
        inverted = loader.invert_negative_items(sample_data, item_cols)

        for item_num in NEGATIVE_ITEMS:
            col = item_cols[item_num - 1]
            assert inverted[col].mean() == pytest.approx(
                7 - original[col].mean(), rel=1e-10
            )

    def test_compute_dimensions(self, sample_data, item_cols):
        loader = DataLoader()
        result = loader.compute_dimensions(sample_data, item_cols)

        for dim in DIMENSION_ORDER:
            assert f"Dim_{dim}" in result.columns

        assert "Métrica Global" in result.columns

        for dim in DIMENSION_ORDER:
            col = f"Dim_{dim}"
            assert result[col].mean() >= 1
            assert result[col].mean() <= 6

    def test_get_weight_column(self, sample_data):
        loader = DataLoader()
        col = loader.get_weight_column(sample_data)
        assert col == "factor_expansion"


# ── Pruebas de profiling ─────────────────────────────────────────────────

class TestDataProfiler:
    def test_basic_profile(self, sample_data):
        profiler = DataProfiler()
        profile = profiler.profile(sample_data)
        assert profile.n_rows == 200
        assert profile.n_columns > 0

    def test_summary(self, sample_data):
        profiler = DataProfiler()
        profile = profiler.profile(sample_data)
        summary = profiler.summary(profile)
        assert "PERFIL DEL DATASET" in summary

    def test_sensitive_detection(self):
        profiler = DataProfiler()
        df = pd.DataFrame({
            "nombre": ["Ana", "Luis"],
            "correo": ["a@b.com", "c@d.com"],
            "edad": [20, 22],
        })
        profile = profiler.profile(df)
        assert "nombre" in profile.sensitive_columns
        assert "correo" in profile.sensitive_columns


# ── Pruebas de validacion de datos ────────────────────────────────────────

class TestDataValidator:
    def test_validate_prepared_structure(self, sample_data):
        validator = DataValidator()
        report = validator.validate_prepared(sample_data)
        assert report.n_checks > 0

    def test_validate_irt(self, sample_data):
        validator = DataValidator()
        report = validator.validate_irt(sample_data)
        assert report.n_checks > 0


# ── Pruebas de estadisticos descriptivos ─────────────────────────────────

class TestDescriptiveStats:
    def test_weighted_mean(self):
        x = np.array([1, 2, 3, 4, 5])
        w = np.array([1, 1, 1, 1, 1])
        assert DescriptiveStats.weighted_mean(x, w) == 3.0

    def test_weighted_mean_unequal_weights(self):
        x = np.array([1, 2, 3])
        w = np.array([1, 2, 1])
        expected = (1 * 1 + 2 * 2 + 3 * 1) / 4
        assert DescriptiveStats.weighted_mean(x, w) == pytest.approx(expected)

    def test_weighted_variance(self):
        x = np.array([1, 2, 3, 4, 5])
        w = np.array([1, 1, 1, 1, 1])
        var = DescriptiveStats.weighted_variance(x, w)
        assert var == pytest.approx(np.var(x, ddof=1), rel=1e-10)

    def test_weighted_std(self):
        x = np.array([1, 2, 3, 4, 5])
        w = np.array([1, 1, 1, 1, 1])
        std = DescriptiveStats.weighted_std(x, w)
        assert std == pytest.approx(np.std(x, ddof=1), rel=1e-10)

    def test_weighted_median(self):
        x = np.array([1, 2, 3, 4, 5])
        w = np.array([1, 1, 1, 1, 1])
        assert DescriptiveStats.weighted_median(x, w) == 3.0

    def test_weighted_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        w = np.array([1, 1, 1, 1, 1])
        corr = DescriptiveStats.weighted_correlation(x, y, w)
        assert corr == pytest.approx(1.0, rel=1e-10)

    def test_weighted_cronbach_alpha(self):
        np.random.seed(42)
        data = pd.DataFrame(np.random.randint(1, 7, (100, 5)))
        w = np.ones(100)
        alpha = DescriptiveStats.weighted_cronbach_alpha(data, w)
        assert 0 <= alpha <= 1


# ── Pruebas de psicometria ──────────────────────────────────────────────

class TestPsychometricScale:
    def test_validate_item_inversion(self, sample_data, item_cols):
        scale = PsychometricScale()
        inverted = scale._invert_items(sample_data, item_cols) if hasattr(
            scale, '_invert_items'
        ) else sample_data.copy()

        for i in NEGATIVE_ITEMS:
            col = item_cols[i - 1]
            inverted[col] = 7 - inverted[col]

        results = scale.validate_item_inversion(sample_data, inverted, item_cols)

        for item_num in NEGATIVE_ITEMS:
            assert results[item_num]["difference"] < 0.01


# ── Pruebas de IRT ───────────────────────────────────────────────────────

class TestIRTAnalyzer:
    def test_compute_t_scores(self):
        analyzer = IRTAnalyzer()
        theta = np.array([-2, -1, 0, 1, 2])
        t_scores = analyzer.compute_t_scores(theta, method="prior")
        expected = np.array([30, 40, 50, 60, 70])
        np.testing.assert_array_almost_equal(t_scores, expected)

    def test_classify_wellbeing(self):
        analyzer = IRTAnalyzer()
        t_scores = np.array([35, 42, 50, 58, 65])
        classification = analyzer.classify_wellbeing(t_scores)
        assert classification["n_total"] == 5
        assert classification["counts"]["Muy bajo"] == 1
        assert classification["counts"]["Bajo"] == 1
        assert classification["counts"]["Medio"] == 1
        assert classification["counts"]["Alto"] == 1
        assert classification["counts"]["Muy alto"] == 1


# ── Pruebas de regresion ────────────────────────────────────────────────

class TestRegressionEngine:
    def test_auto_select_method_ols(self):
        engine = RegressionEngine()
        df = pd.DataFrame({
            "y": np.random.normal(0, 1, 100),
            "x1": np.random.normal(0, 1, 100),
            "x2": np.random.normal(0, 1, 100),
        })
        method = engine.auto_select_method(df, "y", ["x1", "x2"])
        assert method == "ols"

    def test_auto_select_method_logistic(self):
        engine = RegressionEngine()
        df = pd.DataFrame({
            "y": np.random.choice([0, 1], 100),
            "x1": np.random.normal(0, 1, 100),
        })
        method = engine.auto_select_method(df, "y", ["x1"])
        assert method == "logistic"

    def test_ols_basic(self):
        engine = RegressionEngine()
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = 2 * x + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "x": x})

        result = engine.ols(df, "y", ["x"])
        assert result.r_squared > 0.5
        assert result.f_p_value < 0.05
        assert abs(result.coefficients.loc["x", "coef"] - 2) < 0.3


# ── Pruebas de validacion de resultados ──────────────────────────────────

class TestResultValidator:
    def test_comparison_table(self):
        from agent.validation.result_validator import ComparisonRow
        validator = ResultValidator()
        rows = [
            ComparisonRow("n_valid", 1813, 1813, 0, 0, "VALIDADO", "ok"),
            ComparisonRow("T_global", 50.7, 50.8, 0.1, 0.2, "VALIDADO", "ok"),
        ]
        table = validator.generate_comparison_table(rows)
        assert len(table) == 2
        assert "Resultado" in table.columns

    def test_summary(self):
        from agent.validation.result_validator import ComparisonRow
        validator = ResultValidator()
        rows = [
            ComparisonRow("n_valid", 1813, 1813, 0, 0, "VALIDADO", "ok"),
            ComparisonRow("T_global", 50.7, 50.8, 0.1, 0.2, "VALIDADO", "ok"),
        ]
        summary = validator.summary(rows)
        assert "VALIDADOS" in summary or "Validados" in summary


# ── Pruebas de validacion psicometrica ──────────────────────────────────

class TestPsychometricValidator:
    def test_validate_all(self, sample_data, item_cols):
        validator = PsychometricValidator()
        results = validator.validate_all(sample_data, item_cols)
        assert len(results) > 0

    def test_summary(self, sample_data, item_cols):
        validator = PsychometricValidator()
        results = validator.validate_all(sample_data, item_cols)
        summary = validator.summary(results)
        assert "VALIDACION PSICOMETRICA" in summary


# ── Pruebas de validacion IRT ──────────────────────────────────────────

class TestIRTValidator:
    def test_validate_with_t_global(self):
        validator = IRTValidator()
        df = pd.DataFrame({
            "theta": np.random.normal(0, 1, 100),
            "T_Global": np.random.normal(50, 10, 100),
        })
        results = validator.validate_all(df, [])
        assert len(results) > 0


# ── Pruebas de validacion de pesos ──────────────────────────────────────

class TestSurveyWeightValidator:
    def test_validate(self, sample_data):
        validator = SurveyWeightValidator()
        results = validator.validate(sample_data)
        assert len(results) > 0


# ── Pruebas de motor estadistico ────────────────────────────────────────

class TestStatisticalEngine:
    def test_select_comparison_method(self, sample_data):
        engine = StatisticalEngine()
        plan = engine.select_comparison_method(
            sample_data, "T_Global", "seccional"
        )
        assert plan.method in ("t-test_student", "t-test_welch", "mann_whitney",
                                "anova", "kruskal_wallis")

    def test_descriptive_summary(self, sample_data):
        engine = StatisticalEngine()
        result = engine.descriptive_summary(sample_data, "T_Global")
        assert "n" in result
        assert "media" in result


# ── Pruebas de orquestador ──────────────────────────────────────────────

class TestOrchestrator:
    def test_create_agent(self):
        agent = StatisticalAgent()
        assert agent is not None

    def test_fallback_plan(self):
        agent = StatisticalAgent()
        plan = agent._fallback_plan("Cual es el bienestar general?")
        assert plan.analysis_type.value in ("resumen", "descriptivo")

    def test_classify_t(self):
        agent = StatisticalAgent()
        assert agent._classify_t(35) == "Muy bajo"
        assert agent._classify_t(42) == "Bajo"
        assert agent._classify_t(50) == "Medio"
        assert agent._classify_t(57) == "Alto"
        assert agent._classify_t(65) == "Muy alto"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
