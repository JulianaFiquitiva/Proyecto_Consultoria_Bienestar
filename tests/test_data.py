"""
Pruebas para el módulo de datos.

Verifica: carga de dataset, número de registros, número de ítems,
inversión de ítems, cálculo de dimensiones, factores de expansión.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Agregar directorio raíz al path
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


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Dataset de ejemplo para pruebas (simula estructura del proyecto)."""
    np.random.seed(42)
    n = 200
    data = {
        "seccional": np.random.choice(["Bogotá", "Tunja", "Bucaramanga"], n),
        "Programa de formación": np.random.choice(["Ing. Sistemas", "Derecho"], n),
        "Núcleo básico de conocimiento": np.random.choice(["Ingeniería", "Derecho"], n),
    }
    # 29 ítems con nombres que empiezan con número (simula encuesta)
    for i in range(1, 30):
        data[f"{i}. Item de la escala"] = np.random.randint(1, 7, n)
    data["Métrica Global"] = sum(data[f"{i}. Item de la escala"] for i in range(1, 30))
    data["factor_expansion"] = np.random.uniform(5, 25, n)
    return pd.DataFrame(data)


@pytest.fixture
def item_cols():
    """Lista de columnas de ítems."""
    return [f"{i}. Item de la escala" for i in range(1, 30)]


# ── Pruebas de configuración ─────────────────────────────────────────────

class TestConfig:
    """Pruebas de configuración del instrumento."""

    def test_n_items(self):
        """Verifica que hay 29 ítems."""
        assert N_ITEMS == 29

    def test_negative_items_count(self):
        """Verifica que hay 10 ítems negativos."""
        assert len(NEGATIVE_ITEMS) == 10

    def test_dimensions_count(self):
        """Verifica que hay 6 dimensiones."""
        assert len(DIMENSION_MAP) == 6

    def test_all_items_in_dimensions(self):
        """Verifica que todos los ítems están en alguna dimensión."""
        all_items = set()
        for items in DIMENSION_MAP.values():
            all_items.update(items)
        assert all_items == set(range(1, 30))

    def test_dimension_order(self):
        """Verifica el orden de las dimensiones."""
        assert len(DIMENSION_ORDER) == 6
        assert DIMENSION_ORDER[0] == "Autoaceptación"


# ── Pruebas de carga de datos ────────────────────────────────────────────

class TestDataLoader:
    """Pruebas del cargador de datos."""

    def test_get_item_columns(self, sample_data, item_cols):
        """Verifica identificación de columnas de ítems."""
        loader = DataLoader()
        # Con columna Métrica Global
        found = loader.get_item_columns(sample_data)
        # El dataset de ejemplo tiene exactamente 29 columnas de ítems
        assert len(found) == 29

    def test_invert_negative_items(self, sample_data, item_cols):
        """Verifica inversión de ítems negativos."""
        loader = DataLoader()
        original = sample_data.copy()
        inverted = loader.invert_negative_items(sample_data, item_cols)

        for item_num in NEGATIVE_ITEMS:
            col = item_cols[item_num - 1]
            # Verificar que se invirtió: new = 7 - old
            assert inverted[col].mean() == pytest.approx(
                7 - original[col].mean(), rel=1e-10
            )

    def test_compute_dimensions(self, sample_data, item_cols):
        """Verifica cálculo de dimensiones."""
        loader = DataLoader()
        result = loader.compute_dimensions(sample_data, item_cols)

        # Verificar que existen las columnas de dimensiones
        for dim in DIMENSION_ORDER:
            assert f"Dim_{dim}" in result.columns

        # Verificar que Métrica Global existe
        assert "Métrica Global" in result.columns

        # Verificar razonabilidad de valores
        for dim in DIMENSION_ORDER:
            col = f"Dim_{dim}"
            assert result[col].mean() >= 1
            assert result[col].mean() <= 6


# ── Pruebas de profiling ─────────────────────────────────────────────────

class TestDataProfiler:
    """Pruebas del data profiler."""

    def test_basic_profile(self, sample_data):
        """Verifica perfil básico del dataset."""
        profiler = DataProfiler()
        profile = profiler.profile(sample_data)

        assert profile.n_rows == 200
        assert profile.n_columns > 0
        assert len(profile.columns) > 0

    def test_summary(self, sample_data):
        """Verifica generación de resumen."""
        profiler = DataProfiler()
        profile = profiler.profile(sample_data)
        summary = profiler.summary(profile)

        assert "PERFIL DEL DATASET" in summary
        assert "Filas:" in summary

    def test_sensitive_detection(self):
        """Verifica detección de variables sensibles."""
        profiler = DataProfiler()
        df = pd.DataFrame({
            "nombre": ["Ana", "Luis"],
            "correo": ["a@b.com", "c@d.com"],
            "edad": [20, 22],
        })
        profile = profiler.profile(df)
        assert "nombre" in profile.sensitive_columns
        assert "correo" in profile.sensitive_columns


# ── Pruebas de validación ────────────────────────────────────────────────

class TestDataValidator:
    """Pruebas del validador de datos."""

    def test_validate_prepared_structure(self, sample_data):
        """Verifica validación de estructura del dataset preparado."""
        validator = DataValidator()
        # El dataset de ejemplo no tiene todas las columnas esperadas
        # pero verificar que el validador funciona
        report = validator.validate_prepared(sample_data)
        assert report.n_checks > 0

    def test_validate_irt(self, sample_data):
        """Verifica validación del dataset IRT."""
        # Agregar T_Global
        sample_data["T_Global"] = np.random.normal(50, 10, len(sample_data))
        validator = DataValidator()
        report = validator.validate_irt(sample_data)
        assert report.n_checks > 0


# ── Pruebas de estadísticos descriptivos ─────────────────────────────────

class TestDescriptiveStats:
    """Pruebas de estadísticos descriptivos."""

    def test_weighted_mean(self):
        """Verifica media ponderada."""
        from agent.statistics.descriptive import DescriptiveStats
        x = np.array([1, 2, 3, 4, 5])
        w = np.array([1, 1, 1, 1, 1])
        assert DescriptiveStats.weighted_mean(x, w) == 3.0

    def test_weighted_mean_unequal_weights(self):
        """Verifica media ponderada con pesos desiguales."""
        from agent.statistics.descriptive import DescriptiveStats
        x = np.array([1, 2, 3])
        w = np.array([1, 2, 1])
        expected = (1 * 1 + 2 * 2 + 3 * 1) / 4
        assert DescriptiveStats.weighted_mean(x, w) == pytest.approx(expected)

    def test_weighted_variance(self):
        """Verifica varianza ponderada."""
        from agent.statistics.descriptive import DescriptiveStats
        x = np.array([1, 2, 3, 4, 5])
        w = np.array([1, 1, 1, 1, 1])
        var = DescriptiveStats.weighted_variance(x, w)
        assert var == pytest.approx(np.var(x, ddof=1), rel=1e-10)

    def test_weighted_std(self):
        """Verifica desviación estándar ponderada."""
        from agent.statistics.descriptive import DescriptiveStats
        x = np.array([1, 2, 3, 4, 5])
        w = np.array([1, 1, 1, 1, 1])
        std = DescriptiveStats.weighted_std(x, w)
        assert std == pytest.approx(np.std(x, ddof=1), rel=1e-10)

    def test_weighted_median(self):
        """Verifica mediana ponderada."""
        from agent.statistics.descriptive import DescriptiveStats
        x = np.array([1, 2, 3, 4, 5])
        w = np.array([1, 1, 1, 1, 1])
        assert DescriptiveStats.weighted_median(x, w) == 3.0

    def test_weighted_correlation(self):
        """Verifica correlación ponderada."""
        from agent.statistics.descriptive import DescriptiveStats
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        w = np.array([1, 1, 1, 1, 1])
        corr = DescriptiveStats.weighted_correlation(x, y, w)
        assert corr == pytest.approx(1.0, rel=1e-10)

    def test_weighted_cronbach_alpha(self):
        """Verifica alpha de Cronbach ponderado."""
        from agent.statistics.descriptive import DescriptiveStats
        np.random.seed(42)
        data = pd.DataFrame(np.random.randint(1, 7, (100, 5)))
        w = np.ones(100)
        alpha = DescriptiveStats.weighted_cronbach_alpha(data, w)
        assert 0 <= alpha <= 1


# ── Pruebas de psicometría ──────────────────────────────────────────────

class TestPsychometricScale:
    """Pruebas del análisis psicométrico."""

    def test_validate_item_inversion(self, sample_data, item_cols):
        """Verifica validación de inversión de ítems."""
        from agent.psychometrics.scale import PsychometricScale

        scale = PsychometricScale()
        inverted = scale._invert_items(sample_data, item_cols) if hasattr(
            scale, '_invert_items'
        ) else sample_data.copy()

        # Invertir manualmente
        for i in NEGATIVE_ITEMS:
            col = item_cols[i - 1]
            inverted[col] = 7 - inverted[col]

        results = scale.validate_item_inversion(sample_data, inverted, item_cols)

        for item_num in NEGATIVE_ITEMS:
            assert results[item_num]["difference"] < 0.01


# ── Pruebas de IRT ───────────────────────────────────────────────────────

class TestIRTAnalyzer:
    """Pruebas del analizador IRT."""

    def test_compute_t_scores(self):
        """Verifica conversión θ → T-score."""
        from agent.psychometrics.irt import IRTAnalyzer

        analyzer = IRTAnalyzer()
        theta = np.array([-2, -1, 0, 1, 2])
        t_scores = analyzer.compute_t_scores(theta, method="prior")

        expected = np.array([30, 40, 50, 60, 70])
        np.testing.assert_array_almost_equal(t_scores, expected)

    def test_classify_wellbeing(self):
        """Verifica clasificación de bienestar."""
        from agent.psychometrics.irt import IRTAnalyzer

        analyzer = IRTAnalyzer()
        t_scores = np.array([35, 42, 50, 58, 65])
        classification = analyzer.classify_wellbeing(t_scores)

        assert classification["n_total"] == 5
        assert classification["counts"]["Muy bajo"] == 1
        assert classification["counts"]["Bajo"] == 1
        assert classification["counts"]["Medio"] == 1
        assert classification["counts"]["Alto"] == 1
        assert classification["counts"]["Muy alto"] == 1


# ── Pruebas de regresión ────────────────────────────────────────────────

class TestRegressionEngine:
    """Pruebas del motor de regresión."""

    def test_auto_select_method_ols(self):
        """Verifica selección automática OLS."""
        from agent.statistics.regression import RegressionEngine

        engine = RegressionEngine()
        df = pd.DataFrame({
            "y": np.random.normal(0, 1, 100),
            "x1": np.random.normal(0, 1, 100),
            "x2": np.random.normal(0, 1, 100),
        })
        method = engine.auto_select_method(df, "y", ["x1", "x2"])
        assert method == "ols"

    def test_auto_select_method_logistic(self):
        """Verifica selección automática logística."""
        from agent.statistics.regression import RegressionEngine

        engine = RegressionEngine()
        df = pd.DataFrame({
            "y": np.random.choice([0, 1], 100),
            "x1": np.random.normal(0, 1, 100),
        })
        method = engine.auto_select_method(df, "y", ["x1"])
        assert method == "logistic"

    def test_ols_basic(self):
        """Verifica OLS básico."""
        from agent.statistics.regression import RegressionEngine

        engine = RegressionEngine()
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = 2 * x + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "x": x})

        result = engine.ols(df, "y", ["x"])
        assert result.r_squared > 0.5
        assert result.f_p_value < 0.05
        # El coeficiente de x debe ser cercano a 2
        assert abs(result.coefficients.loc["x", "coef"] - 2) < 0.3


# ── Pruebas de validación de resultados ──────────────────────────────────

class TestResultValidator:
    """Pruebas del validador de resultados."""

    def test_comparison_table(self):
        """Verifica generación de tabla de comparación."""
        from agent.validation.result_validator import ResultValidator, ComparisonRow

        validator = ResultValidator()
        rows = [
            ComparisonRow("n_valid", 1813, 1813, 0, 0, "VALIDADO", "ok"),
            ComparisonRow("T_global", 50.7, 50.8, 0.1, 0.2, "VALIDADO", "ok"),
        ]
        table = validator.generate_comparison_table(rows)
        assert len(table) == 2
        assert "Resultado" in table.columns

    def test_summary(self):
        """Verifica generación de resumen."""
        from agent.validation.result_validator import ResultValidator, ComparisonRow

        validator = ResultValidator()
        rows = [
            ComparisonRow("n_valid", 1813, 1813, 0, 0, "VALIDADO", "ok"),
            ComparisonRow("T_global", 50.7, 50.8, 0.1, 0.2, "VALIDADO", "ok"),
        ]
        summary = validator.summary(rows)
        assert "VALIDADOS" in summary or "Validados" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
