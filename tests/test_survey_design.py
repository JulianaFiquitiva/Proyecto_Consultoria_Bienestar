# -*- coding: utf-8 -*-
"""
Inferencia con el diseño muestral (entrega 2, pregunta 2.1) y validación
contra los resultados reportados (pregunta 2.0).

Todo con datos sintéticos: los casos de control tienen solución conocida
(muestreo aleatorio simple, estratos independientes).
"""
import numpy as np
import pandas as pd
import pytest

from agent.statistics.survey_design import SurveyDesign
from agent.validation.result_validator import ResultValidator
from agent.validation.survey_weights import SurveyWeightValidator


def _mas(n=400, N=10000, seed=1):
    """Muestreo aleatorio simple: un estrato y pesos iguales N/n."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "y": rng.normal(50, 10, n),
        "w": np.full(n, N / n),
        "estrato": "unico",
        "N_h": N,
    })


def _estratificado(seed=2):
    """Dos estratos con medias distintas y pesos distintos."""
    rng = np.random.default_rng(seed)
    a = pd.DataFrame({"y": rng.normal(45, 8, 300), "estrato": "A",
                      "N_h": 6000, "w": 6000 / 300})
    b = pd.DataFrame({"y": rng.normal(55, 8, 200), "estrato": "B",
                      "N_h": 2000, "w": 2000 / 200})
    df = pd.concat([a, b], ignore_index=True)
    df["grupo"] = np.where(rng.random(len(df)) < 0.5, "g1", "g2")
    return df


def _diseno(df):
    return SurveyDesign(df, weight_col="w", strata_cols=["estrato"], pop_col="N_h")


class TestMuestreoAleatorioSimple:
    def test_media_y_error_estandar_de_mas(self):
        df = _mas()
        est = _diseno(df).mean("y")
        n, N = len(df), 10000
        se_mas = np.sqrt((1 - n / N) * df["y"].var(ddof=1) / n)
        assert est.estimate == pytest.approx(df["y"].mean())
        assert est.se == pytest.approx(se_mas, rel=1e-9)

    def test_deff_es_uno_en_mas(self):
        assert _diseno(_mas()).mean("y").deff == pytest.approx(1.0, rel=1e-9)

    def test_intervalo_al_95(self):
        est = _diseno(_mas()).mean("y")
        assert est.ci_lower == pytest.approx(est.estimate - 1.959964 * est.se, rel=1e-6)
        assert est.ci_upper == pytest.approx(est.estimate + 1.959964 * est.se, rel=1e-6)

    def test_proporcion(self):
        df = _mas()
        est = _diseno(df).proportion("y", lambda y: y < 40)
        assert est.estimate == pytest.approx((df["y"] < 40).mean())
        assert 0 < est.se < 0.05


class TestEstratificado:
    def test_media_ponderada(self):
        df = _estratificado()
        esperada = np.average(df["y"], weights=df["w"])
        assert _diseno(df).mean("y").estimate == pytest.approx(esperada)

    def test_varianza_es_suma_de_varianzas_por_estrato(self):
        """Media estratificada: V = Σ (N_h/N)² (1 − f_h) S²_h / n_h."""
        df = _estratificado()
        N = 8000
        esperada = sum(
            (g["N_h"].iloc[0] / N) ** 2
            * (1 - len(g) / g["N_h"].iloc[0])
            * g["y"].var(ddof=1) / len(g)
            for _, g in df.groupby("estrato")
        )
        assert _diseno(df).mean("y").se == pytest.approx(np.sqrt(esperada), rel=1e-9)

    def test_estratificar_reduce_la_varianza(self):
        """Estratos con medias muy distintas: deff < 1."""
        assert _diseno(_estratificado()).mean("y").deff < 1

    def test_diferencia_entre_dominios(self):
        d = _diseno(_estratificado())
        dif = d.difference("y", "grupo", "g1", "g2")
        e1, e2 = d.mean("y", "grupo", "g1"), d.mean("y", "grupo", "g2")
        assert dif.difference == pytest.approx(e1.estimate - e2.estimate)
        assert dif.ci_lower < dif.difference < dif.ci_upper
        assert 0 <= dif.p_value <= 1

    def test_diferencia_detecta_un_efecto_real(self):
        df = _estratificado()
        df.loc[df["grupo"] == "g1", "y"] += 5
        dif = _diseno(df).difference("y", "grupo", "g1", "g2")
        assert dif.p_value < 0.001 and dif.significant

    def test_pairwise_ajusta_por_bonferroni_y_filtra_grupos_pequenos(self):
        df = _estratificado()
        df["sec"] = np.resize(["a", "b", "c"], len(df))
        df.loc[:9, "sec"] = "pequeno"          # n = 10 < 30
        pares = _diseno(df).pairwise("y", "sec", min_n=30)
        assert len(pares) == 3                 # a-b, a-c, b-c
        for p in pares:
            assert p.p_adjusted == pytest.approx(min(p.p_value * 3, 1.0))
            assert "pequeno" not in (p.group1, p.group2)

    def test_pairwise_con_grupos_numericos(self):
        """Regresion: el estrato socioeconomico es numerico (1.0, 2.0...)."""
        df = _estratificado()
        df["estrato_se"] = np.resize([1.0, 2.0, 3.0], len(df))
        pares = _diseno(df).pairwise("y", "estrato_se", min_n=30)
        assert {(p.group1, p.group2) for p in pares} == {
            ("1.0", "2.0"), ("1.0", "3.0"), ("2.0", "3.0")}

    def test_faltan_columnas_del_diseno(self):
        with pytest.raises(ValueError, match="Faltan columnas"):
            SurveyDesign(pd.DataFrame({"y": [1.0]}), weight_col="w")


class TestRegresionDeDiseno:
    def _datos(self, seed=4, n=500):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "x2": rng.binomial(1, 0.4, n).astype(float),
            "w": rng.uniform(1, 20, n),
            "estrato": "unico",
            "N_h": 10 ** 9,                    # fpc despreciable
        })
        df["y"] = 50 + 2 * df["x1"] - 3 * df["x2"] + rng.normal(0, 5 + 3 * df["x2"], n)
        return df

    def test_coeficientes_iguales_a_wls(self):
        sm = pytest.importorskip("statsmodels.api")
        df = self._datos()
        reg = _diseno(df).regression("y", df[["x1", "x2"]])
        fit = sm.WLS(df["y"], sm.add_constant(df[["x1", "x2"]]), weights=df["w"]).fit()
        np.testing.assert_allclose(reg.coefficients["coef"].values, fit.params.values, rtol=1e-9)

    def test_un_estrato_equivale_a_sandwich_robusto(self):
        """Sin estratos ni fpc: V = HC0 · n/(n−1)."""
        sm = pytest.importorskip("statsmodels.api")
        df = self._datos()
        reg = _diseno(df).regression("y", df[["x1", "x2"]])
        fit = sm.WLS(df["y"], sm.add_constant(df[["x1", "x2"]]),
                     weights=df["w"]).fit(cov_type="HC0")
        n = len(df)
        np.testing.assert_allclose(reg.coefficients["se"].values,
                                   fit.bse.values * np.sqrt(n / (n - 1)), rtol=1e-6)

    def test_excluye_filas_con_faltantes(self):
        df = self._datos()
        df.loc[:9, "x1"] = np.nan
        assert _diseno(df).regression("y", df[["x1", "x2"]]).n == len(df) - 10


class TestDeffKish:
    def test_pesos_iguales_dan_uno(self):
        w = pd.Series(np.full(100, 7.0))
        assert ResultValidator()._compute_deff(w) == pytest.approx(1.0)
        assert SurveyWeightValidator()._compute_deff(w) == pytest.approx(1.0)

    def test_formula(self):
        w = pd.Series([1.0, 1.0, 2.0, 4.0])
        esperado = 4 * (1 + 1 + 4 + 16) / 8 ** 2
        assert ResultValidator()._compute_deff(w) == pytest.approx(esperado)

    def test_el_deff_no_es_la_muestra_efectiva(self):
        """Regresión: antes se devolvía (Σw)²/Σw² (= n efectivo)."""
        w = pd.Series(np.random.default_rng(3).uniform(1, 30, 1813))
        assert 1 <= ResultValidator()._compute_deff(w) < 5


class TestToleranciaPorRedondeo:
    @pytest.mark.parametrize("reportado,reproducido,estado", [
        (50.7, 50.6845, "VALIDADO"),             # redondea a 50.7
        (54.0, 54.0471, "VALIDADO"),             # 54.0 conserva su decimal
        (54.0, 54.06, "DIFERENCIA MENOR"),       # ya no redondea a 54.0
        (0.148, 0.1481, "VALIDADO"),
        (0.192, 0.1937, "DIFERENCIA MENOR"),
        (51.2, 50.6845, "DIFERENCIA IMPORTANTE"),  # 1,007 % > 1 % (Informe Ejecutivo)
        (0.296, 0.1937, "DIFERENCIA IMPORTANTE"),
        (1619, 1618.58, "VALIDADO"),
    ])
    def test_estado(self, reportado, reproducido, estado):
        nombre = "high_proportion" if reportado in (0.192, 0.296) else "T_global_mean"
        assert ResultValidator()._compare(nombre, reportado, reproducido).status == estado


class TestReproduccion:
    def _datos(self):
        df = _estratificado()
        df = df.rename(columns={"y": "T_Global", "w": "factor_expansion",
                                "N_h": "N_poblacion_estrato"})
        df["seccional"] = df["estrato"]
        df["modalidad_exp"] = np.where(df.index % 3 == 0, "Posgrado", "Pregrado")
        return df

    def test_media_y_proporciones_ponderadas(self):
        df = self._datos()
        r = ResultValidator().reproduce(df_irt=df)
        w = df["factor_expansion"]
        assert r["T_global_mean"] == pytest.approx(np.average(df["T_Global"], weights=w))
        assert r["risk_proportion"] == pytest.approx(np.average(df["T_Global"] < 40, weights=w))
        assert r["T_global_ci_lower"] < r["T_global_mean"] < r["T_global_ci_upper"]
        assert r["pre_posgrado_gap"] == pytest.approx(
            r["pre_posgrado_T_pos"] - r["pre_posgrado_T_pre"])

    def test_tabla_de_fuentes(self):
        df = self._datos()
        tabla = ResultValidator().compare_sources(
            {"A": {"T_global_mean": 48.0}, "B": {"T_global_mean": 99.0}}, df_irt=df)
        fila = tabla[tabla["Indicador"] == "T global poblacional"].iloc[0]
        assert fila["A"] == 48.0 and fila["B"] == 99.0
        assert fila["Estado vs. B"] == "DIFERENCIA IMPORTANTE"
