# -*- coding: utf-8 -*-
"""
Encuesta viva (agent/encuesta_viva): formulario -> hallazgos -> app.

- Con data/: el pipeline reproduce exactamente los archivos 01 y 03 del
  estudio (preparación, calibración por género y puntajes IRT).
- Sin data/: un formulario sintético recorre todo el camino, incluida la
  página «Encuesta en vivo» de la app.
"""
import json

import numpy as np
import pandas as pd
import pytest

from agent.encuesta_viva import actualizar as A
from agent.encuesta_viva import pipeline as P
from agent.encuesta_viva.catalogos import COLUMNAS_FORMULARIO

SEDES = {
    "Sede Principal Bogotá": ("Programas de formación Sede Principal Bogotá y centros de "
                              "atención universitaria CAU", "DERECHO"),
    "Seccional Tunja": ("Programas de formación\xa0Seccional Tunja", "INGENIERÍA CIVIL"),
    "Seccional Bucaramanga": ("Programas de formación\xa0Seccional Bucaramanga", "PSICOLOGÍA"),
    "Seccional Villavicencio": ("Programas de formación\xa0Seccional Villavicencio",
                                "CONTADURÍA PÚBLICA"),
}


def formulario_sintetico(n=160, semilla=0, estilo_google=True) -> pd.DataFrame:
    """Respuestas inventadas con la estructura del formulario (sin datos reales)."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({c: pd.Series([None] * n, dtype=object) for c in COLUMNAS_FORMULARIO})
    df["ID"] = np.arange(1, n + 1)
    df["Consentimiento informado"] = "Acepto"
    df["Autorización expresa para el tratamiento de datos personales"] = "Acepto"
    df["¿Soy mayor de edad?"] = rng.choice(["Sí"] * 9 + ["No"], n)
    df["Género"] = rng.choice(["Femenino", "Masculino", "No binario"], n, p=[.5, .48, .02])
    df["Edad"] = rng.integers(18, 40, n)
    df["Estrato socioeconómico"] = rng.integers(1, 6, n)
    df["Nivel de formación"] = rng.choice(["Pregrado", "Especialización", "Maestría"], n,
                                          p=[.7, .2, .1])
    df["Modalidad"] = "Presencial"
    df["Semestre"] = rng.integers(1, 10, n)
    sedes = rng.choice(list(SEDES), n)
    df["Sede o seccional"] = sedes
    for i, s in enumerate(sedes):
        col, prog = SEDES[s]
        df.loc[i, col] = prog
    rasgo = rng.normal(0, 1, n)
    for col in COLUMNAS_FORMULARIO[-29:]:
        df[col] = np.clip(np.round(3.8 + rasgo + rng.normal(0, 1, n)), 1, 6)
    if estilo_google:   # Google Forms: sin ID/horas/correo, «Marca temporal», \xa0 -> espacio
        df = df.drop(columns=COLUMNAS_FORMULARIO[:6])
        df.columns = [c.replace("\xa0", " ") for c in df.columns]
        df.insert(0, "Marca temporal", "18/11/2026 10:00:00")
    return df


# ── Formulario ────────────────────────────────────────────────────────
def test_normaliza_formulario_de_google():
    g = formulario_sintetico(20)
    g = g[list(g.columns[::-1])]                 # otro orden de columnas
    out = P.normalizar_formulario(g)
    assert list(out.columns) == COLUMNAS_FORMULARIO
    assert out["ID"].tolist() == list(range(1, 21))


def test_formulario_sin_una_pregunta_falla_claro():
    g = formulario_sintetico(20).drop(columns=[COLUMNAS_FORMULARIO[-1]])
    with pytest.raises(P.FormularioInvalido, match="faltan 1"):
        P.normalizar_formulario(g)


def test_calibracion_respeta_los_dos_margenes():
    prep = P.preparar(P.normalizar_formulario(formulario_sintetico(200)))
    N = prep.groupby("strata_key")["factor_expansion"].sum()
    meta = prep.groupby("strata_key")["N_poblacion_estrato"].first()
    np.testing.assert_allclose(N, meta, rtol=1e-9)
    assert (prep["factor_expansion"] > 0).all()


# ── Puntajes ──────────────────────────────────────────────────────────
def test_eap_con_parametros_fijos_en_escala_del_estudio():
    par = json.loads(P.PARAMETROS_GRM.read_text(encoding="utf-8"))
    assert set(par) == {"Global", *P.DIM_ORDER}
    todo6 = np.full((1, 29), 6.0)
    todo1 = np.full((1, 29), 1.0)
    alto, bajo = P.theta_eap(todo6, par["Global"]), P.theta_eap(todo1, par["Global"])
    assert alto[0] > 1.5 and bajo[0] < -2.5
    # Un ítem sin respuesta no rompe el cálculo
    medio = np.full((1, 29), 4.0)
    medio[0, 3] = np.nan
    assert np.isfinite(P.theta_eap(medio, par["Global"])).all()


# ── Actualización y app ───────────────────────────────────────────────
@pytest.fixture(scope="module")
def salida_viva(tmp_path_factory):
    d = tmp_path_factory.mktemp("vivo")
    fuente = d / "respuestas.csv"
    formulario_sintetico(160).to_csv(fuente, index=False, encoding="utf-8-sig")
    salida = d / "hallazgos.json"
    h = A.actualizar(str(fuente), salida)
    return fuente, salida, h


def test_actualizar_escribe_agregados(salida_viva):
    _, salida, h = salida_viva
    assert salida.exists()
    v = h["encuesta_viva"]
    assert v["respuestas_recibidas"] == 160
    assert 0 < v["respuestas_validas"] <= 160
    assert "reproducibilidad" not in h
    assert {"resumen", "comparaciones", "dimensiones"} <= set(h)
    texto = salida.read_text(encoding="utf-8")
    assert "Marca temporal" not in texto and "18/11/2026" not in texto   # nada individual


def test_actualizar_no_reescribe_si_no_cambia(salida_viva):
    fuente, salida, _ = salida_viva
    antes = salida.stat().st_mtime_ns
    h = A.actualizar(str(fuente), salida)
    assert h.get("sin_cambios") and salida.stat().st_mtime_ns == antes


def test_pagina_en_vivo(salida_viva, monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from tests.test_dashboard import APP, _hay_insumos
    if not _hay_insumos():
        pytest.skip("sin hallazgos del estudio")
    monkeypatch.delenv("APP_PASSWORD", raising=False)

    # Sin respuestas procesadas la página no está en el menú
    monkeypatch.setattr(A, "SALIDA", tmp_path / "no_existe.json")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.switch_page("paginas/en_vivo.py").run()     # no registrada: no se abre
    assert not any("Todavía no hay respuestas" in i.value for i in at.info)

    # Con respuestas aparece y muestra los resultados
    monkeypatch.setattr(A, "SALIDA", salida_viva[1])
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.switch_page("paginas/en_vivo.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Última actualización" in x.value for x in at.caption)


# ── Reproducción del estudio (necesita data/) ─────────────────────────
@pytest.mark.datos
def test_reproduce_preparado_e_irt_del_estudio():
    from agent.data.loader import DataLoader
    L = DataLoader()
    prep = P.preparar(L.load_raw(), L.load_sample_frames())
    ref = L.load_prepared()
    assert list(prep.columns) == list(ref.columns)
    for col in ["factor_expansion", "factor_expansion_original", "Métrica Global",
                "N_poblacion_estrato", "r_respondentes_estrato"]:
        np.testing.assert_allclose(prep[col].to_numpy(float), ref[col].to_numpy(float),
                                   rtol=1e-9, err_msg=col)
    # Sin marcos (constantes agregadas) da lo mismo
    np.testing.assert_allclose(P.preparar(L.load_raw())["factor_expansion"],
                               ref["factor_expansion"], rtol=1e-9)

    irt, r3 = P.estandarizar(prep), L.load_irt()
    assert list(irt.columns) == list(r3.columns)
    for esc in ["Global", *P.DIM_ORDER]:
        np.testing.assert_allclose(irt[f"θ_{esc}"], r3[f"θ_{esc}"], atol=1.5e-4)
        np.testing.assert_allclose(irt[f"T_{esc}"], r3[f"T_{esc}"], atol=1e-9)
        assert (irt[f"Nivel_{esc}"] == r3[f"Nivel_{esc}"]).all(), esc


# ── Formulario generado (docs/encuesta_viva) ──────────────────────────
def test_hoja_del_formulario_generado_es_aceptada():
    """Las columnas que crea crear_formulario.gs bastan para el pipeline."""
    import re
    from agent.encuesta_viva import formulario as F
    from agent.encuesta_viva.catalogos import PROGRAMA_A_NBC, PROGRAMAS_POR_SEDE
    gs = F.apps_script()
    titulos = (["Marca temporal", "Consentimiento informado",
                "Autorización expresa para el tratamiento de datos personales",
                "¿Soy mayor de edad?"]
               + [t for t, *_ in F.DATOS_PERSONALES]
               + ["Sede o seccional", "Centro de Atención Universitaria CAU"]
               + list(PROGRAMAS_POR_SEDE) + F.ITEMS)
    for t in titulos[1:]:
        assert json.dumps(t, ensure_ascii=False) in gs or f"'{t}'" in gs, t
    P.normalizar_formulario(pd.DataFrame([[None] * len(titulos)], columns=titulos))
    # Toda opción de programa tiene núcleo básico de conocimiento
    assert all(p in PROGRAMA_A_NBC for ps in PROGRAMAS_POR_SEDE.values() for p in ps)
    assert len(re.findall(r"addScaleItem", gs)) == 1 and len(F.ITEMS) == 29


def test_guia_del_formulario_esta_al_dia():
    from agent.encuesta_viva import formulario as F
    for nombre, texto in (("crear_formulario.gs", F.apps_script()), ("FORMULARIO.md", F.guia())):
        ruta = F.DESTINO / nombre
        assert ruta.read_text(encoding="utf-8") == texto, (
            f"{nombre} desactualizado: python -m agent.encuesta_viva.formulario")
