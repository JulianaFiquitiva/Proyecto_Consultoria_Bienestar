# -*- coding: utf-8 -*-
"""
Hallazgos de la entrega 2 en un solo lugar (dashboard y diapositivas).

    python -m agent.reports.entrega2            # calcula y guarda
    python -m agent.reports.entrega2 --sin-dif  # sin DIF (~70 s menos)

Escribe resultados/entrega2/hallazgos_entrega2.json (solo agregados: nunca
filas de estudiantes ni grupos con n < 30). No modifica ningún otro archivo.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.core.config import (DIMENSION_ORDER, ITEM_LABELS, PROJECT_ROOT,
                               REPORTED_INFORME_EJECUTIVO, REPORTED_RESULTS)

SALIDA = PROJECT_ROOT / "resultados" / "entrega2" / "hallazgos_entrega2.json"
MIN_N = 30

# variable de agrupación -> etiqueta para mostrar
COMPARACIONES = {
    "seccional": "Seccional",
    "modalidad_exp": "Pregrado / posgrado",
    "Género": "Género",
    "Estrato socioeconómico": "Estrato socioeconómico",
    "Nivel de formación": "Nivel de formación",
}


def _est(e) -> Dict[str, Any]:
    return {"media": round(e.estimate, 3), "ic": [round(e.ci_lower, 3), round(e.ci_upper, 3)],
            "se": round(e.se, 4), "deff": round(e.deff, 3) if e.deff else None,
            "n": e.n, "N": round(e.population)}


def _etiqueta_grupo(g: str) -> str:
    """'3.0' -> 'Estrato 3' para el estrato socioeconómico."""
    try:
        return f"Estrato {int(float(g))}"
    except ValueError:
        return g


def _seleccion_metodo(df: pd.DataFrame, col: str) -> Optional[Dict[str, Any]]:
    """Cómo elige la prueba el agente de la entrega 1 (supuestos -> prueba) para
    esta comparación. Solo estadísticos agregados, para mostrarlo en la app."""
    from agent.statistics.statistical_engine import StatisticalEngine
    try:
        plan, res = StatisticalEngine().run_comparison(df, "T_Global", col, "factor_expansion")
    except Exception:          # la app sigue sin esta parte si una prueba falla
        return None
    if plan.method == "none":
        return None

    def tildes(s):       # el motor escribe sin tildes
        return s.replace("parametrica", "paramétrica").replace("una via", "una vía")

    return {
        "prueba": plan.test_name, "razon": tildes(plan.reason),
        "alternativa": tildes(plan.alternative),
        "supuestos": [{"nombre": a.name.replace(a.name.removeprefix("Normalidad en "),
                                                _etiqueta_grupo(a.name.removeprefix(
                                                    "Normalidad en "))),
                       "prueba": a.test_name, "p": float(a.p_value),
                       "cumple": bool(a.passed)} for a in plan.assumptions_checked],
        "resultado": None if res is None else {
            "prueba": res.test_name, "estadistico": round(float(res.statistic), 3),
            "p": float(res.p_value), "efecto": round(float(res.effect_size), 3)
            if res.effect_size is not None else None,
            "efecto_nombre": res.effect_size_name},
    }


def calcular(prep: pd.DataFrame, irt: pd.DataFrame, incluir_dif: bool = True) -> Dict[str, Any]:
    from agent.statistics.gap_analysis import adjusted_gap
    from agent.statistics.multilevel import icc_programa
    from agent.statistics.survey_design import SurveyDesign
    from agent.validation.result_validator import ResultValidator

    d = SurveyDesign(irt)
    t = irt["T_Global"]
    w = irt["factor_expansion"]
    g = d.mean("T_Global")
    out: Dict[str, Any] = {"generado": datetime.now().isoformat(timespec="seconds"),
                           "min_n_grupo": MIN_N}

    # ── Resumen poblacional ────────────────────────────────────────────
    riesgo = d.proportion("T_Global", lambda s: s < 40)
    alto = d.proportion("T_Global", lambda s: s >= 60)
    out["resumen"] = {
        "T_global": _est(g),
        "riesgo": _est(riesgo),
        "alto": _est(alto),
        "media_sin_ponderar": round(float(t.mean()), 3),
        "n": int(len(irt)), "N": round(float(w.sum())),
        "deff_kish_pesos": round(d.kish_deff(), 3),
        "histograma": np.histogram(t, bins=np.arange(10, 92, 4), weights=w)[0].round(1).tolist(),
        "histograma_bordes": np.arange(10, 92, 4).tolist(),
    }

    # ── 2.0 Reproducibilidad ───────────────────────────────────────────
    v = ResultValidator()
    tabla = v.compare_sources({"Referencia entrega 1": REPORTED_RESULTS,
                               "Informe Ejecutivo": REPORTED_INFORME_EJECUTIVO}, prep, irt)
    out["reproducibilidad"] = tabla.astype(object).where(tabla.notna(), None).to_dict(orient="records")

    # ── 2.1 IC: tres maneras de calcular el mismo intervalo ────────────
    n = len(t)
    se_mas = float(t.std(ddof=1) / np.sqrt(n))
    from agent.statistics.descriptive import DescriptiveStats as D
    se_pond = float(D.weighted_std(t, w) / np.sqrt(n))
    mp = float(np.average(t, weights=w))
    out["ic_metodos"] = [
        {"metodo": "Muestral, sin pesos (MAS)", "media": round(float(t.mean()), 3),
         "ic": [round(float(t.mean()) - 1.96 * se_mas, 3), round(float(t.mean()) + 1.96 * se_mas, 3)],
         "se": round(se_mas, 4)},
        {"metodo": "Ponderada, varianza de MAS", "media": round(mp, 3),
         "ic": [round(mp - 1.96 * se_pond, 3), round(mp + 1.96 * se_pond, 3)], "se": round(se_pond, 4)},
        {"metodo": "Ponderada, varianza del diseño (Taylor)", "media": round(g.estimate, 3),
         "ic": [round(g.ci_lower, 3), round(g.ci_upper, 3)], "se": round(g.se, 4)},
    ]

    # ── Dimensiones ───────────────────────────────────────────────────
    out["dimensiones"] = {dim: _est(d.mean(f"T_{dim}")) for dim in DIMENSION_ORDER
                          if f"T_{dim}" in irt.columns}

    # ── Comparaciones por grupo (n >= 30) ─────────────────────────────
    comp = {}
    for col, etiqueta in COMPARACIONES.items():
        if col not in irt.columns:
            continue
        medias = d.domain_means("T_Global", col, min_n=MIN_N)
        pares = d.pairwise("T_Global", col, min_n=MIN_N)
        excluidos = irt[col].value_counts()
        excluidos = excluidos[excluidos < MIN_N]
        grupos = {_etiqueta_grupo(k): _est(e) for k, e in
                  sorted(medias.items(), key=lambda kv: -kv[1].estimate)}
        # Niveles de bienestar y perfil de dimensiones de cada grupo (n >= 30)
        for valor in irt[col].dropna().unique():
            if str(valor) not in medias:
                continue
            grupos[_etiqueta_grupo(str(valor))].update({
                "riesgo": _est(d.proportion("T_Global", lambda s: s < 40, col, valor)),
                "alto": _est(d.proportion("T_Global", lambda s: s >= 60, col, valor)),
                "dimensiones": {dim: _est(d.mean(f"T_{dim}", col, valor))
                                for dim in DIMENSION_ORDER if f"T_{dim}" in irt.columns},
            })
        comp[col] = {
            "etiqueta": etiqueta,
            "grupos": grupos,
            "pares": [{**p.as_dict(3), "grupo_1": _etiqueta_grupo(p.group1),
                       "grupo_2": _etiqueta_grupo(p.group2)} for p in pares],
            "excluidos": {"categorias": int(len(excluidos)), "casos": int(excluidos.sum())},
            "seleccion": _seleccion_metodo(irt[irt[col].astype(str).isin(medias)], col),
        }
    out["comparaciones"] = comp

    # ── 2.3 Brecha ajustada ───────────────────────────────────────────
    out["brecha"] = adjusted_gap(irt)

    # ── 2.4 Multinivel ────────────────────────────────────────────────
    out["multinivel"] = icc_programa(irt)

    # ── 2.2 DIF (lento) ───────────────────────────────────────────────
    if incluir_dif:
        from agent.data.loader import DataLoader
        from agent.psychometrics.dif import dif_escala
        items = DataLoader().get_item_columns(prep)
        etiquetas = {c: f"{i}. {ITEM_LABELS[i]}" for i, c in enumerate(items, 1)}
        out["dif"] = {}
        for clave, (col, focal, ref) in {"genero": ("Género", "Femenino", "Masculino"),
                                         "nivel": ("modalidad_exp", "Posgrado", "Pregrado")}.items():
            r = dif_escala(prep[items], prep[col], focal, ref, etiquetas)
            r["tabla"] = r["tabla"].to_dict(orient="records")
            out["dif"][clave] = r
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")


def guardar(hallazgos: Dict[str, Any], ruta: Path = SALIDA) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(hallazgos, ensure_ascii=False, indent=1,
                               default=_json_default), encoding="utf-8")
    return ruta


def cargar(ruta: Path = SALIDA) -> Optional[Dict[str, Any]]:
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Calcula los hallazgos de la entrega 2")
    ap.add_argument("--sin-dif", action="store_true", help="omite el DIF (lento)")
    ap.add_argument("--salida", type=Path, default=SALIDA)
    args = ap.parse_args(argv)

    from agent.data.loader import DataLoader
    loader = DataLoader()
    prep, irt = loader.load_prepared(), loader.load_irt()
    print("Calculando hallazgos de la entrega 2"
          + (" (sin DIF)" if args.sin_dif else " (el DIF tarda ~70 s)") + "...")
    ruta = guardar(calcular(prep, irt, incluir_dif=not args.sin_dif), args.salida)
    print(f"Guardado en {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
