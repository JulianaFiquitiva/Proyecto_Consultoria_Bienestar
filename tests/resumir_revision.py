#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Une la hoja de revision ciega con la clave de condicion y calcula metricas.

Cruza por la columna "fila" (numero de fila de la hoja, 1..38):

    hoja_revision_ciega_<ts>.csv   ->  fila, pregunta, respuesta_esperada,
                                        documento_esperado, pagina_pdf_esperada,
                                        cita_textual_esperada, respuesta,
                                        fuentes_citadas, correcta,
                                        cita_correcta, observacion
    clave_condicion_<ts>.csv        ->  fila, id, condicion

Calcula, desglosado POR CONDICION, POR TIPO y por CONDICION x TIPO:
  n, n con correcta = si, proporcion, intervalo de Wilson al 95 %,
  y la proporcion de cita_correcta = si entre las filas donde aplica
  (es decir, donde cita_correcta != no_aplica).

Si falta alguna celda que alimenta el calculo (correcta o cita_correcta),
AVISA y NO CALCULA. "observacion" es texto libre: se reporta cuantas estan
vacias pero no bloquea el calculo.

Uso:
    python tests/resumir_revision.py
    python tests/resumir_revision.py --hoja ruta/a/hoja.csv --clave ruta/a/clave.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
BANCO = RAIZ / "tests" / "banco_preguntas_literatura.json"

VALORES_CORRECTA = {"si", "no", "parcial"}
VALORES_CITA = {"si", "no", "no_aplica"}
# Columnas que alimentan el calculo: sin ellas no se puede medir.
COLUMNAS_REQUERIDAS = ("correcta", "cita_correcta")

Z_95 = 1.959963984540054


def norm(valor: Optional[str]) -> str:
    """Minusculas y sin acentos: 'Si' y 'si' se leen igual."""
    texto = (valor or "").strip().lower()
    return "".join(c for c in unicodedata.normalize("NFD", texto)
                   if unicodedata.category(c) != "Mn")


def wilson(k: int, n: int, z: float = Z_95) -> Tuple[Optional[float], Optional[float]]:
    """Intervalo de confianza de Wilson al 95 % para una proporcion k/n."""
    if n <= 0:
        return (None, None)
    p = k / n
    denom = 1.0 + (z * z) / n
    centro = (p + (z * z) / (2.0 * n)) / denom
    radio = (z * ((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n)) ** 0.5) / denom
    return (max(0.0, centro - radio), min(1.0, centro + radio))


def mas_reciente(patrones: List[str]) -> Optional[Path]:
    candidatos: List[Path] = []
    for patron in patrones:
        candidatos.extend(Path(p) for p in glob.glob(str(patron)))
    if not candidatos:
        return None
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def leer(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def validar(hoja: List[Dict[str, str]]) -> List[str]:
    """Devuelve la lista de problemas que impiden calcular."""
    problemas: List[str] = []
    for reg in hoja:
        fila = reg.get("fila", "?")
        for col in COLUMNAS_REQUERIDAS:
            bruto = (reg.get(col) or "").strip()
            valor = norm(bruto)
            if not valor:
                problemas.append("fila %s: falta '%s'" % (fila, col))
            elif col == "correcta" and valor not in VALORES_CORRECTA:
                problemas.append("fila %s: correcta='%s' (no es si/no/parcial)"
                                 % (fila, bruto))
            elif col == "cita_correcta" and valor not in VALORES_CITA:
                problemas.append("fila %s: cita_correcta='%s' (no es si/no/no_aplica)"
                                 % (fila, bruto))
    return problemas


def metricas(filas: List[Dict[str, str]]) -> Dict:
    """filas ya cruzadas: llevan condicion, tipo, correcta, cita_correcta."""
    n = len(filas)
    n_si = sum(1 for f in filas if norm(f["correcta"]) == "si")
    conteo = {v: 0 for v in ("si", "no", "parcial")}
    for f in filas:
        clave_c = norm(f["correcta"])
        conteo[clave_c] = conteo.get(clave_c, 0) + 1
    aplicables = [f for f in filas if norm(f["cita_correcta"]) != "no_aplica"]
    n_cita_si = sum(1 for f in aplicables if norm(f["cita_correcta"]) == "si")
    lo, hi = wilson(n_si, n)
    clo, chi = wilson(n_cita_si, len(aplicables))
    return {
        "n": n,
        "n_si": n_si,
        "prop": (n_si / n) if n else None,
        "ic_lo": lo, "ic_hi": hi,
        "detalle": conteo,
        "n_cita_aplica": len(aplicables),
        "n_cita_si": n_cita_si,
        "prop_cita": (n_cita_si / len(aplicables)) if aplicables else None,
        "cita_lo": clo, "cita_hi": chi,
    }


def imprimir(titulo: str, grupos: List[Tuple[str, Dict]]) -> None:
    print()
    print(titulo)
    print("-" * 104)
    print("%-22s %5s %7s %8s %22s %9s %20s"
          % ("grupo", "n", "si", "prop", "Wilson 95%", "cita=si", "Wilson 95% (cita)"))
    for nombre, m in grupos:
        if m["n"] == 0:
            print("%-22s %5d  (sin filas: no se calcula)" % (nombre, m["n"]))
            continue
        ic = "[%.3f, %.3f]" % (m["ic_lo"], m["ic_hi"])
        if m["n_cita_aplica"] == 0:
            cita = "n/a (0 aplica)"
            ic_cita = "-"
        else:
            cita = "%d/%d = %.3f" % (m["n_cita_si"], m["n_cita_aplica"], m["prop_cita"])
            ic_cita = "[%.3f, %.3f]" % (m["cita_lo"], m["cita_hi"])
        print("%-22s %5d %7d %8.3f %22s %9s %20s"
              % (nombre, m["n"], m["n_si"], m["prop"], ic, cita, ic_cita))
    print("-" * 104)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hoja", default=None,
                        help="hoja_revision_ciega_<ts>.csv (por defecto: la mas reciente)")
    parser.add_argument("--clave", default=None,
                        help="clave_condicion_<ts>.csv (por defecto: la mas reciente)")
    args = parser.parse_args(argv)

    hoja_path = (Path(args.hoja) if args.hoja else
                 mas_reciente([str(RES / "hoja_revision_ciega_*.csv")]))
    clave_path = (Path(args.clave) if args.clave else
                  mas_reciente([str(RES / "clave_condicion_*.csv")]))
    if not hoja_path or not hoja_path.exists():
        print("ERROR: no se encontro la hoja de revision en %s" % RES)
        return 1
    if not clave_path or not clave_path.exists():
        print("ERROR: no se encontro la clave de condicion en %s" % RES)
        return 1

    hoja = leer(hoja_path)
    clave = leer(clave_path)
    print("hoja  : %s  (%d filas)" % (hoja_path.name, len(hoja)))
    print("clave : %s  (%d filas)" % (clave_path.name, len(clave)))

    # "tipo" no viene ni en la hoja ni en la clave: se recupera del banco por id.
    with open(BANCO, encoding="utf-8") as fh:
        tipo_por_id = {str(b["id"]): b["tipo"] for b in json.load(fh)}

    indice = {reg["fila"]: reg for reg in clave}
    cruzadas: List[Dict[str, str]] = []
    sin_clave = []
    for reg in hoja:
        c = indice.get(reg["fila"])
        if c is None:
            sin_clave.append(reg["fila"])
            continue
        cruzadas.append({
            "fila": reg["fila"],
            "id": c["id"],
            "condicion": c["condicion"],
            "tipo": tipo_por_id.get(c["id"], "?"),
            "correcta": (reg.get("correcta") or "").strip(),
            "cita_correcta": (reg.get("cita_correcta") or "").strip(),
            "observacion": (reg.get("observacion") or "").strip(),
        })
    if sin_clave:
        print("\nAVISO: %d filas de la hoja no tienen clave: %s"
              % (len(sin_clave), ", ".join(sin_clave)))
    if len(cruzadas) != len(hoja):
        print("AVISO: la hoja tiene %d filas y se cruzaron %d"
              % (len(hoja), len(cruzadas)))

    vacias_obs = sum(1 for c in cruzadas if not c["observacion"])
    if vacias_obs:
        print("AVISO: 'observacion' vacia en %d filas "
              "(texto libre: no bloquea el calculo)" % vacias_obs)

    problemas = validar(hoja)
    if problemas:
        print("\n" + "=" * 74)
        print("FALTAN CELDAS: no se calcula ninguna proporcion.")
        print("=" * 74)
        for pr in problemas[:60]:
            print("  - " + pr)
        if len(problemas) > 60:
            print("  ... y %d mas" % (len(problemas) - 60))
        print("\nCompleta correcta (si/no/parcial) y cita_correcta "
              "(si/no/no_aplica) en las filas senaladas y vuelve a ejecutar.")
        return 2

    por_cond: Dict[str, List[Dict[str, str]]] = {}
    por_tipo: Dict[str, List[Dict[str, str]]] = {}
    por_cruzado: Dict[str, List[Dict[str, str]]] = {}
    for c in cruzadas:
        por_cond.setdefault(c["condicion"], []).append(c)
        por_tipo.setdefault(c["tipo"], []).append(c)
        por_cruzado.setdefault("%s / %s" % (c["condicion"], c["tipo"]),
                               []).append(c)

    print("\n('tipo' se recupera del banco por id: %s)" % BANCO.name)

    imprimir("POR CONDICION",
             [(k, metricas(v)) for k, v in sorted(por_cond.items())])
    imprimir("POR TIPO",
             [(k, metricas(v)) for k, v in sorted(por_tipo.items())])
    imprimir("POR CONDICION x TIPO",
             [(k, metricas(v)) for k, v in sorted(por_cruzado.items())])
    print("\ncorrecta=si es el unico valor que cuenta como acierto; "
          "'parcial' queda fuera del numerador.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
