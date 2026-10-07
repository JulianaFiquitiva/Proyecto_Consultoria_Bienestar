#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convierte la hoja de revision ciega (CSV) a .xlsx para revision a mano.

Lee results/hoja_revision_ciega_*.csv (la mas reciente, o --csv) con el
modulo csv de Python, que respeta los saltos de linea dentro de las comillas.
NO abre ni lee clave_condicion_*.csv.

Genera tres archivos IDENTICOS en resultados/:
    hoja_revision_ciega_<ts>.xlsx
    hoja_Luisa.xlsx
    hoja_Juliana.xlsx

Hoja "Revision": 11 columnas, texto ajustado, encabezado y columna "fila"
fijos (panel B2), listas desplegables en "correcta" (si,no,parcial) y
"cita_correcta" (si,no,no_aplica), y relleno amarillo en correcta,
cita_correcta y observacion.

Comprobacion final: cada .xlsx con 40 filas de datos y columna "fila"
valores 1..40 sin repetirse.

Uso:
    python tests/hoja_a_xlsx.py
    python tests/hoja_a_xlsx.py --csv resultados/hoja_revision_ciega_X.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"

COLUMNAS = [
    "fila", "pregunta", "respuesta_esperada", "documento_esperado",
    "pagina_pdf_esperada", "cita_textual_esperada", "respuesta",
    "fuentes_citadas", "correcta", "cita_correcta", "observacion",
]

# Columnas que el revisor llena: relleno amarillo.
COLUMNAS_REVISION = ("correcta", "cita_correcta", "observacion")

OPCIONES_CORRECTA = ("si", "no", "parcial")
OPCIONES_CITA = ("si", "no", "no_aplica")

ANCHOS = {
    "fila": 6, "pregunta": 58, "respuesta_esperada": 52,
    "documento_esperado": 46, "pagina_pdf_esperada": 16,
    "cita_textual_esperada": 58, "respuesta": 58, "fuentes_citadas": 42,
    "correcta": 11, "cita_correcta": 14, "observacion": 34,
}

AMARILLO = "FFFFFF00"
GRIS_ENCABEZADO = "FFD9D9D9"

LIMITE_CELDA = 32767          # Excel rechaza celdas mas largas
_RE_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def hoja_mas_reciente() -> Path:
    candidatos = sorted(RES.glob("hoja_revision_ciega_*.csv"),
                        key=lambda p: p.stat().st_mtime)
    if not candidatos:
        raise SystemExit("No hay hoja_revision_ciega_*.csv en %s" % RES)
    return candidatos[-1]


def leer_hoja(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        lector = csv.DictReader(fh)
        faltantes = [c for c in COLUMNAS if c not in (lector.fieldnames or [])]
        if faltantes:
            raise SystemExit("Faltan columnas en el CSV: %s" % faltantes)
        filas = [dict(f) for f in lector]
    if not filas:
        raise SystemExit("La hoja CSV esta vacia: %s" % path)
    return filas


def limpiar(valor) -> str:
    """Texto seguro para Excel: sin caracteres XML ilegales ni > 32767."""
    texto = _RE_ILLEGAL.sub("", "" if valor is None else str(valor))
    if len(texto) > LIMITE_CELDA:
        texto = texto[:LIMITE_CELDA - 1] + "…"
    return texto


def construir(filas: List[Dict[str, str]]) -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.title = "Revision"

    delgado = Side(style="thin", color="FFBFBFBF")
    borde = Border(left=delgado, right=delgado, top=delgado, bottom=delgado)
    fill_revision = PatternFill(start_color=AMARILLO, end_color=AMARILLO,
                                fill_type="solid")
    alineacion = Alignment(wrap_text=True, vertical="top")
    alineacion_centro = Alignment(wrap_text=True, vertical="top",
                                  horizontal="center")

    # Encabezado
    for col, nombre in enumerate(COLUMNAS, 1):
        celda = ws.cell(row=1, column=col, value=nombre)
        celda.font = Font(bold=True)
        celda.fill = PatternFill(start_color=GRIS_ENCABEZADO,
                                 end_color=GRIS_ENCABEZADO, fill_type="solid")
        celda.alignment = alineacion_centro
        celda.border = borde

    # Datos
    for i, fila in enumerate(filas, start=2):
        for col, nombre in enumerate(COLUMNAS, 1):
            valor = limpiar(fila.get(nombre, ""))
            if nombre == "fila":
                try:
                    valor = int(valor)          # numerica: facilita la comprobacion
                except (TypeError, ValueError):
                    pass
            celda = ws.cell(row=i, column=col, value=valor)
            celda.alignment = alineacion_centro if nombre in (
                "fila", "pagina_pdf_esperada", "correcta",
                "cita_correcta") else alineacion
            celda.border = borde
            if nombre in COLUMNAS_REVISION:
                celda.fill = fill_revision

    ultima = len(filas) + 1

    # Listas desplegables sobre la columna completa (con holgura)
    dv_correcta = DataValidation(
        type="list", formula1='"%s"' % ",".join(OPCIONES_CORRECTA),
        allow_blank=True, showErrorMessage=True,
        errorTitle="Valor no valido",
        error="Use: %s" % ", ".join(OPCIONES_CORRECTA))
    dv_cita = DataValidation(
        type="list", formula1='"%s"' % ",".join(OPCIONES_CITA),
        allow_blank=True, showErrorMessage=True,
        errorTitle="Valor no valido",
        error="Use: %s" % ", ".join(OPCIONES_CITA))
    ws.add_data_validation(dv_correcta)
    ws.add_data_validation(dv_cita)
    col_correcta = get_column_letter(COLUMNAS.index("correcta") + 1)
    col_cita = get_column_letter(COLUMNAS.index("cita_correcta") + 1)
    dv_correcta.add("%s2:%s1048576" % (col_correcta, col_correcta))
    dv_cita.add("%s2:%s1048576" % (col_cita, col_cita))

    # Relleno amarillo tambien en columnas aun no rellenadas (por si se
    # pegan filas nuevas: la regla se aplica a toda la columna).
    for nombre in COLUMNAS_REVISION:
        letra = get_column_letter(COLUMNAS.index(nombre) + 1)
        ws.conditional_formatting.add(
            "%s2:%s1048576" % (letra, letra),
            FormulaRule(formula=['LEN(%s2)>0' % letra],
                        fill=PatternFill(start_color=AMARILLO,
                                         end_color=AMARILLO,
                                         fill_type="solid"),
                        stopIfTrue=False))

    # Anchos, panel fijo (fila 1 + columna A) y filtros
    for nombre, ancho in ANCHOS.items():
        ws.column_dimensions[
            get_column_letter(COLUMNAS.index(nombre) + 1)].width = ancho
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = "A1:%s%d" % (get_column_letter(len(COLUMNAS)),
                                      ultima)
    ws.row_dimensions[1].height = 30
    return wb


def comprobar(path: Path, n_esperado: int) -> Tuple[bool, str]:
    wb = load_workbook(path, read_only=False, data_only=True)
    if "Revision" not in wb.sheetnames:
        return False, "%s: no existe la hoja 'Revision'" % path.name
    ws = wb["Revision"]
    encabezados = [ws.cell(row=1, column=c).value
                   for c in range(1, len(COLUMNAS) + 1)]
    if encabezados != COLUMNAS:
        return False, "%s: encabezados != %s" % (path.name, encabezados)

    datos = []
    fila = 2
    while True:
        v = ws.cell(row=fila, column=1).value
        if v is None and all(ws.cell(row=fila, column=c).value in (None, "")
                             for c in range(1, len(COLUMNAS) + 1)):
            break
        datos.append(v)
        fila += 1
        if fila > 100000:
            break

    problemas = []
    n_dv = len(ws.data_validations.dataValidation)
    if n_dv < 2:
        problemas.append("faltan las listas desplegables (%d validaciones)"
                         % n_dv)
    if len(datos) != n_esperado:
        problemas.append("%d filas de datos (se esperaban %d)"
                         % (len(datos), n_esperado))
    try:
        enteros = [int(v) for v in datos]
    except (TypeError, ValueError):
        enteros = []
        problemas.append("la columna 'fila' tiene valores no enteros: %r"
                         % datos[:5])
    if enteros:
        if len(set(enteros)) != len(enteros):
            problemas.append("la columna 'fila' repite valores")
        if enteros != list(range(1, n_esperado + 1)):
            problemas.append("la columna 'fila' no va de 1 a %d: %s...%s"
                             % (n_esperado, enteros[:3], enteros[-3:]))

    wb.close()
    if problemas:
        return False, "%s: %s" % (path.name, "; ".join(problemas))
    return True, ("%s: hoja Revision, %d filas de datos, fila 1..%d sin "
                  "repetir, %d validaciones de lista"
                  % (path.name, len(datos), n_esperado, n_dv))


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV de origen (por defecto: el mas reciente)")
    parser.add_argument("--destino", type=Path, default=RES,
                        help="directorio de salida (por defecto: resultados/)")
    args = parser.parse_args(argv)

    origen = Path(args.csv) if args.csv else hoja_mas_reciente()
    if not origen.exists():
        raise SystemExit("No existe %s" % origen)
    ts = origen.stem.replace("hoja_revision_ciega_", "", 1)
    filas = leer_hoja(origen)
    n = len(filas)
    print("origen : %s  (%d filas de datos)" % (origen.name, n))

    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    salidas = [
        destino / ("hoja_revision_ciega_%s.xlsx" % ts),
        destino / "hoja_Luisa.xlsx",
        destino / "hoja_Juliana.xlsx",
    ]

    # Un solo Workbook guardado en tres rutas: identicos byte a byte.
    wb = construir(filas)
    wb.save(salidas[0])
    base = salidas[0].read_bytes()
    for salida in salidas[1:]:
        salida.write_bytes(base)

    ok = True
    for salida in salidas:
        bueno, mensaje = comprobar(salida, n)
        ok = ok and bueno
        print(("OK   " if bueno else "FALLA") + " " + mensaje)
        if not bueno:
            print("      -> %s" % salida)

    print("-" * 74)
    for salida in salidas:
        print("  %s  (%d bytes)" % (salida, salida.stat().st_size))
    if not ok:
        print("LA COMPROBACION HA FALLADO")
        return 1
    print("comprobacion: %d filas de datos y columna fila 1..%d sin repetir "
          "en los 3 archivos" % (n, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
