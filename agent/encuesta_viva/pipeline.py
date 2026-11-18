# -*- coding: utf-8 -*-
"""
Pipeline de la encuesta viva: respuestas crudas del formulario -> hallazgos.

Pasa a código lo que hacían a mano los notebooks:

    01_Preparacion_y_Diseno_Muestral.ipynb -> preparar()
    03_Estandarizacion_IRT.ipynb           -> estandarizar()
    agent/reports/entrega2.py              -> calcular() (ya existía)

Con las 1.813 respuestas actuales debe reproducir exactamente
data/01_dataset_preparado.xlsx y data/03_dataset_IRT_ponderado.xlsx; la
prueba tests/test_encuesta_viva.py lo verifica.

Nada de este módulo escribe en data/ ni en resultados/.
"""
import json
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

from agent.core.config import DIMENSION_ORDER, NEGATIVE_ITEMS
from agent.encuesta_viva.catalogos import (COLUMNAS_FORMULARIO, NORMALIZAR_PROGRAMA,
                                          POBLACION_ESTRATOS, PROGRAMA_A_NBC, TOTALES_GENERO)

# Ítem -> dimensión de Ryff (mismo mapa del notebook 01)
DIMENSION_DE_ITEM = {
    1: 'Autoaceptación', 7: 'Autoaceptación', 17: 'Autoaceptación', 24: 'Autoaceptación',
    2: 'Relaciones positivas', 8: 'Relaciones positivas', 12: 'Relaciones positivas',
    22: 'Relaciones positivas', 25: 'Relaciones positivas',
    3: 'Autonomía', 4: 'Autonomía', 9: 'Autonomía', 13: 'Autonomía',
    18: 'Autonomía', 23: 'Autonomía',
    5: 'Dominio del entorno', 10: 'Dominio del entorno', 14: 'Dominio del entorno',
    19: 'Dominio del entorno', 29: 'Dominio del entorno',
    6: 'Propósito de vida', 11: 'Propósito de vida', 15: 'Propósito de vida',
    16: 'Propósito de vida', 20: 'Propósito de vida',
    21: 'Crecimiento personal', 26: 'Crecimiento personal',
    27: 'Crecimiento personal', 28: 'Crecimiento personal',
}
DIM_ORDER = ['Autoaceptación', 'Crecimiento personal', 'Propósito de vida',
             'Dominio del entorno', 'Relaciones positivas', 'Autonomía']
assert set(DIM_ORDER) == set(DIMENSION_ORDER)

# Columnas del marco muestral de cada archivo (cada uno las nombra distinto)
COLUMNAS_MARCO = {
    'Bogotá': ('Bogotá', 'MODALIDAD', 'DIVISION', 'NOM_UNIDAD'),
    'Bucaramanga_Pre': ('Bucaramanga', 'Modalidad', 'División', 'Programa'),
    'Bucaramanga_Pos': ('Bucaramanga', 'Modalidad', 'Division', 'Programa'),
    'Tunja': ('Tunja', 'MODALIDAD', 'DIVISION', 'PROGRAMA'),
    'Villavicencio_Pre': ('Villavicencio', 'Modalidad', 'Decanatura de Division', 'Programa'),
    'Villavicencio_Pos': ('Villavicencio', None, 'Decanatura de Division', 'Programa'),
}

COLS_ID = ['ID', 'Género', 'Edad', 'Estrato socioeconómico', 'Nivel de formación',
           'Modalidad', 'Semestre', 'Sede o seccional',
           'Centro de Atención Universitaria CAU',
           'Programa de formación', 'Núcleo básico de conocimiento']
COLS_DISENO = ['seccional', 'modalidad_exp', 'factor_expansion',
               'peso_diseno', 'ajuste_no_respuesta',
               'N_poblacion_estrato', 'n_muestra_estrato', 'r_respondentes_estrato']
W = 'factor_expansion'


class FormularioInvalido(ValueError):
    """El formulario no tiene la estructura que espera el pipeline."""


# ── 0. Formulario nuevo -> esquema del original ──────────────────────
def _clave(texto) -> str:
    """Nombre de columna comparable: sin tildes, espacios raros ni mayúsculas."""
    t = unicodedata.normalize('NFKD', str(texto).replace(' ', ' '))
    t = ''.join(ch for ch in t if not unicodedata.combining(ch))
    return ' '.join(t.lower().split()).rstrip(' .:?')


def normalizar_formulario(df: pd.DataFrame) -> pd.DataFrame:
    """Acomoda las respuestas de un formulario nuevo al esquema del original.

    Empareja columnas por su texto (sin tildes ni espacios), descarta las que
    sobran (p. ej. «Marca temporal» de Google Forms) y crea vacías las de
    identificación que falten. Falla si falta una pregunta que el análisis usa.
    """
    disponibles = {_clave(c): c for c in df.columns}
    salida, faltan = {}, []
    opcionales = set(COLUMNAS_FORMULARIO[:6])  # ID, horas, correo, nombre
    for col in COLUMNAS_FORMULARIO:
        origen = disponibles.get(_clave(col))
        if origen is not None:
            salida[col] = df[origen].values
        elif col in opcionales or 'Programas de formación' in col or col.startswith(
                ('Centro de Atención', 'Ciudad de residencia', 'Departamento', 'País', 'Grupo')):
            salida[col] = np.nan
        else:
            faltan.append(col)
    if faltan:
        raise FormularioInvalido(
            f'Al formulario le faltan {len(faltan)} pregunta(s) del original, p. ej.: '
            + '; '.join(f[:60] for f in faltan[:3]))
    out = pd.DataFrame(salida)
    if out['ID'].isna().all():
        out['ID'] = np.arange(1, len(out) + 1)
    items = COLUMNAS_FORMULARIO[-29:]
    out[items] = out[items].apply(pd.to_numeric, errors='coerce')
    for col in ('Edad', 'Estrato socioeconómico', 'Semestre'):
        out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


# ── 1. Preparación (notebook 01) ─────────────────────────────────────
def _consolidar_programa(raw: pd.DataFrame) -> pd.DataFrame:
    prog_cols = [c for c in raw.columns if 'Programas de formación' in c]
    if not prog_cols:
        raise FormularioInvalido("No hay columnas 'Programas de formación ...'.")
    df = raw.copy()
    df['programa_raw'] = df[prog_cols].bfill(axis=1).iloc[:, 0]
    df['Programa de formación'] = df['programa_raw'].map(
        lambda x: NORMALIZAR_PROGRAMA.get(x, x) if pd.notna(x) else x)
    df['Núcleo básico de conocimiento'] = df['Programa de formación'].map(PROGRAMA_A_NBC)

    df = df.drop(columns=prog_cols + ['programa_raw'])
    cols = list(df.columns)
    pos = cols.index('Centro de Atención Universitaria CAU') + 1
    cols.remove('Programa de formación')
    cols.remove('Núcleo básico de conocimiento')
    cols.insert(pos, 'Programa de formación')
    cols.insert(pos + 1, 'Núcleo básico de conocimiento')
    return df[cols]


def programas_sin_nbc(raw: pd.DataFrame) -> list:
    """Programas que el catálogo no conoce (hay que agregarlos a catalogos.py)."""
    df = _consolidar_programa(raw)
    falta = df['Programa de formación'].notna() & df['Núcleo básico de conocimiento'].isna()
    return sorted(df.loc[falta, 'Programa de formación'].unique())


def columnas_items(df: pd.DataFrame) -> list:
    items = list(df.columns[df.columns.get_loc('Núcleo básico de conocimiento') + 1:])
    return items[:29]


def poblacion_por_estrato(marcos: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """N y n diseñados por seccional x modalidad, desde los 6 marcos muestrales.

    Sin marcos usa POBLACION_ESTRATOS (los mismos totales, ya agregados).
    """
    if marcos is None:
        return pd.DataFrame(POBLACION_ESTRATOS,
                            columns=['seccional', 'modalidad', 'N_total', 'n_total'])
    partes = []
    for clave, (seccional, col_mod, col_div, col_prog) in COLUMNAS_MARCO.items():
        m = marcos[clave]
        partes.append(pd.DataFrame({
            'seccional': seccional,
            'modalidad': m[col_mod] if col_mod else 'Posgrado',
            'division': m[col_div],
            'programa': m[col_prog],
            'estrato': m['Estrato'],
            'peso_muestral': m['peso_muestral'],
        }))
    muestra = pd.concat(partes, ignore_index=True)
    finos = muestra.groupby(['seccional', 'modalidad', 'estrato']).agg(
        n_h=('peso_muestral', 'count'), w_h=('peso_muestral', 'first')).reset_index()
    finos['N_h'] = finos['n_h'] * finos['w_h']
    return finos.groupby(['seccional', 'modalidad']).agg(
        N_total=('N_h', 'sum'), n_total=('n_h', 'sum')).reset_index()


def _seccional(s):
    if pd.isna(s):
        return None
    if 'Bogotá' in s or 'Medellín' in s or 'CAU' in s:
        return 'Bogotá'
    for nombre in ('Bucaramanga', 'Tunja', 'Villavicencio'):
        if nombre in s:
            return nombre
    return None


def _modalidad(n):
    if pd.isna(n):
        return None
    if n in ['Pregrado', 'Técnico Profesional', 'Técnico Laboral', 'Tecnológico']:
        return 'Pregrado'
    return 'Posgrado'


def preparar(raw: pd.DataFrame, marcos: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """Depura, invierte ítems, calcula dimensiones y factores de expansión."""
    df = _consolidar_programa(raw)
    items = columnas_items(df)
    if len(list(df.columns[df.columns.get_loc('Núcleo básico de conocimiento') + 1:])) != 29:
        raise FormularioInvalido('Se esperaban 29 ítems después de "Núcleo básico de conocimiento".')

    ok = ((df['Consentimiento informado'] == 'Acepto')
          & (df['Autorización expresa para el tratamiento de datos personales'] == 'Acepto')
          & (df['¿Soy mayor de edad?'] == 'Sí'))
    dv = df[ok].copy()

    for i in NEGATIVE_ITEMS:
        dv[items[i - 1]] = 7 - dv[items[i - 1]]

    dv['Métrica Global'] = dv[items].sum(axis=1)
    for dim in DIM_ORDER:
        cols = [items[i - 1] for i in sorted(k for k, d in DIMENSION_DE_ITEM.items() if d == dim)]
        dv[f'Dim_{dim}'] = dv[cols].mean(axis=1)

    dv['seccional'] = dv['Sede o seccional'].apply(_seccional)
    dv['modalidad_exp'] = dv['Nivel de formación'].apply(_modalidad)

    resp = dv.groupby(['seccional', 'modalidad_exp']).size().reset_index(name='r_total')
    resp.columns = ['seccional', 'modalidad', 'r_total']
    f = poblacion_por_estrato(marcos).merge(resp, on=['seccional', 'modalidad'])
    f['peso_diseno'] = f['N_total'] / f['n_total']
    f['ajuste_no_respuesta'] = f['n_total'] / f['r_total']
    f['factor_expansion'] = f['N_total'] / f['r_total']
    f = f.rename(columns={'modalidad': 'modalidad_exp', 'N_total': 'N_poblacion_estrato',
                          'n_total': 'n_muestra_estrato', 'r_total': 'r_respondentes_estrato'})
    nuevas = ['factor_expansion', 'peso_diseno', 'ajuste_no_respuesta',
              'N_poblacion_estrato', 'n_muestra_estrato', 'r_respondentes_estrato']
    dv = dv.reset_index().merge(f[['seccional', 'modalidad_exp'] + nuevas],
                                on=['seccional', 'modalidad_exp'], how='left')
    dv = dv.set_index('index').rename_axis(None)
    return calibrar_genero(dv)


def calibrar_genero(dv: pd.DataFrame, tol: float = 1e-12, max_iter: int = 5000) -> pd.DataFrame:
    """Raking del factor de expansión a dos márgenes.

    1. Estrato (seccional x modalidad): total N_h del marco muestral.
    2. Seccional x género (Masculino / Femenino / Otro): TOTALES_GENERO.

    Es el paso que no estaba en los notebooks: se reconstruyó desde
    data/01_dataset_preparado.xlsx (reproduce sus pesos con error < 1e-12).
    El peso sin calibrar queda en factor_expansion_original.
    """
    dv = dv.copy()
    dv['genero_bin'] = dv['Género'].where(dv['Género'].isin(['Masculino', 'Femenino']), 'Otro')
    dv['strata_key'] = dv['seccional'] + '_' + dv['modalidad_exp']
    dv['gender_key'] = dv['seccional'] + '_' + dv['genero_bin']

    meta_estrato = dv.groupby('strata_key')['N_poblacion_estrato'].first()
    meta_genero = pd.Series({f'{s}_{g}': v for (s, g), v in TOTALES_GENERO.items()})
    # Si en una seccional nadie respondió con algún género, ese margen no se
    # puede alcanzar: se reparte su total entre los géneros presentes.
    presentes = set(dv['gender_key'])
    for sec in dv['seccional'].unique():
        claves = [k for k in meta_genero.index if k.startswith(f'{sec}_')]
        faltan = [k for k in claves if k not in presentes]
        if faltan:
            hay = [k for k in claves if k in presentes]
            extra = meta_genero[faltan].sum()
            meta_genero[hay] *= 1 + extra / meta_genero[hay].sum()
            meta_genero = meta_genero.drop(faltan)

    w = dv[W].to_numpy(dtype=float).copy()
    est, gen = dv['strata_key'].to_numpy(), dv['gender_key'].to_numpy()
    obj_e, obj_g = dv['strata_key'].map(meta_estrato).to_numpy(), dv['gender_key'].map(meta_genero).to_numpy()
    for _ in range(max_iter):
        w *= obj_e / pd.Series(w).groupby(est).transform('sum').to_numpy()
        w *= obj_g / pd.Series(w).groupby(gen).transform('sum').to_numpy()
        if np.max(np.abs(pd.Series(w).groupby(est).transform('sum').to_numpy() / obj_e - 1)) < tol:
            break

    dv['peso_calibrado'] = w
    dv['factor_expansion_original'] = dv[W]
    dv[W] = w
    return dv


# ── 2. Estandarización IRT (notebook 03) ─────────────────────────────
def a_t(theta: np.ndarray) -> np.ndarray:
    """T = 50 + 10·θ.

    Es la conversión que produjo data/03_dataset_IRT_ponderado.xlsx (la
    reproduce exactamente). El notebook 03 describe una normalización
    ponderada, T = 50 + 10 (θ − μ_w) / σ_w, pero el archivo no la aplica:
    con ella la media ponderada sería exactamente 50, y la reportada es 50,7.
    θ viene del GRM con prior N(0, 1), así que la escala ya está centrada en
    la muestra sin ponderar.
    """
    return 50 + 10 * np.asarray(theta)


def nivel(t: float) -> str:
    if t < 30:
        return 'Muy bajo'
    if t < 40:
        return 'Bajo'
    if t < 60:
        return 'Medio'
    if t < 70:
        return 'Alto'
    return 'Muy alto'


PARAMETROS_GRM = Path(__file__).resolve().parent / 'parametros_grm.json'


def _items_de(prep: pd.DataFrame) -> list:
    return list(prep.columns[prep.columns.get_loc('Núcleo básico de conocimiento') + 1:
                             prep.columns.get_loc('Métrica Global')])


def _cols_escala(items: list, escala: str) -> list:
    if escala == 'Global':
        return items
    return [items[i - 1] for i in sorted(k for k, d in DIMENSION_DE_ITEM.items() if d == escala)]


def ajustar_parametros(prep: pd.DataFrame) -> dict:
    """Ajusta los 7 GRM (global + 6 dimensiones) y devuelve sus parámetros."""
    import girth
    items = _items_de(prep)
    completos = prep[items].dropna()
    out = {}
    for esc in ['Global'] + DIM_ORDER:
        r = girth.grm_mml(completos[_cols_escala(items, esc)].values.astype(int).T)
        lp = r['LatentPDF']
        out[esc] = {'a': r['Discrimination'].tolist(), 'b': r['Difficulty'].tolist(),
                    'theta': lp.quadrature_locations.tolist(),
                    'peso': (lp.null_distribution * lp.weights).tolist()}
    return out


def theta_eap(respuestas: np.ndarray, par: dict) -> np.ndarray:
    """θ EAP con parámetros fijos del GRM (mismo cálculo que girth).

    respuestas: [personas x ítems] con valores 1..6; un NaN no aporta.
    """
    a, b = np.asarray(par['a']), np.asarray(par['b'])
    th, w = np.asarray(par['theta']), np.asarray(par['peso'])
    L = np.ones((respuestas.shape[0], th.size))
    for j in range(respuestas.shape[1]):
        acum = np.vstack([np.ones_like(th)] + [expit(a[j] * (th - bk)) for bk in b[j]]
                         + [np.zeros_like(th)])
        prob = acum[:-1] - acum[1:]                    # [categoría x θ]
        x = respuestas[:, j]
        ok = ~np.isnan(x)
        L[ok] *= prob[x[ok].astype(int) - 1]
    L *= w
    return (L * th).sum(axis=1) / L.sum(axis=1)


def estandarizar(prep: pd.DataFrame, parametros: Optional[dict] = None) -> pd.DataFrame:
    """Calcula θ y T-scores global y por dimensión.

    Con `parametros` (por defecto los del estudio, parametros_grm.json) cada
    persona se califica con los parámetros fijos: el T de la encuesta viva
    queda en la misma escala del estudio y es comparable. Con
    parametros={} se reestima el GRM con los datos recibidos (notebook 03).
    """
    if parametros is None:
        parametros = json.loads(PARAMETROS_GRM.read_text(encoding='utf-8'))
    if not parametros:
        parametros = ajustar_parametros(prep)

    items = _items_de(prep)
    validos = prep[items].apply(lambda c: c.where(c.isin([1, 2, 3, 4, 5, 6])))
    completos = validos.dropna()
    s = prep.loc[completos.index].copy()
    for esc in ['Global'] + DIM_ORDER:
        theta = theta_eap(completos[_cols_escala(items, esc)].to_numpy(float), parametros[esc])
        s[f'θ_{esc}'] = theta
        s[f'T_{esc}'] = a_t(theta)
        s[f'Nivel_{esc}'] = s[f'T_{esc}'].apply(nivel)   # antes de redondear
    s['Suma_Bruta'] = completos.sum(axis=1).values

    puntajes = (['Suma_Bruta', 'θ_Global', 'T_Global', 'Nivel_Global']
                + [c for d in DIM_ORDER for c in (f'θ_{d}', f'T_{d}', f'Nivel_{d}')])
    cols = ([c for c in COLS_ID if c in s.columns]
            + [c for c in COLS_DISENO if c in s.columns] + puntajes)
    out = s[cols].copy()
    for c in out.columns:
        if c.startswith('θ_'):
            out[c] = out[c].round(4)
        elif c.startswith('T_'):
            out[c] = out[c].round(1)
    return out.reset_index(drop=True)


# ── 3. Todo junto ────────────────────────────────────────────────────
def correr(raw: pd.DataFrame, marcos: Optional[Dict[str, pd.DataFrame]] = None,
           incluir_dif: bool = True, parametros: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Respuestas crudas -> (preparado, IRT, hallazgos para la app)."""
    from agent.reports import entrega2
    prep = preparar(raw, marcos)
    irt = estandarizar(prep, parametros)
    return prep, irt, entrega2.calcular(prep.reset_index(drop=True), irt, incluir_dif=incluir_dif)
