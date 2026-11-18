# -*- coding: utf-8 -*-
"""
Especificación del formulario de la encuesta viva y generadores.

    python -m agent.encuesta_viva.formulario

Genera, a partir de los catálogos (los textos exactos del formulario
original):
- docs/encuesta_viva/crear_formulario.gs: Google Apps Script que crea el
  formulario en Google Forms con su hoja de respuestas, secciones y saltos.
- docs/encuesta_viva/FORMULARIO.md: todas las preguntas, tipo y opciones.

Los títulos de las preguntas deben ser exactamente las columnas de
COLUMNAS_FORMULARIO: así la hoja de Google queda con las columnas que espera
pipeline.normalizar_formulario.
"""
import json
from pathlib import Path

from agent.core.config import PROJECT_ROOT
from agent.encuesta_viva.catalogos import CAU, COLUMNAS_FORMULARIO, PROGRAMAS_POR_SEDE

DESTINO = PROJECT_ROOT / "docs" / "encuesta_viva"
ITEMS = COLUMNAS_FORMULARIO[-29:]
ESCALA = ("Totalmente en desacuerdo", "Totalmente de acuerdo")

# Sede -> columna de programas a la que salta el formulario
SEDES = {
    "Sede Principal Bogotá": "Programas de formación Sede Principal Bogotá y centros de "
                             "atención universitaria CAU",
    "Centro de Atención Universitaria CAU": "CAU",
    "Campus Medellín": "Programas de formación\xa0Campus Medellín",
    "Seccional Bucaramanga": "Programas de formación\xa0Seccional Bucaramanga",
    "Seccional Tunja": "Programas de formación\xa0Seccional Tunja",
    "Seccional Villavicencio": "Programas de formación\xa0Seccional Villavicencio",
}

# (título, tipo, opciones, obligatoria, ayuda). Tipos: unica, lista, numero, texto
DATOS_PERSONALES = [
    ("Género", "unica", ["Femenino", "Masculino", "No binario", "Prefiero no decirlo"], True,
     ""),   # + opción «Otro» con texto libre (showOtherOption)
    ("Edad", "numero", None, True, "Número entero, en años."),
    ("País de residencia", "texto", None, False, ""),
    ("Departamento de residencia", "texto", None, False, ""),
    ("Ciudad de residencia en Colombia", "texto", None, False, ""),
    ("Estrato socioeconómico", "unica", ["1", "2", "3", "4", "5", "6"], True, ""),
    ("Ciudad de residencia fuera de Colombia", "texto", None, False,
     "Solo si vive fuera de Colombia."),
    ("Grupo poblacional", "unica", ["Ninguno", "Afrodescendiente", "Indígena", "Raizal",
                                    "ROM - Gitano", "Población LGBTIQ+",
                                    "Víctima del conflicto armado",
                                    "Ciudadano extranjero (no colombiano)"], True, ""),
    ("Nivel de formación", "unica", ["Pregrado", "Técnico Profesional", "Técnico Laboral",
                                     "Tecnológico", "Especialización", "Maestría",
                                     "Doctorado", "Posdoctorado"], True, ""),
    ("Modalidad", "unica", ["Presencial", "Virtual", "Híbrido (presencial - virtual)",
                            "A distancia"], True, ""),
    ("Semestre", "unica", [str(i) for i in range(1, 11)], True, ""),
]
assert all(t in COLUMNAS_FORMULARIO for t, *_ in DATOS_PERSONALES)


def _js(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def apps_script() -> str:
    """Google Apps Script que crea el formulario completo."""
    datos = ",\n    ".join(
        _js({"t": t, "tipo": tipo, "op": op, "req": req, "ayuda": ayuda})
        for t, tipo, op, req, ayuda in DATOS_PERSONALES)
    programas = _js({SEDES[s]: PROGRAMAS_POR_SEDE[SEDES[s]]
                     for s in SEDES if SEDES[s] != "CAU"})
    return f"""/**
 * Crea el formulario de la Escala de Bienestar (encuesta en vivo) en Google
 * Forms, con su hoja de respuestas. Generado por
 * agent/encuesta_viva/formulario.py: no editar a mano los títulos (deben ser
 * idénticos al formulario original para que el análisis los reconozca).
 *
 * Uso: script.google.com > Proyecto nuevo > pegar este archivo > Ejecutar
 * «crearFormulario» > autorizar. Los enlaces salen en el registro.
 */
function crearFormulario() {{
  var form = FormApp.create('Formulario de la escala de bienestar subjetivo');
  form.setDescription('Universidad Santo Tomás. Estudio de bienestar de la población ' +
    'estudiantil. Sus respuestas son anónimas y se analizan solo de forma agregada.');
  form.setCollectEmail(false);
  form.setProgressBar(true);

  function unica(titulo, opciones, req, ayuda, otro) {{
    var it = form.addMultipleChoiceItem().setTitle(titulo).setRequired(req);
    it.setChoiceValues(opciones);
    if (otro) it.showOtherOption(true);
    if (ayuda) it.setHelpText(ayuda);
    return it;
  }}
  function seccion(titulo) {{ return form.addPageBreakItem().setTitle(titulo); }}

  // 1. Consentimiento (si no acepta, el formulario termina)
  var consentimiento = unica('Consentimiento informado', ['Acepto', 'No acepto'], true,
    'PEGAR AQUÍ EL TEXTO DEL CONSENTIMIENTO INFORMADO DEL FORMULARIO ORIGINAL.');
  seccion('Tratamiento de datos');
  var autorizacion = unica('Autorización expresa para el tratamiento de datos personales',
    ['Acepto', 'No acepto'], true,
    'PEGAR AQUÍ EL TEXTO DE AUTORIZACIÓN DEL FORMULARIO ORIGINAL.');
  seccion('Edad');
  var mayor = unica('¿Soy mayor de edad?', ['Sí', 'No'], true);

  // 2. Datos generales
  seccion('Información general');
  var datos = [
    {datos}
  ];
  datos.forEach(function (d) {{
    if (d.tipo === 'unica') {{
      unica(d.t, d.op, d.req, d.ayuda, d.t === 'Género');
    }} else if (d.tipo === 'numero') {{
      form.addTextItem().setTitle(d.t).setRequired(d.req).setHelpText(d.ayuda)
        .setValidation(FormApp.createTextValidation().requireNumberBetween(14, 99).build());
    }} else {{
      var it = form.addTextItem().setTitle(d.t).setRequired(d.req);
      if (d.ayuda) it.setHelpText(d.ayuda);
    }}
  }});
  var sede = unica('Sede o seccional', {_js(list(SEDES))}, true);

  // 3. Programa según la sede
  var pCau = seccion('Centro de Atención Universitaria');
  form.addListItem().setTitle('Centro de Atención Universitaria CAU').setRequired(true)
    .setChoiceValues({_js(CAU)});
  var programas = {programas};
  var secciones = {{}};
  Object.keys(programas).forEach(function (col) {{
    secciones[col] = seccion('Programa de formación');
    form.addListItem().setTitle(col).setRequired(true).setChoiceValues(programas[col]);
  }});

  // 4. Escala de bienestar (29 ítems, 1 a 6)
  var pEscala = seccion('Escala de bienestar');
  pEscala.setHelpText('Indique qué tan de acuerdo está con cada afirmación: ' +
    '1 = {ESCALA[0]}, 6 = {ESCALA[1]}.');
  {_js(ITEMS)}.forEach(function (t) {{
    form.addScaleItem().setTitle(t).setBounds(1, 6)
      .setLabels({_js(ESCALA[0])}, {_js(ESCALA[1])}).setRequired(true);
  }});

  // Saltos: quien no acepta termina; cada sede va a su lista de programas;
  // después de la lista de programas, a la escala.
  [[consentimiento, 'Acepto', 'No acepto'], [autorizacion, 'Acepto', 'No acepto'],
   [mayor, 'Sí', 'No']].forEach(function (x) {{
    x[0].setChoices([x[0].createChoice(x[1], FormApp.PageNavigationType.CONTINUE),
                     x[0].createChoice(x[2], FormApp.PageNavigationType.SUBMIT)]);
  }});
  var cols = Object.keys(programas);
  var destino = {{'Centro de Atención Universitaria CAU': pCau}};
  var sedeACol = {_js({s: SEDES[s] for s in SEDES if SEDES[s] != "CAU"})};
  Object.keys(sedeACol).forEach(function (s) {{ destino[s] = secciones[sedeACol[s]]; }});
  sede.setChoices({_js(list(SEDES))}.map(function (s) {{
    return sede.createChoice(s, destino[s]);
  }}));
  // El CAU sigue con la lista de Bogotá (la siguiente sección); las demás
  // listas de programas saltan a la escala.
  for (var i = 1; i < cols.length; i++) secciones[cols[i]].setGoToPage(pEscala);

  var hoja = SpreadsheetApp.create('Respuestas - Escala de bienestar (encuesta en vivo)');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, hoja.getId());
  Logger.log('Formulario (para responder): ' + form.getPublishedUrl());
  Logger.log('Formulario (para editar):    ' + form.getEditUrl());
  Logger.log('Hoja de respuestas:          ' + hoja.getUrl());
}}
"""


def guia() -> str:
    """Todas las preguntas, con tipo y opciones, en markdown."""
    L = ["# Formulario de la encuesta en vivo", "",
         "Generado por `python -m agent.encuesta_viva.formulario`. Los títulos deben ser "
         "**idénticos** a estos (el análisis reconoce las columnas por su texto; tildes, "
         "mayúsculas y el punto final no importan, pero cualquier otra diferencia sí).", "",
         "**Forma rápida:** pegar `crear_formulario.gs` en script.google.com y ejecutarlo: "
         "crea el formulario completo con su hoja de respuestas.", "",
         "**Reglas si se arma a mano:**",
         "- Los 29 ítems van como **29 preguntas separadas de «Escala lineal» de 1 a 6**. "
         "No usar «Cuadrícula de varias opciones»: Google la exporta con otros nombres de "
         "columna.",
         "- No pedir correo ni nombre.",
         "- Cada lista de programas es una pregunta aparte en su propia sección; la "
         "pregunta «Sede o seccional» salta a la sección que corresponde.", "",
         "## Sección 1. Consentimiento", "",
         "| # | Pregunta (título exacto) | Tipo | Opciones | Obligatoria |",
         "|---|---|---|---|---|",
         "| 1 | Consentimiento informado | Opción múltiple | Acepto · No acepto (→ enviar) | Sí |",
         "| 2 | Autorización expresa para el tratamiento de datos personales | Opción múltiple "
         "| Acepto · No acepto (→ enviar) | Sí |",
         "| 3 | ¿Soy mayor de edad? | Opción múltiple | Sí · No (→ enviar) | Sí |", "",
         "El texto completo del consentimiento y de la autorización va en la descripción de "
         "cada pregunta: copiarlo del formulario original.", "",
         "## Sección 2. Información general", "",
         "| # | Pregunta (título exacto) | Tipo | Opciones | Obligatoria |",
         "|---|---|---|---|---|"]
    tipos = {"unica": "Opción múltiple", "numero": "Respuesta corta (número)",
             "texto": "Respuesta corta", "lista": "Desplegable"}
    n = 4
    for t, tipo, op, req, ayuda in DATOS_PERSONALES:
        ops = " · ".join(op) if op else "—"
        if t == "Género":
            ops += " · Otro (texto libre)"
        if t == "Edad":
            ops = "entre 14 y 99"
        L.append(f"| {n} | {t} | {tipos[tipo]} | {ops} | {'Sí' if req else 'No'} |")
        n += 1
    L.append(f"| {n} | Sede o seccional | Opción múltiple (salta a la sección de su sede) | "
             + " · ".join(SEDES) + " | Sí |")
    L += ["", "## Sección 3. Programa (según la sede)", "",
          "Quien elige *Centro de Atención Universitaria CAU* responde primero el centro y "
          "luego la lista de Bogotá (la pregunta de Bogotá incluye los CAU).", "",
          "**Centro de Atención Universitaria CAU** (desplegable): " + " · ".join(CAU), ""]
    for s, col in SEDES.items():
        if col == "CAU":
            continue
        L += [f"**{col.replace(chr(160), ' ')}** (desplegable, {len(PROGRAMAS_POR_SEDE[col])} "
              "opciones):", "", "\n".join(f"- {p}" for p in PROGRAMAS_POR_SEDE[col]), ""]
    L += ["## Sección 4. Escala de bienestar (Ryff, 29 ítems)", "",
          f"Todas: **Escala lineal de 1 a 6**, 1 = {ESCALA[0]}, 6 = {ESCALA[1]}, obligatorias. "
          "Los ítems con * están redactados en negativo (el análisis los invierte; en el "
          "formulario se preguntan igual).", "",
          "| # | Pregunta (título exacto) |", "|---|---|"]
    from agent.core.config import NEGATIVE_ITEMS
    for i, t in enumerate(ITEMS, 1):
        L.append(f"| {i} | {t}{' *' if i in NEGATIVE_ITEMS else ''} |")
    return "\n".join(L) + "\n"


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    (DESTINO / "crear_formulario.gs").write_text(apps_script(), encoding="utf-8")
    (DESTINO / "FORMULARIO.md").write_text(guia(), encoding="utf-8")
    print(f"Escrito en {DESTINO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
