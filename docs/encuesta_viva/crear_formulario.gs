/**
 * Crea el formulario de la Escala de Bienestar (encuesta en vivo) en Google
 * Forms, con su hoja de respuestas. Generado por
 * agent/encuesta_viva/formulario.py: no editar a mano los títulos (deben ser
 * idénticos al formulario original para que el análisis los reconozca).
 *
 * Uso: script.google.com > Proyecto nuevo > pegar este archivo > Ejecutar
 * «crearFormulario» > autorizar. Los enlaces salen en el registro.
 */
function crearFormulario() {
  var form = FormApp.create('Formulario de la escala de bienestar subjetivo');
  form.setDescription('Universidad Santo Tomás. Estudio de bienestar de la población ' +
    'estudiantil. Sus respuestas son anónimas y se analizan solo de forma agregada.');
  form.setCollectEmail(false);
  form.setProgressBar(true);

  function unica(titulo, opciones, req, ayuda, otro) {
    var it = form.addMultipleChoiceItem().setTitle(titulo).setRequired(req);
    it.setChoiceValues(opciones);
    if (otro) it.showOtherOption(true);
    if (ayuda) it.setHelpText(ayuda);
    return it;
  }
  function seccion(titulo) { return form.addPageBreakItem().setTitle(titulo); }

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
    {"t": "Género", "tipo": "unica", "op": ["Femenino", "Masculino", "No binario", "Prefiero no decirlo"], "req": true, "ayuda": ""},
    {"t": "Edad", "tipo": "numero", "op": null, "req": true, "ayuda": "Número entero, en años."},
    {"t": "País de residencia", "tipo": "texto", "op": null, "req": false, "ayuda": ""},
    {"t": "Departamento de residencia", "tipo": "texto", "op": null, "req": false, "ayuda": ""},
    {"t": "Ciudad de residencia en Colombia", "tipo": "texto", "op": null, "req": false, "ayuda": ""},
    {"t": "Estrato socioeconómico", "tipo": "unica", "op": ["1", "2", "3", "4", "5", "6"], "req": true, "ayuda": ""},
    {"t": "Ciudad de residencia fuera de Colombia", "tipo": "texto", "op": null, "req": false, "ayuda": "Solo si vive fuera de Colombia."},
    {"t": "Grupo poblacional", "tipo": "unica", "op": ["Ninguno", "Afrodescendiente", "Indígena", "Raizal", "ROM - Gitano", "Población LGBTIQ+", "Víctima del conflicto armado", "Ciudadano extranjero (no colombiano)"], "req": true, "ayuda": ""},
    {"t": "Nivel de formación", "tipo": "unica", "op": ["Pregrado", "Técnico Profesional", "Técnico Laboral", "Tecnológico", "Especialización", "Maestría", "Doctorado", "Posdoctorado"], "req": true, "ayuda": ""},
    {"t": "Modalidad", "tipo": "unica", "op": ["Presencial", "Virtual", "Híbrido (presencial - virtual)", "A distancia"], "req": true, "ayuda": ""},
    {"t": "Semestre", "tipo": "unica", "op": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], "req": true, "ayuda": ""}
  ];
  datos.forEach(function (d) {
    if (d.tipo === 'unica') {
      unica(d.t, d.op, d.req, d.ayuda, d.t === 'Género');
    } else if (d.tipo === 'numero') {
      form.addTextItem().setTitle(d.t).setRequired(d.req).setHelpText(d.ayuda)
        .setValidation(FormApp.createTextValidation().requireNumberBetween(14, 99).build());
    } else {
      var it = form.addTextItem().setTitle(d.t).setRequired(d.req);
      if (d.ayuda) it.setHelpText(d.ayuda);
    }
  });
  var sede = unica('Sede o seccional', ["Sede Principal Bogotá", "Centro de Atención Universitaria CAU", "Campus Medellín", "Seccional Bucaramanga", "Seccional Tunja", "Seccional Villavicencio"], true);

  // 3. Programa según la sede
  var pCau = seccion('Centro de Atención Universitaria');
  form.addListItem().setTitle('Centro de Atención Universitaria CAU').setRequired(true)
    .setChoiceValues(["Armenia", "Barrancabermeja", "Barranquilla", "Bogotá", "Bucaramanga", "Cali", "Chiquinquirá", "Cúcuta", "Duitama", "Facatativá", "Medellín", "Montería", "Pasto", "Tunja", "Valledupar", "Yopal"]);
  var programas = {"Programas de formación Sede Principal Bogotá y centros de atención universitaria CAU": ["ADMINISTRACIÓN AMBIENTAL Y DE LOS RECURSOS NATURALES", "ADMINISTRACIÓN DE EMPRESAS", "ADMINISTRACIÓN DE EMPRESAS AGROPECUARIAS", "COMUNICACIÓN SOCIAL", "COMUNICACIÓN Y EDUCACIÓN DIGITAL", "CONSTRUCCIÓN EN ARQUITECTURA E INGENIERÍA", "CONTADURÍA PÚBLICA", "CULTURA FÍSICA, DEPORTE Y RECREACIÓN", "DERECHO", "DISEÑO GRÁFICO", "DOCTORADO EN DERECHO", "DOCTORADO EN EDUCACIÓN", "DOCTORADO EN FILOSOFÍA", "ECONOMÍA", "ESPECIALIZACIÓN EN DEFENSA DE LOS DERECHOS HUMANOS", "ESPECIALIZACIÓN EN DERECHO ADMINISTRATIVO", "ESPECIALIZACIÓN EN DERECHO PENAL", "ESPECIALIZACIÓN EN FINANZAS", "ESPECIALIZACIÓN EN GERENCIA DE EMPRESAS AGROPECUARIAS", "ESPECIALIZACIÓN EN GERENCIA DE MULTIMEDIA  Y COMUNICACIÓN DIGITAL", "ESPECIALIZACIÓN EN GERENCIA EMPRESARIAL", "ESPECIALIZACIÓN EN GESTIÓN TERRITORIAL Y AVALÚOS", "ESPECIALIZACIÓN EN PATOLOGÍA DE LA CONSTRUCCIÓN", "ESPECIALIZACIÓN EN PEDAGOGÍA PARA LA EDUCACIÓN SUPERIOR", "ESPECIALIZACIÓN EN PSICOLOGÍA JURÍDICA Y FORENSE", "ESTADÍSTICA", "FINANZAS", "GOBIERNO Y RELACIONES INTERNACIONALES", "INGENIERÍA AMBIENTAL", "INGENIERÍA CIVIL", "INGENIERÍA DE DATOS E INTELIGENCIA ARTIFICIAL", "INGENIERÍA DE TELECOMUNICACIONES", "INGENIERÍA ELECTRÓNICA", "INGENIERÍA EN INFORMÁTICA", "INGENIERÍA EN LOGÍSTICA Y OPERACIONES", "INGENIERÍA INDUSTRIAL", "INGENIERÍA MECÁNICA", "LICENCIATURA EN ARTES PLÁSTICAS", "LICENCIATURA EN BIOLOGÍA", "LICENCIATURA EN EDUCACIÓN INFANTIL", "LICENCIATURA EN EDUCACIÓN RELIGIOSA", "LICENCIATURA EN FILOSOFÍA Y LETRAS", "LICENCIATURA EN LENGUAS EXTRANJERAS INGLÉS", "LICENCIATURA EN TEOLOGÍA", "MAESTRÍA EN AMBIENTES BILINGÜES DE APRENDIZAJE", "MAESTRÍA EN CALIDAD Y GESTIÓN INTEGRAL", "MAESTRÍA EN COMUNICACIÓN, CULTURAS DIGITALES Y VISUALIZACIÓN DE DATOS", "MAESTRÍA EN COMUNICACIÓN, DESARROLLO Y CAMBIO SOCIAL", "MAESTRÍA EN CRIMINOLOGÍA Y POLÍTICA PÚBLICA CRIMINAL", "MAESTRÍA EN DEFENSA DE LOS DERECHOS HUMANOS Y DEL DERECHO INTERNACIONAL HUMANITARIO ANTE ORGANISMOS, TRIBUNALES Y CORTES INTERNACIONALES", "MAESTRÍA EN DERECHO PENAL", "MAESTRÍA EN EDUCACIÓN", "MAESTRÍA EN EDUCACIÓN SOSTENIBLE Y GESTIÓN AMBIENTAL TERRITORIAL", "MAESTRÍA EN EDUCACIÓN STEM PARA EL DESARROLLO SOCIAL", "MAESTRÍA EN ESTADÍSTICA APLICADA", "MAESTRÍA EN GESTIÓN DE LA SEGURIDAD Y SALUD EN EL TRABAJO", "MAESTRÍA EN GESTIÓN DEL TALENTO HUMANO", "MAESTRÍA EN GESTIÓN Y EVALUACIÓN EDUCATIVA", "MAESTRÍA EN GOBIERNO Y RELACIONES INTERNACIONALES", "MAESTRÍA EN INFANCIAS", "MAESTRÍA EN INGENIERÍA DE DISEÑO Y MANUFACTURA", "MAESTRÍA EN PATOLOGÍA DE LA CONSTRUCCIÓN", "MAESTRÍA EN PEDAGOGÍAS CRÍTICAS E INTERVENCIÓN SOCIOEDUCATIVA", "MAESTRÍA EN PSICOLOGÍA CLÍNICA Y DE LA FAMILIA", "MAESTRÍA EN TECNOLOGÍA E INNOVACIÓN EDUCATIVA", "MAESTRÍA EN TEOLOGÍA", "MERCADEO", "NEGOCIOS INTERNACIONALES", "PROFESIONAL EN MODA", "PSICOLOGÍA", "SOCIOLOGÍA", "TEOLOGÍA", "ZOOTECNIA"], "Programas de formación Campus Medellín": ["ARQUITECTURA", "DERECHO", "ESPECIALIZACIÓN EN GERENCIA EMPRESARIAL", "ESPECIALIZACIÓN EN GERENCIA LOGÍSTICA INTERNACIONAL", "MAESTRÍA EN ADMINISTRACIÓN MBA", "MAESTRÍA EN INNOVACIÓN", "MAESTRÍA EN INTELIGENCIA DE NEGOCIOS", "NEGOCIOS INTERNACIONALES"], "Programas de formación Seccional Bucaramanga": ["ADMINISTRACIÓN DE EMPRESAS", "ADMINISTRACIÓN DE EMPRESAS AGROPECUARIAS", "ARQUITECTURA", "CONTADURÍA PÚBLICA", "CULTURA FÍSICA, DEPORTE Y RECREACIÓN", "DERECHO", "ECONOMÍA", "ESPECIALIZACIÓN EN ADMINISTRACIÓN DEPORTIVA", "ESPECIALIZACIÓN EN AUDITORÍA DE SALUD", "ESPECIALIZACIÓN EN GERENCIA DE INSTITUCIONES DE SEGURIDAD SOCIAL EN SALUD", "ESPECIALIZACIÓN EN GERENCIA DE LA PRODUCTIVIDAD Y DE LAS OPERACIONES LOGÍSTICAS", "ESPECIALIZACIÓN EN MODELACIÓN Y COORDINACIÓN BIM DE EDIFICACIONES Y OBRAS CIVILES", "ESPECIALIZACIÓN EN OPTOMETRÍA PEDIÁTRICA", "ESPECIALIZACIÓN EN ORTODONCIA", "ESPECIALIZACIÓN EN SEGURIDAD Y SALUD EN EL TRABAJO", "INGENIERÍA CIVIL", "INGENIERÍA DE TELECOMUNICACIONES", "INGENIERÍA INDUSTRIAL", "INGENIERÍA MECATRÓNICA", "LICENCIATURA EN LENGUAS EXTRANJERAS", "MAESTRÍA EN ADMINISTRACIÓN MBA", "MAESTRÍA EN CALIDAD Y GESTIÓN INTEGRAL", "MAESTRÍA EN DIRECCIÓN Y GESTIÓN DE PROYECTOS", "MAESTRÍA EN EDUCACIÓN AMBIENTAL", "MAESTRÍA EN GESTIÓN Y CONSULTORÍA TIC´S", "NEGOCIOS INTERNACIONALES", "ODONTOLOGÍA", "OPTOMETRÍA", "TECNOLOGÍA EN LABORATORIO DENTAL"], "Programas de formación Seccional Tunja": ["ADMINISTRACIÓN DE EMPRESAS", "ARQUITECTURA", "BIOINGENIERÍA", "CONTADURÍA PÚBLICA", "CULTURA FÍSICA, DEPORTE Y RECREACIÓN", "DERECHO", "DISEÑO DE INTERACCIÓN", "DOCTORADO EN PEDAGOGÍA Y NEUROCIENCIA APLICADA A LA EDUCACIÓN", "ESPECIALIZACIÓN EN AUDITORÍA DE SALUD", "ESPECIALIZACIÓN EN CONTRATACIÓN ESTATAL", "ESPECIALIZACIÓN EN DERECHO ADMINISTRATIVO", "ESPECIALIZACIÓN EN DERECHO PENAL", "ESPECIALIZACIÓN EN ESTRUCTURAS", "ESPECIALIZACIÓN EN GEOTECNIA VIAL Y PAVIMENTOS", "ESPECIALIZACIÓN EN GERENCIA DE MANTENIMIENTO Y GESTIÓN DE ACTIVOS", "ESPECIALIZACIÓN EN GERENCIA DE PROYECTOS DE CONSTRUCCIÓN", "ESPECIALIZACIÓN EN GOBIERNO Y GESTIÓN TERRITORIAL", "ESPECIALIZACIÓN EN INGENIERÍA HIDROAMBIENTAL", "ESPECIALIZACIÓN EN INNOVACIÓN Y MARKETING", "ESPECIALIZACIÓN EN PSICOLOGÍA JURÍDICA Y FORENSE", "INGENIERÍA AMBIENTAL", "INGENIERÍA CIVIL", "INGENIERÍA DE DATOS E INTELIGENCIA ARTIFICIAL", "INGENIERÍA DE SISTEMAS", "INGENIERÍA ELECTRÓNICA", "INGENIERÍA INDUSTRIAL", "INGENIERÍA MECÁNICA", "LICENCIATURA EN ESPAÑOL Y LENGUAS EXTRANJERAS INGLÉS Y FRANCÉS", "MAESTRÍA EN ADMINISTRACIÓN MBA", "MAESTRÍA EN DERECHO MINERO AMBIENTAL", "MAESTRÍA EN DERECHO PENAL", "MAESTRÍA EN DERECHO PRIVADO", "MAESTRÍA EN ENTRENAMIENTO DEPORTIVO Y ACTIVIDAD FÍSICA", "MAESTRÍA EN INCLUSIÓN, EDUCACIÓN Y DIVERSIDAD", "MAESTRÍA EN MARKETING INTERNACIONAL Y NEGOCIOS", "MARKETING Y TRANSFORMACIÓN DIGITAL", "NEGOCIOS INTERNACIONALES"], "Programas de formación Seccional Villavicencio": ["ADMINISTRACIÓN DE EMPRESAS AGROPECUARIAS", "ARQUITECTURA", "CONTADURÍA PÚBLICA", "CULTURA FÍSICA, DEPORTE Y RECREACIÓN", "DERECHO", "ESPECIALIZACIÓN EN ESTRUCTURAS", "ESPECIALIZACIÓN EN GERENCIA DE MANTENIMIENTO Y GESTIÓN DE ACTIVOS", "ESPECIALIZACIÓN EN GERENCIA EMPRESARIAL", "ESPECIALIZACIÓN EN PSICOLOGÍA JURÍDICA Y FORENSE", "INGENIERÍA AMBIENTAL", "INGENIERÍA CIVIL", "INGENIERÍA DE SISTEMAS", "INGENIERÍA INDUSTRIAL", "INGENIERÍA MECATRÓNICA", "INGENIERÍA MECÁNICA", "MAESTRÍA EN ADMINISTRACIÓN MBA", "MAESTRÍA EN DERECHOS HUMANOS", "NEGOCIOS INTERNACIONALES", "PSICOLOGÍA"]};
  var secciones = {};
  Object.keys(programas).forEach(function (col) {
    secciones[col] = seccion('Programa de formación');
    form.addListItem().setTitle(col).setRequired(true).setChoiceValues(programas[col]);
  });

  // 4. Escala de bienestar (29 ítems, 1 a 6)
  var pEscala = seccion('Escala de bienestar');
  pEscala.setHelpText('Indique qué tan de acuerdo está con cada afirmación: ' +
    '1 = Totalmente en desacuerdo, 6 = Totalmente de acuerdo.');
  ["Cuando repaso la historia de mi vida estoy contento con cómo han resultado las cosas.", "A menudo me siento solo porque tengo pocos amigos íntimos con quienes compartir mis preocupaciones.", "No tengo miedo de expresar mis opiniones, incluso cuando son opuestas a las opiniones de la mayoría de la gente.", "Me preocupa cómo otra gente evalúa las elecciones que he hecho en mi vida.", "Me resulta difícil dirigir mi vida hacia un camino que me satisfaga.", "Disfruto haciendo planes para el futuro y trabajar para hacerlos realidad.", "En general, me siento seguro y positivo conmigo mismo.", "No tengo muchas personas que quieran escucharme cuando necesito hablar.", "Tiendo a preocuparme sobre lo que otra gente piensa de mí.", "He sido capaz de construir un hogar y un modo de vida a mi gusto.", "Soy una persona activa al realizar los proyectos que propuse para mí mismo.", "Siento que mis amistades me aportan muchas cosas.", "Tiendo a estar influenciado por la gente con fuertes convicciones.", "En general, siento que soy responsable de la situación en la que vivo.", "Me siento bien cuando pienso en lo que he hecho en el pasado y lo que espero hacer en el futuro.", "Mis objetivos en la vida han sido más una fuente de satisfacción que de frustración para mí.", "Me gusta la mayor parte de los aspectos de mi personalidad.", "Tengo confianza en mis opiniones incluso si son contrarias al consenso general.", "Las demandas de la vida diaria a menudo me deprimen.", "Tengo clara la dirección y el objetivo de mi vida.", "En general, con el tiempo siento que sigo aprendiendo más sobre mí mismo.", "No he experimentado muchas relaciones cercanas y de confianza.", "Es difícil para mí expresar mis propias opiniones en asuntos polémicos.", "En su mayor parte, me siento orgulloso de quien soy y la vida que llevo.", "Sé que puedo confiar en mis amigos, y ellos saben que pueden confiar en mí.", "Cuando pienso en ello, realmente con los años no he mejorado mucho como persona.", "Tengo la sensación de que con el tiempo me he desarrollado mucho como persona.", "Para mí, la vida ha sido un proceso continuo de estudio, cambio y crecimiento.", "Si me sintiera infeliz con mi situación de vida daría los pasos más eficaces para cambiarla."].forEach(function (t) {
    form.addScaleItem().setTitle(t).setBounds(1, 6)
      .setLabels("Totalmente en desacuerdo", "Totalmente de acuerdo").setRequired(true);
  });

  // Saltos: quien no acepta termina; cada sede va a su lista de programas;
  // después de la lista de programas, a la escala.
  [[consentimiento, 'Acepto', 'No acepto'], [autorizacion, 'Acepto', 'No acepto'],
   [mayor, 'Sí', 'No']].forEach(function (x) {
    x[0].setChoices([x[0].createChoice(x[1], FormApp.PageNavigationType.CONTINUE),
                     x[0].createChoice(x[2], FormApp.PageNavigationType.SUBMIT)]);
  });
  var cols = Object.keys(programas);
  var destino = {'Centro de Atención Universitaria CAU': pCau};
  var sedeACol = {"Sede Principal Bogotá": "Programas de formación Sede Principal Bogotá y centros de atención universitaria CAU", "Campus Medellín": "Programas de formación Campus Medellín", "Seccional Bucaramanga": "Programas de formación Seccional Bucaramanga", "Seccional Tunja": "Programas de formación Seccional Tunja", "Seccional Villavicencio": "Programas de formación Seccional Villavicencio"};
  Object.keys(sedeACol).forEach(function (s) { destino[s] = secciones[sedeACol[s]]; });
  sede.setChoices(["Sede Principal Bogotá", "Centro de Atención Universitaria CAU", "Campus Medellín", "Seccional Bucaramanga", "Seccional Tunja", "Seccional Villavicencio"].map(function (s) {
    return sede.createChoice(s, destino[s]);
  }));
  // El CAU sigue con la lista de Bogotá (la siguiente sección); las demás
  // listas de programas saltan a la escala.
  for (var i = 1; i < cols.length; i++) secciones[cols[i]].setGoToPage(pEscala);

  var hoja = SpreadsheetApp.create('Respuestas - Escala de bienestar (encuesta en vivo)');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, hoja.getId());
  Logger.log('Formulario (para responder): ' + form.getPublishedUrl());
  Logger.log('Formulario (para editar):    ' + form.getEditUrl());
  Logger.log('Hoja de respuestas:          ' + hoja.getUrl());
}
