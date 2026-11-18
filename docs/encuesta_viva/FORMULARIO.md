# Formulario de la encuesta en vivo

Generado por `python -m agent.encuesta_viva.formulario`. Los títulos deben ser **idénticos** a estos (el análisis reconoce las columnas por su texto; tildes, mayúsculas y el punto final no importan, pero cualquier otra diferencia sí).

**Forma rápida:** pegar `crear_formulario.gs` en script.google.com y ejecutarlo: crea el formulario completo con su hoja de respuestas.

**Reglas si se arma a mano:**
- Los 29 ítems van como **29 preguntas separadas de «Escala lineal» de 1 a 6**. No usar «Cuadrícula de varias opciones»: Google la exporta con otros nombres de columna.
- No pedir correo ni nombre.
- Cada lista de programas es una pregunta aparte en su propia sección; la pregunta «Sede o seccional» salta a la sección que corresponde.

## Sección 1. Consentimiento

| # | Pregunta (título exacto) | Tipo | Opciones | Obligatoria |
|---|---|---|---|---|
| 1 | Consentimiento informado | Opción múltiple | Acepto · No acepto (→ enviar) | Sí |
| 2 | Autorización expresa para el tratamiento de datos personales | Opción múltiple | Acepto · No acepto (→ enviar) | Sí |
| 3 | ¿Soy mayor de edad? | Opción múltiple | Sí · No (→ enviar) | Sí |

El texto completo del consentimiento y de la autorización va en la descripción de cada pregunta: copiarlo del formulario original.

## Sección 2. Información general

| # | Pregunta (título exacto) | Tipo | Opciones | Obligatoria |
|---|---|---|---|---|
| 4 | Género | Opción múltiple | Femenino · Masculino · No binario · Prefiero no decirlo · Otro (texto libre) | Sí |
| 5 | Edad | Respuesta corta (número) | entre 14 y 99 | Sí |
| 6 | País de residencia | Respuesta corta | — | No |
| 7 | Departamento de residencia | Respuesta corta | — | No |
| 8 | Ciudad de residencia en Colombia | Respuesta corta | — | No |
| 9 | Estrato socioeconómico | Opción múltiple | 1 · 2 · 3 · 4 · 5 · 6 | Sí |
| 10 | Ciudad de residencia fuera de Colombia | Respuesta corta | — | No |
| 11 | Grupo poblacional | Opción múltiple | Ninguno · Afrodescendiente · Indígena · Raizal · ROM - Gitano · Población LGBTIQ+ · Víctima del conflicto armado · Ciudadano extranjero (no colombiano) | Sí |
| 12 | Nivel de formación | Opción múltiple | Pregrado · Técnico Profesional · Técnico Laboral · Tecnológico · Especialización · Maestría · Doctorado · Posdoctorado | Sí |
| 13 | Modalidad | Opción múltiple | Presencial · Virtual · Híbrido (presencial - virtual) · A distancia | Sí |
| 14 | Semestre | Opción múltiple | 1 · 2 · 3 · 4 · 5 · 6 · 7 · 8 · 9 · 10 | Sí |
| 15 | Sede o seccional | Opción múltiple (salta a la sección de su sede) | Sede Principal Bogotá · Centro de Atención Universitaria CAU · Campus Medellín · Seccional Bucaramanga · Seccional Tunja · Seccional Villavicencio | Sí |

## Sección 3. Programa (según la sede)

Quien elige *Centro de Atención Universitaria CAU* responde primero el centro y luego la lista de Bogotá (la pregunta de Bogotá incluye los CAU).

**Centro de Atención Universitaria CAU** (desplegable): Armenia · Barrancabermeja · Barranquilla · Bogotá · Bucaramanga · Cali · Chiquinquirá · Cúcuta · Duitama · Facatativá · Medellín · Montería · Pasto · Tunja · Valledupar · Yopal

**Programas de formación Sede Principal Bogotá y centros de atención universitaria CAU** (desplegable, 73 opciones):

- ADMINISTRACIÓN AMBIENTAL Y DE LOS RECURSOS NATURALES
- ADMINISTRACIÓN DE EMPRESAS
- ADMINISTRACIÓN DE EMPRESAS AGROPECUARIAS
- COMUNICACIÓN SOCIAL
- COMUNICACIÓN Y EDUCACIÓN DIGITAL
- CONSTRUCCIÓN EN ARQUITECTURA E INGENIERÍA
- CONTADURÍA PÚBLICA
- CULTURA FÍSICA, DEPORTE Y RECREACIÓN
- DERECHO
- DISEÑO GRÁFICO
- DOCTORADO EN DERECHO
- DOCTORADO EN EDUCACIÓN
- DOCTORADO EN FILOSOFÍA
- ECONOMÍA
- ESPECIALIZACIÓN EN DEFENSA DE LOS DERECHOS HUMANOS
- ESPECIALIZACIÓN EN DERECHO ADMINISTRATIVO
- ESPECIALIZACIÓN EN DERECHO PENAL
- ESPECIALIZACIÓN EN FINANZAS
- ESPECIALIZACIÓN EN GERENCIA DE EMPRESAS AGROPECUARIAS
- ESPECIALIZACIÓN EN GERENCIA DE MULTIMEDIA  Y COMUNICACIÓN DIGITAL
- ESPECIALIZACIÓN EN GERENCIA EMPRESARIAL
- ESPECIALIZACIÓN EN GESTIÓN TERRITORIAL Y AVALÚOS
- ESPECIALIZACIÓN EN PATOLOGÍA DE LA CONSTRUCCIÓN
- ESPECIALIZACIÓN EN PEDAGOGÍA PARA LA EDUCACIÓN SUPERIOR
- ESPECIALIZACIÓN EN PSICOLOGÍA JURÍDICA Y FORENSE
- ESTADÍSTICA
- FINANZAS
- GOBIERNO Y RELACIONES INTERNACIONALES
- INGENIERÍA AMBIENTAL
- INGENIERÍA CIVIL
- INGENIERÍA DE DATOS E INTELIGENCIA ARTIFICIAL
- INGENIERÍA DE TELECOMUNICACIONES
- INGENIERÍA ELECTRÓNICA
- INGENIERÍA EN INFORMÁTICA
- INGENIERÍA EN LOGÍSTICA Y OPERACIONES
- INGENIERÍA INDUSTRIAL
- INGENIERÍA MECÁNICA
- LICENCIATURA EN ARTES PLÁSTICAS
- LICENCIATURA EN BIOLOGÍA
- LICENCIATURA EN EDUCACIÓN INFANTIL
- LICENCIATURA EN EDUCACIÓN RELIGIOSA
- LICENCIATURA EN FILOSOFÍA Y LETRAS
- LICENCIATURA EN LENGUAS EXTRANJERAS INGLÉS
- LICENCIATURA EN TEOLOGÍA
- MAESTRÍA EN AMBIENTES BILINGÜES DE APRENDIZAJE
- MAESTRÍA EN CALIDAD Y GESTIÓN INTEGRAL
- MAESTRÍA EN COMUNICACIÓN, CULTURAS DIGITALES Y VISUALIZACIÓN DE DATOS
- MAESTRÍA EN COMUNICACIÓN, DESARROLLO Y CAMBIO SOCIAL
- MAESTRÍA EN CRIMINOLOGÍA Y POLÍTICA PÚBLICA CRIMINAL
- MAESTRÍA EN DEFENSA DE LOS DERECHOS HUMANOS Y DEL DERECHO INTERNACIONAL HUMANITARIO ANTE ORGANISMOS, TRIBUNALES Y CORTES INTERNACIONALES
- MAESTRÍA EN DERECHO PENAL
- MAESTRÍA EN EDUCACIÓN
- MAESTRÍA EN EDUCACIÓN SOSTENIBLE Y GESTIÓN AMBIENTAL TERRITORIAL
- MAESTRÍA EN EDUCACIÓN STEM PARA EL DESARROLLO SOCIAL
- MAESTRÍA EN ESTADÍSTICA APLICADA
- MAESTRÍA EN GESTIÓN DE LA SEGURIDAD Y SALUD EN EL TRABAJO
- MAESTRÍA EN GESTIÓN DEL TALENTO HUMANO
- MAESTRÍA EN GESTIÓN Y EVALUACIÓN EDUCATIVA
- MAESTRÍA EN GOBIERNO Y RELACIONES INTERNACIONALES
- MAESTRÍA EN INFANCIAS
- MAESTRÍA EN INGENIERÍA DE DISEÑO Y MANUFACTURA
- MAESTRÍA EN PATOLOGÍA DE LA CONSTRUCCIÓN
- MAESTRÍA EN PEDAGOGÍAS CRÍTICAS E INTERVENCIÓN SOCIOEDUCATIVA
- MAESTRÍA EN PSICOLOGÍA CLÍNICA Y DE LA FAMILIA
- MAESTRÍA EN TECNOLOGÍA E INNOVACIÓN EDUCATIVA
- MAESTRÍA EN TEOLOGÍA
- MERCADEO
- NEGOCIOS INTERNACIONALES
- PROFESIONAL EN MODA
- PSICOLOGÍA
- SOCIOLOGÍA
- TEOLOGÍA
- ZOOTECNIA

**Programas de formación Campus Medellín** (desplegable, 8 opciones):

- ARQUITECTURA
- DERECHO
- ESPECIALIZACIÓN EN GERENCIA EMPRESARIAL
- ESPECIALIZACIÓN EN GERENCIA LOGÍSTICA INTERNACIONAL
- MAESTRÍA EN ADMINISTRACIÓN MBA
- MAESTRÍA EN INNOVACIÓN
- MAESTRÍA EN INTELIGENCIA DE NEGOCIOS
- NEGOCIOS INTERNACIONALES

**Programas de formación Seccional Bucaramanga** (desplegable, 29 opciones):

- ADMINISTRACIÓN DE EMPRESAS
- ADMINISTRACIÓN DE EMPRESAS AGROPECUARIAS
- ARQUITECTURA
- CONTADURÍA PÚBLICA
- CULTURA FÍSICA, DEPORTE Y RECREACIÓN
- DERECHO
- ECONOMÍA
- ESPECIALIZACIÓN EN ADMINISTRACIÓN DEPORTIVA
- ESPECIALIZACIÓN EN AUDITORÍA DE SALUD
- ESPECIALIZACIÓN EN GERENCIA DE INSTITUCIONES DE SEGURIDAD SOCIAL EN SALUD
- ESPECIALIZACIÓN EN GERENCIA DE LA PRODUCTIVIDAD Y DE LAS OPERACIONES LOGÍSTICAS
- ESPECIALIZACIÓN EN MODELACIÓN Y COORDINACIÓN BIM DE EDIFICACIONES Y OBRAS CIVILES
- ESPECIALIZACIÓN EN OPTOMETRÍA PEDIÁTRICA
- ESPECIALIZACIÓN EN ORTODONCIA
- ESPECIALIZACIÓN EN SEGURIDAD Y SALUD EN EL TRABAJO
- INGENIERÍA CIVIL
- INGENIERÍA DE TELECOMUNICACIONES
- INGENIERÍA INDUSTRIAL
- INGENIERÍA MECATRÓNICA
- LICENCIATURA EN LENGUAS EXTRANJERAS
- MAESTRÍA EN ADMINISTRACIÓN MBA
- MAESTRÍA EN CALIDAD Y GESTIÓN INTEGRAL
- MAESTRÍA EN DIRECCIÓN Y GESTIÓN DE PROYECTOS
- MAESTRÍA EN EDUCACIÓN AMBIENTAL
- MAESTRÍA EN GESTIÓN Y CONSULTORÍA TIC´S
- NEGOCIOS INTERNACIONALES
- ODONTOLOGÍA
- OPTOMETRÍA
- TECNOLOGÍA EN LABORATORIO DENTAL

**Programas de formación Seccional Tunja** (desplegable, 37 opciones):

- ADMINISTRACIÓN DE EMPRESAS
- ARQUITECTURA
- BIOINGENIERÍA
- CONTADURÍA PÚBLICA
- CULTURA FÍSICA, DEPORTE Y RECREACIÓN
- DERECHO
- DISEÑO DE INTERACCIÓN
- DOCTORADO EN PEDAGOGÍA Y NEUROCIENCIA APLICADA A LA EDUCACIÓN
- ESPECIALIZACIÓN EN AUDITORÍA DE SALUD
- ESPECIALIZACIÓN EN CONTRATACIÓN ESTATAL
- ESPECIALIZACIÓN EN DERECHO ADMINISTRATIVO
- ESPECIALIZACIÓN EN DERECHO PENAL
- ESPECIALIZACIÓN EN ESTRUCTURAS
- ESPECIALIZACIÓN EN GEOTECNIA VIAL Y PAVIMENTOS
- ESPECIALIZACIÓN EN GERENCIA DE MANTENIMIENTO Y GESTIÓN DE ACTIVOS
- ESPECIALIZACIÓN EN GERENCIA DE PROYECTOS DE CONSTRUCCIÓN
- ESPECIALIZACIÓN EN GOBIERNO Y GESTIÓN TERRITORIAL
- ESPECIALIZACIÓN EN INGENIERÍA HIDROAMBIENTAL
- ESPECIALIZACIÓN EN INNOVACIÓN Y MARKETING
- ESPECIALIZACIÓN EN PSICOLOGÍA JURÍDICA Y FORENSE
- INGENIERÍA AMBIENTAL
- INGENIERÍA CIVIL
- INGENIERÍA DE DATOS E INTELIGENCIA ARTIFICIAL
- INGENIERÍA DE SISTEMAS
- INGENIERÍA ELECTRÓNICA
- INGENIERÍA INDUSTRIAL
- INGENIERÍA MECÁNICA
- LICENCIATURA EN ESPAÑOL Y LENGUAS EXTRANJERAS INGLÉS Y FRANCÉS
- MAESTRÍA EN ADMINISTRACIÓN MBA
- MAESTRÍA EN DERECHO MINERO AMBIENTAL
- MAESTRÍA EN DERECHO PENAL
- MAESTRÍA EN DERECHO PRIVADO
- MAESTRÍA EN ENTRENAMIENTO DEPORTIVO Y ACTIVIDAD FÍSICA
- MAESTRÍA EN INCLUSIÓN, EDUCACIÓN Y DIVERSIDAD
- MAESTRÍA EN MARKETING INTERNACIONAL Y NEGOCIOS
- MARKETING Y TRANSFORMACIÓN DIGITAL
- NEGOCIOS INTERNACIONALES

**Programas de formación Seccional Villavicencio** (desplegable, 19 opciones):

- ADMINISTRACIÓN DE EMPRESAS AGROPECUARIAS
- ARQUITECTURA
- CONTADURÍA PÚBLICA
- CULTURA FÍSICA, DEPORTE Y RECREACIÓN
- DERECHO
- ESPECIALIZACIÓN EN ESTRUCTURAS
- ESPECIALIZACIÓN EN GERENCIA DE MANTENIMIENTO Y GESTIÓN DE ACTIVOS
- ESPECIALIZACIÓN EN GERENCIA EMPRESARIAL
- ESPECIALIZACIÓN EN PSICOLOGÍA JURÍDICA Y FORENSE
- INGENIERÍA AMBIENTAL
- INGENIERÍA CIVIL
- INGENIERÍA DE SISTEMAS
- INGENIERÍA INDUSTRIAL
- INGENIERÍA MECATRÓNICA
- INGENIERÍA MECÁNICA
- MAESTRÍA EN ADMINISTRACIÓN MBA
- MAESTRÍA EN DERECHOS HUMANOS
- NEGOCIOS INTERNACIONALES
- PSICOLOGÍA

## Sección 4. Escala de bienestar (Ryff, 29 ítems)

Todas: **Escala lineal de 1 a 6**, 1 = Totalmente en desacuerdo, 6 = Totalmente de acuerdo, obligatorias. Los ítems con * están redactados en negativo (el análisis los invierte; en el formulario se preguntan igual).

| # | Pregunta (título exacto) |
|---|---|
| 1 | Cuando repaso la historia de mi vida estoy contento con cómo han resultado las cosas. |
| 2 | A menudo me siento solo porque tengo pocos amigos íntimos con quienes compartir mis preocupaciones. * |
| 3 | No tengo miedo de expresar mis opiniones, incluso cuando son opuestas a las opiniones de la mayoría de la gente. |
| 4 | Me preocupa cómo otra gente evalúa las elecciones que he hecho en mi vida. * |
| 5 | Me resulta difícil dirigir mi vida hacia un camino que me satisfaga. * |
| 6 | Disfruto haciendo planes para el futuro y trabajar para hacerlos realidad. |
| 7 | En general, me siento seguro y positivo conmigo mismo. |
| 8 | No tengo muchas personas que quieran escucharme cuando necesito hablar. * |
| 9 | Tiendo a preocuparme sobre lo que otra gente piensa de mí. * |
| 10 | He sido capaz de construir un hogar y un modo de vida a mi gusto. |
| 11 | Soy una persona activa al realizar los proyectos que propuse para mí mismo. |
| 12 | Siento que mis amistades me aportan muchas cosas. |
| 13 | Tiendo a estar influenciado por la gente con fuertes convicciones. * |
| 14 | En general, siento que soy responsable de la situación en la que vivo. |
| 15 | Me siento bien cuando pienso en lo que he hecho en el pasado y lo que espero hacer en el futuro. |
| 16 | Mis objetivos en la vida han sido más una fuente de satisfacción que de frustración para mí. |
| 17 | Me gusta la mayor parte de los aspectos de mi personalidad. |
| 18 | Tengo confianza en mis opiniones incluso si son contrarias al consenso general. |
| 19 | Las demandas de la vida diaria a menudo me deprimen. * |
| 20 | Tengo clara la dirección y el objetivo de mi vida. |
| 21 | En general, con el tiempo siento que sigo aprendiendo más sobre mí mismo. |
| 22 | No he experimentado muchas relaciones cercanas y de confianza. * |
| 23 | Es difícil para mí expresar mis propias opiniones en asuntos polémicos. * |
| 24 | En su mayor parte, me siento orgulloso de quien soy y la vida que llevo. |
| 25 | Sé que puedo confiar en mis amigos, y ellos saben que pueden confiar en mí. |
| 26 | Cuando pienso en ello, realmente con los años no he mejorado mucho como persona. * |
| 27 | Tengo la sensación de que con el tiempo me he desarrollado mucho como persona. |
| 28 | Para mí, la vida ha sido un proceso continuo de estudio, cambio y crecimiento. |
| 29 | Si me sintiera infeliz con mi situación de vida daría los pasos más eficaces para cambiarla. |
