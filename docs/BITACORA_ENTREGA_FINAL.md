# Bitácora de la entrega final

Registro de todo lo nuevo, lo que cambió y lo que mejoró en la entrega final,
para documentarla y exponerla. Rama de trabajo: `entrega-final` (sale de
`entrega-2`). Cada entrada dice qué se hizo, por qué y cómo se verificó.

---

## 1. Encuesta en vivo

**Objetivo.** Que los resultados de la app se actualicen solos a medida que
llegan respuestas nuevas al formulario, con la misma metodología del estudio.

### Cómo funciona

```
Formulario (Google Forms o Microsoft Forms)
   └─> hoja de respuestas (CSV publicado o Excel)
         └─> GitHub Action, cada hora (.github/workflows/encuesta_viva.yml)
               └─> python -m agent.encuesta_viva.actualizar
                     1. normalizar_formulario  acomoda las columnas al formulario original
                     2. preparar               depuración, ítems invertidos, dimensiones,
                                               factores de expansión y calibración por género
                     3. estandarizar           θ EAP con los parámetros GRM fijos del estudio
                     4. entrega2.calcular      IC con varianza del diseño, comparaciones,
                                               brecha, multinivel
               └─> datos_vivos/hallazgos.json (solo agregados) -> commit
                     └─> Render vuelve a desplegar -> página «Encuesta en vivo»
```

### Qué se hizo

| Fecha | Cambio | Archivos |
|---|---|---|
| 2026-11-18 | Los notebooks 01 (preparación y diseño muestral) y 03 (IRT) pasan a código reutilizable y probado. | `agent/encuesta_viva/pipeline.py` |
| 2026-11-18 | Catálogos de programas → NBC copiados textualmente del notebook 01. | `agent/encuesta_viva/catalogos.py` |
| 2026-11-18 | Totales de población por estrato guardados como constantes agregadas: el pipeline corre sin los marcos muestrales (que no se pueden publicar). | `catalogos.py` (`POBLACION_ESTRATOS`) |
| 2026-11-18 | Parámetros GRM del estudio (global + 6 dimensiones) guardados para calificar con parámetros fijos. | `agent/encuesta_viva/parametros_grm.json` |
| 2026-11-18 | El formulario nuevo puede tener otro orden de columnas, columnas extra (p. ej. «Marca temporal» de Google) o no tener ID ni correo: se acomoda solo. Si falta una pregunta del análisis, avisa cuál. | `pipeline.normalizar_formulario` |
| 2026-11-18 | Script de actualización: lee CSV o Excel de un enlace o archivo, no reescribe si nada cambió, nunca guarda el enlace ni respuestas individuales. | `agent/encuesta_viva/actualizar.py` |
| 2026-11-18 | Página nueva en la app: «Encuesta en vivo» (KPIs, comparación con el estudio, distribución, niveles y comparaciones por grupo). | `app/paginas/en_vivo.py`, `app/dashboard.py` |
| 2026-11-18 | Tarea programada en GitHub cada hora. | `.github/workflows/encuesta_viva.yml` |
| 2026-11-18 | 8 pruebas nuevas, incluida la reproducción exacta del estudio. | `tests/test_encuesta_viva.py` |
| 2026-11-18 | Formulario en Google Forms generado desde el catálogo: script de Apps Script que lo crea completo (secciones, saltos por sede, 29 escalas 1–6 y hoja de respuestas) y guía con todas las preguntas exactas. El original era de Microsoft Forms; se elige Google porque su hoja se publica como CSV y la GitHub Action la lee sin iniciar sesión. 2 pruebas más: la hoja que produce el formulario la acepta el pipeline y las guías están al día. | `agent/encuesta_viva/formulario.py`, `docs/encuesta_viva/` |

### Decisiones metodológicas (para exponer)

1. **Parámetros IRT fijos, no reestimados.** Cada respuesta nueva se califica
   con los parámetros GRM estimados en el estudio (θ EAP, misma cuadratura que
   `girth`). Así el T de la encuesta viva está en la **misma escala** que el
   estudio y se puede comparar directamente (la app muestra la diferencia).
   Reestimar con pocas respuestas daría parámetros inestables y una escala que
   cambia cada hora. Sigue disponible: `estandarizar(prep, parametros={})`.
2. **Mismo diseño muestral.** Los factores de expansión se recalculan con las
   respuestas nuevas: N del estrato (constante) / respondentes del estrato, y
   luego la calibración por género. El ajuste por no respuesta se actualiza solo.
3. **Privacidad.** El JSON solo tiene agregados; los grupos con menos de 30
   casos no se muestran. El enlace de la hoja va en un secreto de GitHub.
4. **Pocas respuestas.** Con menos de 30 respuestas la página muestra un aviso
   de inestabilidad. El pipeline funciona desde 15 respuestas (probado).

### Hallazgos al pasar los notebooks a código

Estos dos puntos no estaban documentados y salieron al exigir que el código
reprodujera **exactamente** los datos del estudio:

1. **Calibración por género de los factores de expansión.** El
   `factor_expansion` que usan todos los análisis no es el del notebook 01: está
   calibrado por *raking* a dos márgenes (estrato seccional × modalidad, y
   seccional × género). Ese paso no aparece en ningún notebook; el peso sin
   calibrar quedó en `factor_expansion_original`. Se reconstruyó y reproduce los
   pesos con error < 1e-12 (`pipeline.calibrar_genero`).
2. **Conversión a T-score.** El notebook 03 describe T = 50 + 10·(θ − μ_w)/σ_w
   (normalización ponderada), pero `03_dataset_IRT_ponderado.xlsx` contiene
   T = 50 + 10·θ: con la normalización ponderada la media poblacional sería
   exactamente 50, y la reportada es 50,7. El pipeline usa la conversión que
   realmente produjo los datos (`pipeline.a_t`) y lo deja documentado.

### Verificación

- Excel crudo → pipeline → JSON de la app: **0 diferencias en 2.438 valores**
  frente al JSON publicado de la entrega 2.
- `01_dataset_preparado.xlsx`: 73 columnas iguales y en el mismo orden, 0 diferencias.
- `03_dataset_IRT_ponderado.xlsx`: T y niveles idénticos en las 7 escalas; θ
  difiere a lo sumo 1e-4 (redondeo a 4 decimales).
- EAP propio con parámetros fijos vs. `girth`: diferencia 0,0.
- Formulario de Google simulado (otro orden, sin ID, columnas extra): mismo resultado.
- Tiempo: ~22 s por actualización.

### Pendiente para dejarla funcionando (lo hace la dueña del repositorio)

1. Crear el formulario: pegar `docs/encuesta_viva/crear_formulario.gs` en script.google.com y ejecutar `crearFormulario` (o armarlo a mano con `docs/encuesta_viva/FORMULARIO.md`). Pegar el texto del consentimiento y la autorización del formulario original.
2. En la hoja de respuestas: Archivo > Compartir > Publicar en la web > CSV, y
   copiar el enlace.
3. En GitHub: Settings > Secrets and variables > Actions > New secret
   `ENCUESTA_FUENTE` con ese enlace.
4. Llevar `.github/workflows/encuesta_viva.yml` a `main` (GitHub solo programa
   los workflows de la rama por defecto) y probar con «Run workflow».

---

## 2. Anotaciones del profesor sobre la app

Estado de la encuesta en vivo: el código queda listo y probado, pero por ahora
no se activa (no se puede aplicar la encuesta todavía). Por eso la app ya no la
anuncia como entregable; la página «Encuesta en vivo» sigue disponible y muestra
un aviso mientras no haya respuestas.

| Fecha | Anotación | Cambio | Archivos | Verificación |
|---|---|---|---|---|
| 2026-11-18 | **Estudio**: quitar «Entrega final – Encuesta viva» de los entregables y presentar la app con el agente de IA en línea. | El hito «Entrega final» del recorrido ahora describe la aplicación con agente de inteligencia artificial en línea: explica los métodos del estudio, responde preguntas con respuestas contextualizadas, consulta el corpus documental y permite conversar en línea para profundizar en los resultados. Se quita la marca de «futuro» (ya está entregado). | `app/paginas/estudio.py` | Prueba `test_estudio_presenta_el_agente_como_entrega_final`. |
| 2026-11-18 | **Comparaciones**: conservar solo el mapa y quitar la gráfica «Puntaje T por seccional». | Con «Seccional» seleccionada la página muestra solo el mapa, a todo el ancho y con una nota de lectura (tamaño del punto = estudiantes; el T se ve al pasar el cursor). Las demás variables (pregrado/posgrado, género, estrato, nivel de formación) conservan sus dos gráficos sin cambios. | `app/paginas/comparaciones.py` | Pruebas `test_comparaciones_por_seccional_solo_muestra_el_mapa` y `test_comparaciones_otras_variables_conservan_sus_graficos`. |
| 2026-11-18 | **Dimensiones**: quitar la gráfica «Puntaje T por dimensión». | Se quita esa gráfica; el radar «Perfil de la población» pasa a ocupar todo el ancho (ya no hay columnas con huecos). KPIs, nota, mapa de calor por grupo y perfiles comparados no cambian. | `app/paginas/dimensiones.py` | Prueba `test_dimensiones_sin_grafico_puntaje_por_dimension`. |
| 2026-11-18 | **Agente**: que entienda sinónimos para ampliar las preguntas que responde. | Capa nueva de sinónimos: antes de clasificar la pregunta la pasa a minúsculas, le quita las tildes y traduce sinónimos al vocabulario del agente (felicidad, salud mental, satisfacción con la vida → bienestar; alumnos → estudiantes; chicas/chicos → mujeres/hombres; campus, ciudad, sede → seccional; maestría, especialización → posgrado; condición socioeconómica, ricos/pobres → estrato; nivel educativo → nivel de formación; autoestima, sentido de vida, relaciones con los demás → su dimensión de Ryff; aspecto más débil → dimensión más baja; sesgo en los ítems, escala justa → DIF; coinciden con el informe original → reproducibilidad; teniendo en cuenta → controlando). Además: comparación por **nivel de formación** (estaba precalculada pero el agente no la reconocía) y preguntas generales sin grupo («¿cuántos están en riesgo?», «¿cuál es el puntaje promedio?») → resumen. No cambia ningún cálculo; el corpus y la búsqueda reciben la pregunta original. Para añadir un sinónimo basta una línea en `SINONIMOS`. | `agent/core/sinonimos.py` (nuevo), `agent/core/orchestrator.py`, `agent/core/chat.py`, `agent/core/query_planner.py` | Banco de 46 preguntas redactadas con otras palabras: **antes 17 respondidas, ahora 43**; las 3 restantes se abstienen a propósito (presencial/virtual no está en los resultados públicos; «¿es confiable?» es ambigua). Las rutas a corpus, búsqueda y abstención fuera de tema no cambian. 29 pruebas nuevas en `tests/test_sinonimos.py`. |

**Verificación general:** 367 pruebas pasan (todas las páginas de la app abren
sin errores); los datos, cálculos y resultados no se tocaron; el log protegido
del RAG conserva su huella.
