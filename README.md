# Nada sin fuente: agente estadístico y RAG para el Índice de Bienestar Estudiantil USTA

Entrega de Consultoría e Investigación (Universidad Santo Tomás).
Autoras: Luisa Martínez U. y Juliana Fiquitiva.

Agente de línea de comandos que responde preguntas sobre la encuesta de
bienestar (escala de Ryff, 29 ítems, n = 1.813) y sobre un corpus de 20
documentos académicos, citando siempre de dónde sale cada respuesta. Si
no tiene una fuente, se abstiene.

## Qué contiene
- `agent/`: agente estadístico (`core/`, `statistics/`, `psychometrics/`,
  `validation/`) y módulo RAG (`agent/rag/`).
- `prompts/`: prompt versionado del RAG.
- `tests/`: pruebas automáticas y bancos de preguntas (19 del primer banco
  y 20 del banco final, con respuesta y página comprobadas).
- `resultados/`: corridas del banco, diagnósticos y metadatos.
- `docs/experimento/`: respuestas completas de las 6 corridas del experimento.
- `docs/corpus_inventario_20.csv`: procedencia de los 20 documentos del corpus.

## Qué no contiene
- Los PDF del corpus (cada uno tiene su licencia): ver la procedencia en
  `docs/corpus_inventario_20.csv`.
- Los datos de la encuesta (son de la contraparte).
- El índice vectorial (`rag_store/`, unos 100 MB): se regenera a partir de los PDF.

## Instalación
    pip install -r requirements_agent.txt

Para las respuestas redactadas por el modelo se necesita una clave de OpenAI:

    set OPENAI_API_KEY=...        (Windows, cmd)
    $env:OPENAI_API_KEY="..."     (Windows, PowerShell)

Sin clave el agente también funciona: planifica los análisis con reglas y
entrega el resumen estadístico, pero las consultas al corpus se abstienen.

### Datos y corpus
- Excel de la encuesta en `data/` (`01_dataset_preparado.xlsx`,
  `03_dataset_IRT_ponderado.xlsx`, `Parametros_GRM.xlsx`). Para usar otra
  carpeta, definir la variable `INDICE_DATA_DIR`.
- PDF del corpus en `docs/rag_corpus/`.

## Cómo ejecutarlo
Todos los comandos se ejecutan desde la carpeta del proyecto.

| Qué | Comando |
|---|---|
| Agente completo (encuesta + corpus + búsqueda web) | `python -m agent.ui` |
| Solo el corpus (RAG), con fragmentos y trazabilidad | `python -m agent.rag.cli "pregunta"` |

Dentro del agente: `resumen`, `genero`, `seccional`, `modalidad`, `estrato`,
`dimensiones`, `edad`, `regresion`, `validar`, `validacion`,
`corpus <pregunta>`, `buscar <tema>`, `ayuda`, `salir`, o preguntas en
lenguaje natural.

Cada consulta al corpus añade una fila a `agent/rag/logs/rag_consultas.csv`.

### Índice del corpus (solo si cambian los PDF)
    python -m agent.rag.index             # reutiliza el índice si coincide
    python -m agent.rag.index --rebuild   # lo borra y lo reconstruye
    python -m agent.rag.cli --calibrar    # recalcula el umbral

Cuidado: `index` puede borrar y reconstruir `rag_store/`, y `--calibrar`
sobrescribe `rag_store/calibracion.csv`. El umbral (0,0720) solo vale para
el backend TF-IDF; si se cambia de backend hay que recalibrar y volver a
correr el banco.

## Pruebas
    python -m pytest              # pruebas sin internet (por defecto)
    python -m pytest -m red       # pruebas que llaman a OpenAI o a Crossref/PubMed

- La corrida por defecto oculta la clave de OpenAI: no gasta llamadas pagadas.
- Las pruebas marcadas `datos` se saltan solas si faltan los Excel de `data/`.
- Si alguna prueba modifica `data/`, `rag_store/`, `agent/rag/logs/` o
  `resultados/`, la corrida falla e indica qué archivo cambió.

El experimento del banco (respuestas sin sistema vs. con sistema) se corre
con `python tests/correr_banco_rag.py`. Llama a OpenAI y escribe en `resultados/`.

## Origen
Continúa el estudio del Índice de Bienestar Estudiantil de la USTA, cuya
base de datos y análisis previos son de [completar: autoría].
