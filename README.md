# Nada sin fuente: RAG para el Índice de Bienestar Estudiantil USTA

Entrega de Consultoría e Investigación (Universidad Santo Tomás).
Autoras: Luisa Martínez U. y Juliana Fiquitiva.

## Qué contiene
- `agent/`: agente estadístico y módulo RAG (`agent/rag/`).
- `prompts/`: prompt versionado del RAG.
- `tests/`: pruebas y banco de 19 preguntas con respuesta y página comprobadas.
- `resultados/`: corridas del banco, diagnósticos y metadatos.
- `docs/experimento/`: respuestas completas de las 6 corridas del experimento.

## Qué no contiene
- Los PDF del corpus (cada uno tiene su licencia): ver la procedencia en `docs/corpus_inventario.csv`.
- Los datos de la encuesta (son de la contraparte).

## Cómo ejecutarlo
    pip install -r requirements_agent.txt
    python -m agent.rag.cli "pregunta"
Requiere los PDF en `docs/rag_corpus/` y haber indexado con `python -m agent.rag.index`. [verificar este comando]

## Origen
Continúa el estudio del Índice de Bienestar Estudiantil de la USTA, cuya base de datos y análisis previos son de [completar: autoría].