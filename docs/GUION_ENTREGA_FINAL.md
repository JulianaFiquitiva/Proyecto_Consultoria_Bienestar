# Guion de la entrega final

**Proyecto:** Índice de bienestar psicológico (escala de Ryff), Universidad Santo Tomás.
**Equipo:** Juliana Fiquitiva y Luisa Martínez U.
**App:** https://proyecto-consultoria-bienestar.onrender.com
**Duración sugerida:** 10–12 minutos (5 bloques). El detalle técnico de cada
cambio está en `docs/BITACORA_ENTREGA_FINAL.md`.

---

## Resumen en 30 segundos

> En esta entrega cerramos el proyecto con una **aplicación web con un agente de
> inteligencia artificial en línea**. El agente explica los métodos del estudio,
> responde preguntas sobre los resultados y consulta el corpus de documentos
> académicos. Ahora **entiende muchas más formas de preguntar**, porque reconoce
> sinónimos, y **atendimos las anotaciones del profesor** para simplificar la
> app. Además dejamos construida y probada la **encuesta en vivo**, que puede
> actualizar los resultados sola en cuanto se aplique el formulario.

---

## Bloque 1 · Dónde estábamos (1 min)

**Qué decir**
- El estudio original (Consultorio de Estadística USTA, febrero de 2026) midió el
  bienestar con la escala de Ryff: 29 ítems y 6 dimensiones, con puntajes T
  (centro 50, desviación 10) y un diseño muestral por seccional y modalidad.
- **Entrega 1:** un agente estadístico que recibe una pregunta en lenguaje
  natural, elige el método, revisa los supuestos, lo ejecuta y lo interpreta.
- **Avance:** RAG. El agente responde sobre 20 documentos académicos y cita el
  documento y la página. Si no tiene fuente, se abstiene.
- **Entrega 2:** estimaciones con el diseño muestral, reproducción del estudio
  anterior, DIF, brecha ajustada, modelo multinivel y la app publicada.

**Qué mostrar:** la página **«El estudio»**, sección *Recorrido*. El último hito
ya presenta la entrega final.

---

## Bloque 2 · La entrega final: la app con el agente en línea (3 min)

**Qué decir**
- La app es el producto final. El agente virtual permite:
  1. **Consultar los métodos** usados en el estudio y recibir una explicación.
  2. **Preguntar sobre el estudio** y recibir respuestas contextualizadas, con la
     fuente y el método que eligió (se ve en «Cómo decidió»).
  3. **Consultar el corpus** de documentos que sirve como base de conocimiento.
  4. **Conversar en línea** para resolver dudas y profundizar en los resultados.
- La versión publicada usa **solo resultados agregados**: no expone respuestas
  individuales.

**Demostración en «Pregúntale al agente»** (preguntas verificadas):
1. `¿Cómo está el bienestar general?` → resumen con intervalo de confianza.
2. `¿Hay diferencias entre hombres y mujeres?` → comparación con la varianza del diseño.
3. `¿Qué dicen los documentos sobre el bienestar psicológico?` → respuesta del corpus con cita.
4. `¿Cuál es la capital de Francia?` → **se abstiene**: no inventa fuera del estudio.

---

## Bloque 3 · Mejora del agente: entiende sinónimos (2–3 min)

**El problema.** El agente reconocía un vocabulario fijo y sin tildes. Por eso
no respondía *«¿Cómo está la felicidad de los alumnos?»*, ni *«¿Qué campus
tiene mejor bienestar?»*, ni *«¿Cuál es la dimensión más baja?»* (no reconocía
«dimensión» con tilde).

**La solución.** Una capa de sinónimos: antes de interpretar la pregunta, el
agente la pasa a minúsculas, le quita las tildes y traduce los sinónimos a las
palabras que ya conoce. Algunos ejemplos:

| La persona dice | El agente entiende |
|---|---|
| felicidad, salud mental, satisfacción con la vida | bienestar |
| alumnos, universitarios | estudiantes |
| chicas / chicos | mujeres / hombres |
| campus, ciudad, sede | seccional |
| maestría, especialización | posgrado |
| condición socioeconómica, ricos / pobres | estrato |
| nivel educativo | nivel de formación |
| autoestima, sentido de vida, relaciones con los demás | su dimensión de Ryff |
| aspecto más débil | dimensión más baja |
| sesgo en los ítems, ¿la escala es justa? | DIF (¿la escala mide igual?) |
| ¿coinciden con el informe original? | reproducibilidad |

**El resultado (para la diapositiva).** Lo medimos con un banco de 46 preguntas
redactadas con otras palabras: **antes respondía 17, ahora responde 43**. Las 3
restantes se abstienen a propósito:
- presencial frente a virtual (2 preguntas): no está en los resultados públicos;
- «¿Es confiable el análisis?»: es demasiado ambigua.

**Lo importante.** No cambió ningún cálculo, y el agente **sigue
absteniéndose** fuera del estudio. Agregar un sinónimo nuevo es añadir una línea.

**Demostración:**
- `¿Las chicas tienen más bienestar que los chicos?`
- `¿Qué campus tiene mejor bienestar?`
- `¿Cuál es el aspecto más débil del bienestar?`
- `¿Hay diferencias por nivel de formación?` (estaba calculado, pero antes no se podía preguntar)

---

## Bloque 4 · Anotaciones del profesor (1–2 min)

| Sección | Anotación | Qué hicimos |
|---|---|---|
| El estudio | Quitar «Encuesta viva» de los entregables y presentar el agente | El hito «Entrega final» describe la app con el agente de IA en línea. |
| Comparaciones | Dejar solo el mapa | Por seccional se ve solo el mapa, a todo el ancho y con una nota de lectura. Las demás variables no cambian. |
| Dimensiones | Quitar «Puntaje T por dimensión» | El radar del perfil ocupa todo el ancho; el resto de la página no cambia. |

**Qué mostrar:** abrir **Comparaciones** con «Seccional» seleccionada, y luego **Dimensiones**.

---

## Bloque 5 · Trabajo adicional: la encuesta en vivo, lista para activarse (2 min)

**Qué decir**
- Dejamos construido el flujo para que la app se actualice sola con respuestas
  nuevas: formulario de Google → hoja de respuestas → tarea automática en GitHub
  cada hora → mismo análisis del estudio → la app se vuelve a publicar.
- Para lograrlo pasamos los notebooks de preparación e IRT a **código
  reutilizable y probado**. Ese código **reproduce exactamente** el estudio:
  0 diferencias en 2.438 valores.
- Las respuestas nuevas se califican con los **parámetros IRT del estudio**, así
  que quedan en la misma escala y se pueden comparar.
- En el proceso **descubrimos dos pasos que no estaban documentados**:
  1. los factores de expansión tenían una **calibración por género** (raking);
  2. el puntaje T de los datos es **50 + 10·θ**, no la normalización ponderada
     que describe el notebook.
  Los dos quedaron reconstruidos y documentados.
- Está **lista pero no activada**, porque la encuesta todavía no se aplica. La
  página «Encuesta en vivo» aparecerá sola en el menú cuando lleguen las
  primeras respuestas.

---

## Cierre (30 s)

> El resultado es una aplicación pública con un agente que explica, responde,
> cita sus fuentes y se abstiene cuando no sabe. Ahora entiende cómo pregunta
> la gente, y está preparada para actualizarse con nuevas aplicaciones de la
> encuesta. Todo cambio quedó probado (366 pruebas automáticas) y registrado en
> la bitácora.

---

## Cifras para las diapositivas

| Dato | Valor |
|---|---|
| Preguntas con sinónimos respondidas | 17 → 43 de 46 |
| Pruebas automáticas | 366, todas pasan |
| Reproducción del estudio desde el Excel crudo | 0 diferencias en 2.438 valores |
| Tiempo de una actualización de la encuesta en vivo | ~22 s |
| Documentos en el corpus del agente | 20 |

## Posibles preguntas del profesor

- **¿Por qué no se reestiman los parámetros IRT con cada respuesta nueva?**
  Con pocas respuestas serían inestables y la escala cambiaría cada hora. Con
  parámetros fijos, los puntajes son comparables con el estudio.
- **¿El agente puede inventar?** Solo responde con resultados precalculados o
  con documentos citados. Si la pregunta no corresponde, se abstiene y explica
  el motivo.
- **¿Qué pasa si alguien pregunta algo que no está precalculado?** El agente lo
  dice explícitamente y lista los temas sobre los que sí puede responder.
- **¿Cómo se protege la privacidad?** La app pública solo tiene agregados; los
  grupos con menos de 30 casos no se muestran.
