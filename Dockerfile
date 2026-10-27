# App del Indice de Bienestar USTA para publicar (Railway u otro servicio).
#
# Corre en modo PUBLICO: solo lee resultados/entrega2/hallazgos_entrega2.json
# (agregados). Los Excel de data/ no entran en la imagen (.dockerignore).
#
# Variables que se configuran en el servicio, NO aqui:
#   APP_PASSWORD        contraseña de acceso (recomendado)
#   OPENAI_API_KEY      solo si se quiere que el chat consulte el corpus
#   CHAT_MAX_CONSULTAS  consultas externas por sesion (por defecto 20)
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INDICE_MODO=publico

COPY app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway define PORT; en local se usa 8501
CMD streamlit run app/dashboard.py --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 --server.headless true
