# ==============================================================================
# Dockerfile — ml-service (FastAPI + XGBoost + MLflow Client)
# ==============================================================================
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema requeridas:
# - libgomp1: Soporte OpenMP para cómputo multihilo de XGBoost C++
# - libpq-dev: Driver nativo para PostgreSQL
# - curl: Para healthcheck probes de Docker y Kubernetes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias Python en caché limpia
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente, datos de metadata, modelos y configuración
COPY app/ ./app/
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    CONFIG_PATH=config/model_config.json \
    CONFIG_ALQUILER_PATH=config/model_alquiler_config.json \
    METADATA_PATH=data/features_metadata.json \
    METADATA_ALQUILER_PATH=data/features_metadata_alquiler.json \
    MODELS_DIR=models

EXPOSE 8000

# Verificación de salud interna del contenedor
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de arranque del servicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
