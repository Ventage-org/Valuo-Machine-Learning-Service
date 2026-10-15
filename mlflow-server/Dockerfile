FROM python:3.10-slim

WORKDIR /mlflow

# Dependencias del sistema necesarias para compilar drivers si fuese requerido
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar MLflow, driver PostgreSQL y SDK de Azure Blob Storage
RUN pip install --no-cache-dir \
    mlflow==3.16.1 \
    psycopg2-binary==2.9.9 \
    azure-storage-blob==12.19.1 \
    azure-identity==1.15.0

EXPOSE 5000

# Desactivar el middleware de seguridad interno de FastAPI de MLflow 3.5+
# En Azure Container Apps, la seguridad y SSL ya son administrados por el Ingress (HTTPS)
ENV MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true

# Script de arranque optimizado para Cloud:
# - mlflow db upgrade: auto-migración limpia del esquema en PostgreSQL
# - workers 1: bajo consumo de RAM/CPU
# - disable-security-middleware: elimina bloqueos 403 Forbidden de cross-origin en Azure
CMD ["sh", "-c", "mlflow db upgrade ${BACKEND_STORE_URI} && mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 1 \
    --disable-security-middleware"]
