FROM python:3.10-slim

WORKDIR /mlflow

# Dependencias del sistema necesarias para compilar drivers si fuese requerido
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar MLflow LTS estable, driver PostgreSQL y SDK de Azure Blob Storage
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    psycopg2-binary==2.9.9 \
    azure-storage-blob==12.19.1 \
    azure-identity==1.15.0

EXPOSE 5000

# Script de arranque optimizado para Cloud (Bajo consumo de RAM y CPU):
# mlflow server inicializa automáticamente todas las tablas base y migraciones en esquemas limpios
CMD ["sh", "-c", "mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --artifacts-destination ${DEFAULT_ARTIFACT_ROOT} \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 1"]
