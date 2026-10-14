"""
mlflow-server/run_server.py
───────────────────────────
Script para arrancar el Servidor MLflow Open Source localmente sin necesidad de Docker.
Conecta el Backend Store a PostgreSQL en la base de datos 'db_operacional_valuo' con schema 'mlops'.
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno del servidor MLflow
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# Parámetros de conexión a la base de datos operacional
db_user = os.getenv("DB_USER", "postgres")
db_pass = os.getenv("DB_PASSWORD", "")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "db_operacional_valuo")
db_schema = os.getenv("DB_SCHEMA", "mlops")

port = os.getenv("MLFLOW_PORT", "5000")
artifact_root = os.getenv("DEFAULT_ARTIFACT_ROOT", "./artifacts")

# Crear carpeta de artefactos local si se usa almacenamiento local
if not artifact_root.startswith("wasbs://") and not artifact_root.startswith("az://"):
    os.makedirs(artifact_root, exist_ok=True)

backend_store_uri = (
    f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}?options=-csearch_path%3D{db_schema}"
)

print("=" * 65)
print("  VALUO - Servidor MLflow Open Source")
print("=" * 65)
print(f"  Backend Store URI : postgresql://{db_user}:***@{db_host}:{db_port}/{db_name}?schema={db_schema}")
print(f"  Artifact Root     : {artifact_root}")
print(f"  Puerto            : {port}")
print(f"  UI Web de MLflow  : http://localhost:{port}")
print("=" * 65)

cmd = [
    sys.executable,
    "-m",
    "mlflow",
    "server",
    "--backend-store-uri",
    backend_store_uri,
    "--default-artifact-root",
    artifact_root,
    "--host",
    "0.0.0.0",
    "--port",
    str(port),
]

try:
    subprocess.run(cmd, check=True)
except FileNotFoundError:
    print("\n[ERROR] MLflow no está instalado en este entorno Python.")
    print("Instálalo ejecutando: pip install mlflow psycopg2-binary azure-storage-blob")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n[MLflow Server] Servidor detenido por el usuario.")
