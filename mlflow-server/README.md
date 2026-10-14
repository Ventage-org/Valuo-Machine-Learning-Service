# Servidor Externo de MLflow (Open Source)

Este directorio contiene la infraestructura para desplegar y ejecutar el **Servidor MLflow Open Source** de la plataforma Valuo de forma independiente al microservicio de inferencia (`ml-service`).

---

## 1. Arquitectura de Datos y Aislamiento

- **Base de Datos Operacional (`db_operacional_valuo`):**
  - MLflow almacena sus metadatos (experimentos, parámetros, métricas y Model Registry) **exclusivamente** en el schema `mlops`.
  - Cadena de conexión: `postgresql://postgres:pass@localhost:5432/db_operacional_valuo?options=-csearch_path%3Dmlops`.
- **Aislamiento de `inmobiliaria_ml_db`:**
  - La base de datos `inmobiliaria_ml_db` contiene los datos analíticos del modelo y **nunca** es modificada por MLflow.
- **Artifact Store (Modelos binarios y SHAP):**
  - **Desarrollo local:** Almacenamiento en `./artifacts/` (o volumen Docker).
  - **Producción / Cloud:** Contenedor de **Azure Blob Storage** configurado mediante `AZURE_STORAGE_CONNECTION_STRING`.

---

## 2. Cómo Levantar el Servidor MLflow

### Opción A: Con Docker (Recomendado)

Requiere Docker y Docker Desktop:

```bash
cd mlflow-server/

# Iniciar contenedor en segundo plano
docker compose up -d

# Ver logs
docker compose logs -f

# Detener contenedor
docker compose down
```

### Opción B: Localmente con Python (Sin Docker)

Si prefieres ejecutarlo directamente en tu entorno de desarrollo:

1. Asegúrate de tener instalado MLflow y las librerías necesarias:
   ```bash
   pip install mlflow psycopg2-binary azure-storage-blob
   ```
2. Ejecuta el script de arranque:
   ```bash
   cd mlflow-server/
   python run_server.py
   ```

---

## 3. Acceso a la Interfaz Web (UI)

Una vez iniciado, abre tu navegador web en:
👉 **http://localhost:5000**

Allí podrás ver:
- Lista de experimentos (`experimento_venta`, `experimento_alquiler`).
- Runs registrados con sus métricas (MAPE, $R^2$, MAE, RMSE).
- **Model Registry:** Versiones de los modelos y sus estados (`Candidate`, `Production`, `Archived`).

---

## 4. Configuración para Azure Blob Storage

Para almacenar los archivos `.pkl` en Azure Blob Storage en lugar del disco local, configura en `mlflow-server/.env`:

```env
DEFAULT_ARTIFACT_ROOT=wasbs://ml-artifacts@tu_cuenta_storage.blob.core.windows.net/
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=tu_cuenta;AccountKey=tu_key;EndpointSuffix=core.windows.net
```
