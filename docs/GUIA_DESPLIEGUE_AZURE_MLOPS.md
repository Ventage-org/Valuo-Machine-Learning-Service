# Guía y Registro de Despliegue MLOps en Azure — Valuo

**Proyecto**: Valuo — Servicio de Tasación Inmobiliaria (Machine Learning & MLOps)  
**Entorno**: Desarrollo (`rg-valuo-dev1`)  
**Región de Azure**: `chilecentral` (Chile Central — Azure for Students)  
**Fecha de registro**: Octubre 2026  
**Arquitecto**: Cloud & MLOps Solutions Architecture  
**Estado Actual**: `mlflow-server` y `ml-service` 100% OPERATIVOS Y DESPLEGADOS EN AZURE  

---

## 1. Topología de Infraestructura en Azure

Toda la solución se consolida dentro de un único Grupo de Recursos para optimizar costos, simplificar la gobernanza y compartir resolución de red interna:

```mermaid
graph TD
    subgraph RG ["Resource Group: rg-valuo-dev1 (Chile Central)"]
        ACR["Azure Container Registry\n(acrvaluodev - SKU Basic)"]
        PG["Azure Database for PostgreSQL Flexible Server\n(psql-valuo-dev-01.postgres.database.azure.com)\nBD: db_operational_valuo & inmobiliaria_ml_db"]
        SA["Azure Storage Account: stvaluodev (LRS Standard)\nContainers: mlflow-artifacts, datasets, reportes"]

        subgraph CAE ["Container Apps Environment: cae-valuo-dev1"]
            MLFLOW["Container App: mlflow-server [OPERATIVO]\n(0.25 vCPU / 0.5 GiB RAM)\nPuerto Ingress: 5000 (Público HTTPS)"]
            MLSERVICE["Container App: ml-service [OPERATIVO :v1]\n(0.5 vCPU / 1.0 GiB RAM)\nPuerto Ingress: 8000 (Público HTTPS)"]
        end

        ACR -.->|Docker Pull (Admin Auth)| CAE
        MLFLOW -->|Metadatos / Runs| PG
        MLFLOW -->|Pesos / Modelos / Artifacts| SA
        MLSERVICE -->|Tracking & Registry| MLFLOW
        MLSERVICE -->|Contexto Distrital| PG
    end
```

### Inventario de Recursos Aprovisionados

| Recurso | Nombre / Host | SKU / Nivel | Propósito |
| :--- | :--- | :--- | :--- |
| **Resource Group** | `rg-valuo-dev1` | N/A | Contenedor principal de ciclo de vida en `chilecentral`. |
| **Container Registry** | `acrvaluodev` | Basic (10 GB) | Almacén unificado de imágenes Docker para todos los microservicios. |
| **PostgreSQL Flexible Server** | `psql-valuo-dev-01` | Estándar / Dev | Base de datos relacional multi-esquema (*Schema-per-Service*). |
| **Storage Account** | `stvaluodev` | Standard LRS | Almacén de blobs (artefactos MLflow, datasets Excel/CSV y reportes PDF). |
| **Container Apps Environment** | `cae-valuo-dev1` | Consumption | Entorno administrado de Kubernetes para alojar los microservicios. |

---

## 2. Decisiones de Arquitectura y Buenas Prácticas

### 2.1. Un solo ACR para todos los microservicios
No se crean múltiples registros de contenedores. Un único ACR (`acrvaluodev.azurecr.io`) aloja los distintos repositorios:
* `acrvaluodev.azurecr.io/mlflow:latest`
* `acrvaluodev.azurecr.io/ml-service:v1`
* `acrvaluodev.azurecr.io/valuo-data-pipeline:latest` *(job de ingesta)*

### 2.2. Exclusión de secretos en imágenes (.dockerignore & 12-Factor App)
* Los archivos `.env` **nunca se compilan dentro de las imágenes Docker**.
* Las credenciales se inyectan en tiempo de ejecución a través de las *Environment Variables* y *Secrets* de Azure Container Apps.

### 2.3. Limitación Regional de `chilecentral` (ACR Tasks)
* Al intentar ejecutar `az acr build`, Azure reportó que la función de compilación remota (ACR Tasks) no está habilitada en la región de Chile Central (`NoRegisteredProviderFound`).
* **Solución adoptada**: Compilación local mediante Docker Desktop y publicación remota con `docker push`.

### 2.4. Estrategia de Modelos en `ml-service` (Bootstrap vs Desacoplado)
* **Fase 1 (Arranque / Bootstrap)**: Los modelos baseline iniciales (`xgboost_venta_v2.pkl` y `xgboost_alquiler_v1.pkl`) se incluyen dentro del contenedor para garantizar arranque inmediato y disponibilidad de inferencia sin depender de un MLflow preexistente.
* **Fase 2 (Producción 100% Cloud Native)**: Una vez registrado el modelo en MLflow y almacenado en Blob Storage, las futuras revisiones de la imagen excluirán la carpeta `models/`, descargando los artefactos dinámicamente desde Azure Storage.

---

## 3. Proceso de Construcción y Publicación de Contenedores

Ejecutado desde la raíz del repositorio (`Valuo-Machine-Learning-Service`):

```powershell
# 1. Autenticación en el registro de Azure
az acr login --name acrvaluodev

# 2. Compilación y subida de MLflow Server
docker build -t acrvaluodev.azurecr.io/mlflow:latest ./mlflow-server
docker push acrvaluodev.azurecr.io/mlflow:latest

# 3. Compilación y subida de ML Service (FastAPI + XGBoost + SHAP) con tag inmutable v1
docker build -t acrvaluodev.azurecr.io/ml-service:v1 ./ml-service
docker push acrvaluodev.azurecr.io/ml-service:v1
```

---

## 4. Despliegue de `mlflow-server` en Azure Container Apps

### 4.1. Configuración de Cómputo y Entorno
* **Nombre de la App**: `mlflow-server`
* **Entorno**: `cae-valuo-dev1`
* **Asignación de recursos**: `0.25 vCPU` y `0.5 GiB de RAM` (Consumo real en reposo ~150 MB).

### 4.2. Autenticación de Imagen en el Asistente del Portal
Para evitar el conflicto de identidades administradas en entornos Express de Azure:
* **Origen de la imagen**: `Docker Hub or other registries` (Privado)
* **Servidor de registro**: `acrvaluodev.azurecr.io`
* **Nombre de usuario**: `acrvaluodev`
* **Contraseña**: Clave de acceso de administrador de `acrvaluodev`
* **Imagen y etiqueta**: `mlflow:latest` *(Importante: no anteponer el dominio del servidor porque Azure lo duplica)*.

### 4.3. Aislamiento de Esquema (Schema-per-Service) y Usuario Dedicado
En el proyecto coexisten 7 módulos sobre una misma base de datos usando el patrón **Schema-per-Service** para ahorro presupuestal. 

Para evitar que MLflow contaminara el esquema `public`, y para **NO alterar el `search_path` del superadministrador `valuo_admin`** (lo cual afectaría a los otros 6 microservicios), se aprovisionó un usuario de base de datos exclusivo para MLflow:

```sql
-- Ejecutado en PostgreSQL (db_operational_valuo):
-- 1. Asegurar public estándar para el admin
CREATE SCHEMA IF NOT EXISTS public;
ALTER USER valuo_admin SET search_path TO public;

-- 2. Crear esquema mlops
CREATE SCHEMA IF NOT EXISTS mlops;

-- 3. Crear usuario dedicado para MLflow
CREATE USER mlflow_user WITH PASSWORD 'MlflowPass2026!';

-- 4. Asignar propiedad (OWNER) y privilegios exclusivos sobre el esquema mlops
ALTER SCHEMA mlops OWNER TO mlflow_user;
GRANT CONNECT ON DATABASE db_operational_valuo TO mlflow_user;
GRANT ALL ON SCHEMA mlops TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA mlops GRANT ALL ON TABLES TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA mlops GRANT ALL ON SEQUENCES TO mlflow_user;

-- 5. ASIGNACIÓN CLAVE: Este usuario siempre creará y consultará en mlops
ALTER USER mlflow_user SET search_path TO mlops;
```

### 4.4. Variables de Entorno de `mlflow-server` (Runtime)

| Variable | Valor Configurado |
| :--- | :--- |
| `BACKEND_STORE_URI` | `postgresql://mlflow_user:MlflowPass2026!@psql-valuo-dev-01.postgres.database.azure.com:5432/db_operational_valuo?sslmode=require` |
| `DEFAULT_ARTIFACT_ROOT` | `wasbs://mlflow-artifacts@stvaluodev.blob.core.windows.net/` |
| `AZURE_STORAGE_CONNECTION_STRING` | `DefaultEndpointsProtocol=https;AccountName=stvaluodev;AccountKey=/OPp0MyMHzx60pqyPlVfLyWPiexIVjBLNNQmjwPjcyM5LZdcc9+SSEQC5bSncrPKdF1uEmAlsz9U+AStSg368A==;EndpointSuffix=core.windows.net` |
| `AZURE_STORAGE_ACCESS_KEY` | `/OPp0MyMHzx60pqyPlVfLyWPiexIVjBLNNQmjwPjcyM5LZdcc9+SSEQC5bSncrPKdF1uEmAlsz9U+AStSg368A==` |

### 4.5. Configuración de Red e Ingress
* **Entrada (Ingress)**: Habilitada
* **Tipo de tráfico**: Aceptación desde cualquier lugar (*Accepting traffic from anywhere*)
* **Tipo**: HTTP / Transporte Automático
* **Conexiones no seguras**: Deshabilitadas (Fuerza HTTPS en puerto 443 administrado por Azure)
* **Puerto de destino**: `5000` (Puerto interno expuesto por MLflow)
* **URL Pública Asignada**: `https://mlflow-server.delightfuldesert-51ebdce7.chilecentral.azurecontainerapps.io/`

### 4.6. Lecciones y Troubleshooting Resueltos en MLflow
1. **Error `ExpressEnvironmentManagedIdentityNotSupported`**:
   * *Solución*: Usar la opción "Docker Hub or other registries" con credenciales de admin del ACR.
2. **Error `BuildFailed: image was not found` (Duplicación FQDN)**:
   * *Solución*: En el campo "Image and tag" colocar solo `mlflow:latest`.
3. **Error `InvalidSchemaName: no schema has been selected to create in`**:
   * *Causa*: Al no estar en `public`, PostgreSQL no sabía en qué esquema crear tablas. Si el usuario no es dueño del esquema, rechaza la operación.
   * *Solución*: Ejecutar `ALTER SCHEMA mlops OWNER TO mlflow_user;` y fijar `ALTER USER mlflow_user SET search_path TO mlops;`.
4. **Error `Got unexpected extra arguments (| | AZURE_STORAGE_CONNECTION_STRING)`**:
   * *Solución*: Limpiar caracteres de tablas Markdown pegados accidentalmente en las cajas de variables.

---

## 5. Despliegue de `ml-service` en Azure Container Apps

### 5.1. Justificación de Dimensionamiento y Puerto 8000
* **Memoria y Cómputo**: `0.5 vCPU` y `1.0 GiB de RAM`.
  * *Motivo*: FastAPI en reposo consume ~120 MB, pero al instanciar simultáneamente en memoria los dos modelos XGBoost (`xgboost_venta_v2.pkl` y `xgboost_alquiler_v1.pkl`) junto con el explicador de árboles SHAP (*TreeExplainer*), el consumo en inferencia activa alcanza entre 350 MB y 450 MB. 1.0 GiB garantiza margen de seguridad del 100%.
* **Puerto de Destino 8000**:
  * *Motivo*: Es el puerto estándar de FastAPI/Uvicorn. En Azure Container Apps, cada contenedor posee su propio espacio de red aislado y subdominio público en HTTPS (puerto 443). **No existe conflicto de puertos** con los otros microservicios que también utilicen el puerto 8000.

### 5.2. Configuración en el Portal de Azure
* **Nombre de la App**: `ml-service`
* **Entorno**: Reutiliza `cae-valuo-dev1`
* **Origen de imagen**: `Docker Hub or other registries` (Privado)
  * Servidor: `acrvaluodev.azurecr.io`
  * Usuario: `acrvaluodev`
  * Imagen y etiqueta: `ml-service:v1` *(Etiqueta inmutable)*
* **Asignación de recursos**: `0.5 vCPU` / `1.0 GiB RAM`

### 5.3. Variables de Entorno (Runtime)

| Variable | Valor Configurado | Propósito |
| :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | `https://mlflow-server.delightfuldesert-51ebdce7.chilecentral.azurecontainerapps.io` | Conexión HTTPS hacia el servidor MLflow en Azure |
| `DB_HOST` | `psql-valuo-dev-01.postgres.database.azure.com` | Host de Azure PostgreSQL |
| `DB_PORT` | `5432` | Puerto estándar PostgreSQL |
| `DB_USER` | `valuo_admin` | Usuario de conexión |
| `DB_PASSWORD` | `V@aluo_admin` | Contraseña de conexión |
| `DB_NAME` | `inmobiliaria_ml_db` | Base de datos con tablas de contexto distrital e histórico |
| `MLFLOW_EXPERIMENT_VENTA` | `experimento_tasacion_venta` | Nombre del experimento MLflow para modelos de venta |
| `MLFLOW_EXPERIMENT_ALQUILER` | `experimento_tasacion_alquiler` | Nombre del experimento MLflow para modelos de alquiler |
| `MLFLOW_MODEL_NAME_VENTA` | `xgboost_venta` | Modelo registrado en MLflow (Venta) |
| `MLFLOW_MODEL_NAME_ALQUILER` | `xgboost_alquiler` | Modelo registrado en MLflow (Alquiler) |
| `ADMIN_TOKEN` | *(Vacío)* | Permite acceso libre a endpoints administrativos en fase DEV |
| `AZURE_STORAGE_CONNECTION_STRING` | `DefaultEndpointsProtocol=https;AccountName=stvaluodev;AccountKey=/OPp0MyMHzx60pqyPlVfLyWPiexIVjBLNNQmjwPjcyM5LZdcc9+SSEQC5bSncrPKdF1uEmAlsz9U+AStSg368A==;EndpointSuffix=core.windows.net` |
| `AZURE_STORAGE_ACCESS_KEY` | `/OPp0MyMHzx60pqyPlVfLyWPiexIVjBLNNQmjwPjcyM5LZdcc9+SSEQC5bSncrPKdF1uEmAlsz9U+AStSg368A==` |


### 5.4. Configuración de Red e Ingress
* **Entrada (Ingress)**: Habilitada
* **Tráfico**: Aceptación desde cualquier lugar (*Accepting traffic from anywhere*)
* **Tipo**: HTTP / Transporte Automático
* **Conexiones no seguras**: Deshabilitadas (Fuerza HTTPS en puerto 443)
* **Puerto de destino**: `8000`

### 5.5. Lecciones y Troubleshooting Resueltos en `ml-service`
1. **Error `ModuleNotFoundError: No module named 'psutil'`**:
   * *Causa*: `monitoreo_service.py` importa `psutil` para reportar métricas del host y memoria, pero no estaba incluido en `requirements.txt`.
   * *Solución*: Se agregó `psutil>=5.9.0` a `requirements.txt` y se recompiló la imagen.
2. **La Trampa de Caché de Kubernetes/ACA con la etiqueta `:latest`**:
   * *Causa*: Azure Container Apps utiliza `ImagePullPolicy: IfNotPresent` en sus nodos subyacentes. Al hacer un reinicio simple manteniendo `:latest`, el nodo ejecutó la imagen en caché local sin descargar la nueva versión de ACR.
   * *Solución y Buena Práctica*: Implementar versionamiento inmutable mediante etiquetas (e.g., `ml-service:v1`). Al desplegar una nueva revisión apuntando a `:v1`, Azure detectó un cambio real de imagen y forzó la descarga limpia desde el registro.

---

## 6. Checklist de Validación y Verificación

1. **Salud del Servicio (`/health`)**:
   Verificar respuesta JSON con estado `healthy`, modelos cargados (`venta` y `alquiler`) y distritos disponibles.
2. **Documentación Swagger / OpenAPI (`/docs`)**:
   Acceso a la interfaz interactiva para realizar pruebas directas.
3. **Predicción de Venta (`POST /api/v1/prediccion/venta`)**:
   Enviar un payload de prueba con datos inmobiliarios de Lima Metropolitana y confirmar respuesta con:
   * Precio en Soles constantes y nominales.
   * Valores explicativos de SHAP (Top 10 variables explicativas).
