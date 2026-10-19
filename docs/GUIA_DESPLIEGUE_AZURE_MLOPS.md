# Guía y Registro de Despliegue MLOps en Azure — Valuo

**Proyecto**: Valuo — Servicio de Tasación Inmobiliaria (Machine Learning & MLOps)  
**Entorno**: Desarrollo (`rg-valuo-dev1`)  
**Región de Azure**: `chilecentral` (Chile Central — Azure for Students)  
**Fecha de actualización**: Octubre 2026  
**Arquitecto**: Cloud & MLOps Solutions Architecture  
**Estado Actual**: `mlflow-server` y `ml-service` 100% OPERATIVOS Y DESPLEGADOS EN AZURE  

---

## 1. Topología de Infraestructura en Azure

Toda la solución se consolida dentro de un único Grupo de Recursos para optimizar costos, simplificar la gobernanza y compartir resolución de red interna:

```mermaid
graph TD
    subgraph RG ["Resource Group: rg-valuo-dev1 (Chile Central)"]
        ACR["Azure Container Registry
(acrvaluodev - SKU Basic)"]
        PG["Azure Database for PostgreSQL Flexible Server
(psql-valuo-dev-01.postgres.database.azure.com)
BD: db_operational_valuo & db_dataset_valuo"]
        SA["Azure Storage Account: stvaluodev (LRS Standard)
Containers: mlflow-artifacts, datasets, reportes"]

        subgraph CAE ["Container Apps Environment: cae-valuo-dev1"]
            MLFLOW["Container App: mlflow-server [OPERATIVO :v4]
(0.5 vCPU / 1.0 GiB RAM)
Puerto Ingress: 5000 (Público HTTPS)"]
            MLSERVICE["Container App: ml-service [OPERATIVO :v2]
(0.5 vCPU / 1.0 GiB RAM)
Puerto Ingress: 8000 (Público HTTPS)"]
        end

        ACR -.->|Docker Pull (Admin Auth)| CAE
        MLFLOW -->|Metadatos / Runs / Model Registry| PG
        MLFLOW -->|Pesos / Modelos / Artifacts| SA
        MLSERVICE -->|Tracking & Model Registry| MLFLOW
        MLSERVICE -->|Dataset & Contexto Distrital| PG
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
* `acrvaluodev.azurecr.io/mlflow:v3`
* `acrvaluodev.azurecr.io/ml-service:v2`
* `acrvaluodev.azurecr.io/valuo-data-pipeline:latest` *(job de ingesta ETL)*

### 2.2. Exclusión de secretos y binarios (.dockerignore & 12-Factor App)
* Los archivos `.env` **nunca se compilan dentro de las imágenes Docker**.
* Las credenciales se inyectan en tiempo de ejecución a través de las *Environment Variables* y *Secrets* de Azure Container Apps.
* **Pure Cloud Registry (Sin modelos locales)**: La imagen de `ml-service` excluye mediante `.dockerignore` todos los archivos `models/*.pkl` y `models/*.joblib`. El contenedor es 100% desacoplado de los datos y pesos, descargándolos dinámicamente desde Azure Blob Storage vía MLflow.

### 2.3. Limitación Regional de `chilecentral` (ACR Tasks)
* Al intentar ejecutar `az acr build`, Azure reportó que la función de compilación remota (ACR Tasks) no está habilitada en la región de Chile Central (`NoRegisteredProviderFound`).
* **Solución adoptada**: Compilación local mediante Docker Desktop y publicación remota con `docker push`.

---

## 3. Proceso de Construcción y Publicación de Contenedores

Ejecutado desde la raíz del repositorio (`Valuo-Machine-Learning-Service`):

```powershell
# 1. Autenticación en el registro de Azure ACR
az acr login --name acrvaluodev

# 2. Compilación y subida de MLflow Server (con auto-migración y middleware desactivado)
docker build -t acrvaluodev.azurecr.io/mlflow:v3 ./mlflow-server
docker push acrvaluodev.azurecr.io/mlflow:v3

# 3. Compilación y subida de ML Service (ultraliviano sin binarios locales y endpoint de auditoría)
docker build -t acrvaluodev.azurecr.io/ml-service:v2 ./ml-service
docker push acrvaluodev.azurecr.io/ml-service:v2
```

---

## 4. Despliegue de `mlflow-server` en Azure Container Apps

### 4.1. Configuración de Cómputo y Recursos
* **Nombre de la App**: `mlflow-server`
* **Entorno**: `cae-valuo-dev1`
* **Asignación de recursos**: **`0.5 vCPU` y `1.0 GiB de RAM`** *(Optimizado con MLflow 2.19.0 LTS)*.
  * *Justificación Técnica*: Con la migración a MLflow 2.19.0 LTS (REST API liviana sin sobrecarga de GraphQL ni evaluadores pesados de GenAI), el servidor opera de manera estable con un consumo en reposo de ~250-350 MB de RAM. Con 0.5 vCPU y 1.0 GiB se elimina el 100% CPU Throttling y se reduce a la mitad el costo en Azure Container Apps manteniendo compatibilidad total con Model Registry y Azure Blob Storage.
* **Estrategia de Optimización de Costos**:
  * Configurar **Min Replicas: 0** y **Max Replicas: 1**.
  * Tras 5 minutos de inactividad, Azure escala a 0 réplicas (costo S/ 0.00). Al recibir una petición, realiza un *cold-start* de ~15 segundos.

### 4.2. Autenticación de Imagen en el Asistente del Portal
Para evitar el conflicto de identidades administradas en entornos Express de Azure:
* **Origen de la imagen**: `Docker Hub or other registries` (Privado)
* **Servidor de registro**: `acrvaluodev.azurecr.io`
* **Nombre de usuario**: `acrvaluodev`
* **Contraseña**: Clave de acceso de administrador de `acrvaluodev`
* **Imagen y etiqueta**: `mlflow:v3` *(Importante: no anteponer el dominio del servidor porque Azure lo duplica)*.

### 4.3. Aislamiento de Esquema (Schema-per-Service) y Usuario Dedicado
En el proyecto coexisten múltiples módulos sobre una misma base de datos usando el patrón **Schema-per-Service** para ahorro presupuestal. 

Para evitar que MLflow contaminara el esquema `public`, y para **NO alterar el `search_path` del superadministrador `valuo_admin`** (lo cual afectaría a los otros microservicios), se aprovisionó un usuario de base de datos exclusivo para MLflow:

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

| Variable | Valor Configurado | Propósito |
| :--- | :--- | :--- |
| `BACKEND_STORE_URI` | `postgresql://mlflow_user:MlflowPass2026!@psql-valuo-dev-01.postgres.database.azure.com:5432/db_operational_valuo?sslmode=require` | Conexión con PostgreSQL al esquema dedicado `mlops` |
| `DEFAULT_ARTIFACT_ROOT` | `wasbs://mlflow-artifacts@stvaluodev.blob.core.windows.net/` | Ruta raíz en Azure Blob Storage para guardar modelos serializados |
| `AZURE_STORAGE_CONNECTION_STRING` | `DefaultEndpointsProtocol=https;AccountName=stvaluodev;AccountKey=<KEY>;EndpointSuffix=core.windows.net` | Cadena de autenticación para Blob Storage |
| `AZURE_STORAGE_ACCESS_KEY` | `<KEY>` | Llave simétrica de acceso a Azure Storage |
| `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE` | `true` | **CRÍTICO:** Desactiva el middleware local de FastAPI de MLflow 3.5+ que bloquea peticiones con `403 Forbidden` desde Azure Ingress |
| `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` | `https://mlflow-server.delightfuldesert-51ebdce7.chilecentral.azurecontainerapps.io` | Origen explícito permitido para evitar advertencias de credenciales W3C CORS |

### 4.5. Configuración de Red e Ingress
* **Entrada (Ingress)**: Habilitada
* **Tipo de tráfico**: Aceptación desde cualquier lugar (*Accepting traffic from anywhere*)
* **Tipo**: HTTP / Transporte Automático
* **Conexiones no seguras**: Deshabilitadas (Fuerza HTTPS en puerto 443 administrado por Azure)
* **Puerto de destino**: `5000` (Puerto interno expuesto por MLflow)
* **URL Pública Asignada**: `https://mlflow-server.delightfuldesert-51ebdce7.chilecentral.azurecontainerapps.io/`

### 4.6. Lecciones y Troubleshooting Resueltos en MLflow

1. **Error `Blocked cross-origin request / 403 Forbidden` en búsqueda de modelos y corridas**:
   * *Causa*: En MLflow 3.5.0+, el middleware `fastapi_security` bloquea las llamadas AJAX/POST si el header `Origin` no coincide con `localhost`. Al acceder desde el FQDN de Azure, respondía `403 Forbidden` en `/logged-models/search`.
   * *Solución*: Inyectar `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true` y pasar `--disable-security-middleware` en el arranque del servidor.
2. **Error `relation "online_scoring_configs" does not exist` y `relation "experiments" does not exist`**:
   * *Causa*: Al reiniciar o resetear tablas en PostgreSQL, MLflow 3.x requiere ejecutar las migraciones de Alembic (`mlflow db upgrade`) para crear tablas de scorers y linajes.
   * *Solución*: En el script de arranque del `Dockerfile` se antepuso `mlflow db upgrade ${BACKEND_STORE_URI} && mlflow server ...`.
3. **Advertencia de CORS `disabling credentialed CORS. Wildcard origins with credentials is a CORS spec violation`**:
   * *Causa*: El estándar W3C prohíbe usar comodines (`*`) si la petición envía credenciales/cookies.
   * *Solución*: Quitar el comodín `*` y usar el dominio FQDN explícito o deshabilitar el middleware interno en Azure.
4. **Congelamiento de la interfaz tras promociones o rollbacks (White Screen of Death)**:
   * *Causa*: CPU Throttling severo en instancias con 0.25 vCPU al procesar ráfagas de GraphQL.
   * *Solución*: Elevar los recursos de `mlflow-server` a `1.0 vCPU` y `2.0 GiB de RAM`.

---

## 5. Despliegue de `ml-service` en Azure Container Apps

### 5.1. Justificación de Dimensionamiento y Recursos
* **Memoria y Cómputo**: **`0.5 vCPU` y `1.0 GiB de RAM`**.
  * *Estado en Reposo / Inferencia*: Consume ~250 MB a 320 MB de RAM (FastAPI + 2 modelos XGBoost + SHAP TreeExplainers + índice RNTP en memoria). Tiene más del 65% de holgura.
  * *Pico de Reentrenamiento*: Al descargar el dataset desde PostgreSQL y ajustar los árboles de XGBoost, el consumo pico transitorio alcanza entre 550 MB y 650 MB. 1.0 GiB garantiza un **margen de seguridad del 35%** antes de cualquier intervención del OOM Killer de Kubernetes.
  * *Semáforo de Seguridad*: El servicio cuenta con un semáforo de exclusión mutua (`_training_semaphore`) que impide reentrenar venta y alquiler al mismo tiempo, blindando la memoria.
* **Puerto de Destino 8000**:
  * Es el puerto estándar de FastAPI/Uvicorn, expuesto de forma segura mediante HTTPS en el puerto 443 por Azure Ingress.

### 5.2. Configuración en el Portal de Azure
* **Nombre de la App**: `ml-service`
* **Entorno**: Reutiliza `cae-valuo-dev1`
* **Origen de imagen**: `Docker Hub or other registries` (Privado)
  * Servidor: `acrvaluodev.azurecr.io`
  * Usuario: `acrvaluodev`
  * Imagen y etiqueta: `ml-service:v2` *(Etiqueta inmutable sin modelos locales)*
* **Asignación de recursos**: `0.5 vCPU` / `1.0 GiB RAM`

### 5.3. Variables de Entorno (Runtime)

| Variable | Valor Configurado | Propósito |
| :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | `https://mlflow-server.delightfuldesert-51ebdce7.chilecentral.azurecontainerapps.io` | Conexión HTTPS hacia el servidor MLflow en Azure |
| `DB_HOST` | `psql-valuo-dev-01.postgres.database.azure.com` | Host de Azure PostgreSQL |
| `DB_PORT` | `5432` | Puerto estándar PostgreSQL |
| `DB_USER` | `valuo_admin` | Usuario de conexión a la BD |
| `DB_PASSWORD` | `V@aluo_admin` | Contraseña de conexión |
| `DB_NAME` | `db_dataset_valuo` | Base de datos con tablas de contexto distrital e histórico inmobiliario |
| `MLFLOW_EXPERIMENT_VENTA` | `experimento_tasacion_venta` | Experimento de tasación de venta |
| `MLFLOW_EXPERIMENT_ALQUILER` | `experimento_tasacion_alquiler` | Experimento de tasación de alquiler |
| `MLFLOW_MODEL_NAME_VENTA` | `xgboost_venta` | Nombre del modelo en Model Registry (Venta) |
| `MLFLOW_MODEL_NAME_ALQUILER` | `xgboost_alquiler` | Nombre del modelo en Model Registry (Alquiler) |
| `ADMIN_TOKEN` | *(Vacío)* | Permite acceso libre a endpoints administrativos en fase DEV |

### 5.4. Nuevo Endpoint de Auditoría del Pipeline ETL
Se integró un endpoint administrativo exclusivo para que el administrador supervise las corridas del pipeline de datos:
* **`GET /api/v1/admin/monitoreo/pipeline-ejecuciones`** (opcional: `limit=50`).
* Lee directamente la tabla `public.pipeline_ejecuciones` en `db_dataset_valuo`.
* Reporta: `id`, `tipo_ejecucion`, `ejecutado_por`, `iniciado_en`, `duracion_segundos`, `registros_leidos`, `registros_guardados`, `registros_filtrados`, `tasa_efectividad_pct`, `estado`, `errores` y `metricas_salida`.

### 5.5. Lecciones y Troubleshooting Resueltos en `ml-service`
1. **Modo Cold-Start Resiliente**:
   * *Causa*: Si la imagen no empaqueta archivos `.pkl` locales y MLflow aún no tiene modelos en producción, el servicio antes arrojaba `RuntimeError` y crasheaba en el arranque.
   * *Solución*: `model_loader.py` ahora entra en modo Cold-Start gracefully; `/health` responde `status: degraded`, los endpoints de predicción responden `HTTP 503` explicando la situación, y los endpoints de entrenamiento permanecen 100% operativos para recibir el primer reentrenamiento.
2. **Clean-Up de Binarios Temporales**:
   * *Causa*: Tras cada reentrenamiento, los `.pkl` quedaban ocupando espacio en el contenedor.
   * *Solución*: En `entrenamiento_service.py`, tan pronto el modelo se carga en RAM y se sube a Azure Blob Storage, se ejecuta `os.remove(pkl_path)` para mantener el contenedor limpio.

---

## 6. Checklist de Validación y Verificación

1. **Salud del Servidor MLflow**:
   * Acceder a `https://mlflow-server.delightfuldesert-51ebdce7.chilecentral.azurecontainerapps.io/`.
   * Verificar carga instantánea de experimentos y modelos sin errores 403.
2. **Salud de ML Service (`/health`)**:
   * Verificar respuesta JSON con estado `ok` (o `degraded` si está en cold-start).
3. **Auditoría de Ingesta ETL (`/api/v1/admin/monitoreo/pipeline-ejecuciones`)**:
   * Confirmar lectura de la tabla de ejecuciones del pipeline en `db_dataset_valuo`.
4. **Entrenamiento Inicial y Promoción (`POST /api/v1/admin/entrenamiento/iniciar/venta`)**:
   * Ejecutar con `guardar_como_activo: true`.
   * En ~8 segundos se creará la versión `v1` oficial en MLflow, se guardará en Azure Blob Storage y se cargará en RAM con Hot-Reload.
5. **Inferencia Activa con SHAP (`POST /api/v1/prediccion/venta`)**:
   * Validar estimación en Soles constantes y nominales junto al Top 10 de variables explicativas SHAP y los 3 inmuebles comparables del RNTP.
