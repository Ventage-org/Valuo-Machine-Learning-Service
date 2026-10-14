# Valuo — Machine Learning Service & MLOps Platform

Repositorio oficial del microservicio de inteligencia artificial y plataforma MLOps para la tasación inmobiliaria masiva automatizada de **Valuo** en Lima Metropolitana.

Desarrollado con **FastAPI**, **XGBoost**, **SHAP (TreeExplainer)** y **MLflow Tracking & Model Registry**, orquestado mediante **Docker Compose**.

---

## 1. Arquitectura del Servicio

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCKER COMPOSE ORCHESTRATION                       │
│                                                                             │
│  ┌────────────────────────┐  HTTP:8000  ┌────────────────────────────────┐  │
│  │       ml-service       │ <─────────> │        Backend Principal       │  │
│  │   (FastAPI + XGBoost)  │             │   (NestJS / Spring / Node)     │  │
│  └───────────┬────────────┘             └────────────────────────────────┘  │
│              │                                                              │
│              ├──────────────────────────────┐                               │
│              ▼                              ▼                               │
│  ┌────────────────────────┐     ┌───────────────────────┐                   │
│  │      mlflow-server     │     │ PostgreSQL Contextual │                   │
│  │    (MLflow Registry)   │     │ (inmobiliaria_ml_db)  │                   │
│  │      Port: 5000        │     │      Port: 5432       │                   │
│  └────────────────────────┘     └───────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Componentes:
- **`ml-service/`**: API de alta velocidad para inferencia en tiempo real, explicabilidad SHAP (33 variables), reentrenamiento asíncrono con gobernanza Human-in-the-Loop y hot-reload en memoria RAM.
- **`mlflow-server/`**: Servidor de tracking de experimentos, registro de modelos (`Production`, `Candidate`, `Archived`) y versionamiento de artefactos.
- **`docs/`**: Especificaciones técnicas canónicas de integración, contratos DTOs, métricas IAAO y guías de despliegue.

---

## 2. Estructura del Repositorio

```
Valuo-Machine-Learning-Service/
├── docker-compose.yml              # Orquestación de ml-service + mlflow-server
├── .env.example                    # Plantilla de variables de entorno
├── .gitignore                      # Exclusiones de Git
├── docs/                           # Documentación técnica completa
│   ├── GUIA_ENDPOINTS_ENTRENAMIENTO_Y_MONITOREO.md
│   ├── DOCUMENTACION_REENTRENAMIENTO.md
│   ├── AUDITORIA_NORMAS_TASACION_Y_METRICAS_IAAO.md
│   ├── OPTIMIZACION_TARGET_M2_Y_CALIBRACION_IAAO.md
│   └── PLAN_IMPLEMENTACION_MLFLOW_AZURE.md
├── ml-service/                     # Microservicio FastAPI
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── core/                   # ModelLoader, ModelState (Singleton)
│   │   ├── routers/                # Endpoints (predicción, sensibilidad, monitoreo, etc.)
│   │   ├── schemas/                # Contratos Pydantic
│   │   └── services/               # Lógica de inferencia, SHAP, entrenamiento e IPC
│   ├── config/                     # Configuraciones JSON oficiales e IPC
│   ├── data/                       # Metadata de features y distritos
│   └── models/                     # Modelos XGBoost serializados (.pkl) y métricas
└── mlflow-server/                  # Servidor MLflow independiente
    ├── Dockerfile
    ├── run_server.py
    └── init_schema.sql
```

---

## 3. Inicio Rápido con Docker

### Requisitos previos:
- Docker y Docker Compose instalados.
- Acceso a PostgreSQL (`inmobiliaria_ml_db`) en el host o red de Docker.

### Pasos de ejecución:
```bash
# 1. Configurar variables de entorno
cp .env.example .env

# 2. Levantar la infraestructura
docker compose up -d --build

# 3. Verificar estado de los servicios
docker compose ps
```

- **Swagger UI (FastAPI):** `http://localhost:8000/docs`
- **MLflow Tracking & Registry UI:** `http://localhost:5000`
- **Health Check:** `http://localhost:8000/health`

---

## 4. Endpoints Principales

| Método | Ruta | Acceso | Descripción |
| :--- | :--- | :---: | :--- |
| `POST` | `/api/v1/prediccion/venta` | Público | Inferencia de precio de venta (soles constantes y nominales, intervalo MAPE y las 33 variables SHAP). |
| `POST` | `/api/v1/prediccion/alquiler` | Público | Inferencia de canon de arrendamiento mensual y explicabilidad SHAP. |
| `GET` | `/api/v1/modelos/sensibilidad-factores` | Público | Análisis de sensibilidad de macro-factores para la pantalla principal (Home / Landing). |
| `GET` | `/api/v1/distritos` | Público | Lista los 22 distritos disponibles de Lima Metropolitana. |
| `GET` | `/health` | Público | Health check para balanceadores de carga y sondas Kubernetes. |
| `POST` | `/api/v1/admin/entrenamiento/{venta\|alquiler}` | Admin | Lanza reentrenamiento asíncrono sobre PostgreSQL con MLflow. |
| `GET` | `/api/v1/admin/entrenamiento/estado/{job_id}` | Admin | Polling del progreso del pipeline (pasos 1/8 a 8/8) y métricas obtenidas. |
| `GET` | `/api/v1/admin/modelos/benchmark/{tipo}` | Admin | Comparativa de métricas: Modelo en Producción vs Candidatos de MLflow. |
| `POST` | `/api/v1/admin/modelos/promover` | Admin | Promoción formal de modelo a producción con Hot-Reload en memoria RAM. |
| `GET/PUT` | `/api/v1/admin/config/ipc` | Admin | Consulta y actualización dinámica del factor IPC en caliente. |

---

## 5. Estrategia de Ramas (Git Flow)

- **`main`**: Rama de producción estable. Lista para desplegarse en entornos productivos.
- **`develop`**: Rama de desarrollo activo e integración continua.
- **`azure-deployment`**: Manifiestos de infraestructura como código (Bicep/Terraform), GitHub Actions y configuraciones para Azure Container Apps y Azure Container Registry.

---

## 6. Documentación Detallada

Para consultar especificaciones completas de integración, DTOs y normativas:
- [Guía de Integración Backend y Endpoints de Monitoreo](docs/GUIA_ENDPOINTS_ENTRENAMIENTO_Y_MONITOREO.md)
- [Documentación Técnica del Pipeline de Reentrenamiento](docs/DOCUMENTACION_REENTRENAMIENTO.md)
- [Auditoría de Normas de Tasación y Métricas IAAO](docs/AUDITORIA_NORMAS_TASACION_Y_METRICAS_IAAO.md)
- [Plan de Implementación de MLflow en Azure](docs/PLAN_IMPLEMENTACION_MLFLOW_AZURE.md)
