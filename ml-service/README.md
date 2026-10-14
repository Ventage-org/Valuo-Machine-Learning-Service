# ML Service — Tasación Inmobiliaria Lima

Microservicio FastAPI de predicción de precios inmobiliarios con XGBoost + explicabilidad SHAP.

**Modelo**: XGBoost v2 | **MAPE**: 15.04% | **R²**: 0.7469 | **22 distritos** de Lima Metropolitana

---

## Requisitos

- Python 3.10+
- Los artefactos del modelo (ver sección **Configuración de artefactos**)

## Instalación

```bash
cd ml-service/
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

## Configuración de artefactos

El servicio necesita los siguientes archivos (no incluidos en el repo por su tamaño):

```
ml-service/
├── models/
│   └── xgboost_venta_v2.pkl           ← del directorio python_modelo_tesis/models/
├── data/
│   ├── features_metadata.json         ← de python_modelo_tesis/data/processed/
│   └── distrito_contexto_ref.csv      ← de python_modelo_tesis/data/processed/
└── config/
    └── model_config.json              ← incluido en el repo ✓
```

Copiar desde el proyecto de entrenamiento (Windows PowerShell):

```powershell
Copy-Item ..\models\xgboost_venta_v2.pkl .\models\
Copy-Item ..\data\processed\features_metadata.json .\data\
Copy-Item ..\data\processed\distrito_contexto_ref.csv .\data\
```

## Variables de entorno

```bash
cp .env.example .env
# Editar .env si los artefactos están en rutas distintas
```

Parámetros clave en `.env`:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODEL_PATH` | `models/xgboost_venta_v2.pkl` | Ruta al modelo serializado |
| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODEL_PATH` | `models/xgboost_venta_v2.pkl` | Ruta al modelo de venta |
| `METADATA_PATH` | `data/features_metadata.json` | Metadata y encoding de venta |
| `CONFIG_PATH` | `config/model_config.json` | Configuración oficial de venta |
| `MODEL_ALQUILER_PATH` | `models/xgboost_alquiler_v1.pkl` | Ruta al modelo de alquiler |
| `METADATA_ALQUILER_PATH` | `data/features_metadata_alquiler.json` | Metadata y encoding de alquiler |
| `CONFIG_ALQUILER_PATH` | `config/model_alquiler_config.json` | Configuración oficial de alquiler |
| `DB_HOST` | `localhost` | Host de PostgreSQL (`inmobiliaria_ml_db`) |
| `DB_PORT` | `5432` | Puerto de PostgreSQL |
| `DB_NAME` | `inmobiliaria_ml_db` | Base de datos |
| `DB_USER` | `postgres` | Usuario |
| `DB_PASSWORD` | `...` | Contraseña |
| `ADMIN_TOKEN` | (opcional) | Token para endpoints `/admin/entrenamiento/*` |

## Arrancar el servidor

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Documentación interactiva: http://localhost:8000/docs

---

## Endpoints Principales

### 1. Inferencia y Pantalla Principal (Públicos)
- **`POST /api/v1/prediccion/venta`**: Predice precio de venta de un inmueble (soles constantes y nominales, intervalo de confianza ±15.04%, explicabilidad SHAP).
- **`POST /api/v1/prediccion/alquiler`**: Predice canon de arrendamiento mensual (soles constantes y nominales, intervalo de confianza ±13.85%, explicabilidad SHAP).
- **`GET /api/v1/modelos/sensibilidad-factores`**: Análisis de sensibilidad de macro-factores del modelo para la pantalla principal (Home / Landing Page). Admite `?tipo_operacion=venta` o `alquiler`.
- **`GET /api/v1/distritos`**: Lista los 22 distritos disponibles de Lima Metropolitana.
- **`GET /health`**: Health check para balanceadores.

### 2. Administración y Reentrenamiento
- **`POST /api/v1/admin/entrenamiento/venta`**: Lanza pipeline de reentrenamiento del modelo de venta consumiendo `dataset_inmuebles_venta` desde PostgreSQL.
- **`POST /api/v1/admin/entrenamiento/alquiler`**: Lanza pipeline de reentrenamiento del modelo de alquiler consumiendo `dataset_inmuebles_alquiler` desde PostgreSQL con ponderación temporal E1.
- **`GET /api/v1/admin/entrenamiento/estado/{job_id}`**: Consulta progreso en tiempo real de un job (1/8 a 8/8) y métricas obtenidas.
- **`GET /api/v1/admin/entrenamiento/jobs`**: Historial de jobs ejecutados en la sesión.

---

### Ejemplo de Predicción de Venta / Alquiler

**Request (`POST /api/v1/prediccion/venta` o `/api/v1/prediccion/alquiler`):**
```json
{
  "distrito": "San Miguel",
  "superficie": 75.0,
  "habitaciones": 3,
  "banios": 2,
  "garajes": 1,
  "piso": 5,
  "antiguedad": 8,
  "vista_exterior": true
}
```
> **Nota:** `anio` y `trimestre` son opcionales. Si se omiten, el backend los calcula automáticamente con la fecha del sistema.

**Response (200):**
```json
{
  "distrito": "San Miguel",
  "superficie_m2": 75.0,
  "prediccion": {
    "soles_constantes": 287000.0,
    "soles_nominales": 485416.0,
    "precio_m2_constantes": 3826.67,
    "precio_m2_nominales": 6472.21
  },
  "intervalo_confianza": {
    "inferior_nominales": 412000.0,
    "superior_nominales": 558000.0,
    "inferior_constantes": 243780.0,
    "superior_constantes": 330220.0,
    "mape_pct": 15.04
  },
  "explicabilidad": {
    "valor_base_constantes": 432295.69,
    "contribuciones": [
      {
        "feature": "distrito_encoded",
        "label": "Distrito (San Miguel)",
        "shap_value": -0.198,
        "valor_feature": 287982.23,
        "impacto": "negativo",
        "peso_porcentaje": 31.2,
        "nivel_impacto": "Impacto Muy Alto"
      },
      {
        "feature": "Superficie",
        "label": "Superficie (75.0 m²)",
        "shap_value": 0.165,
        "valor_feature": 75.0,
        "impacto": "positivo",
        "peso_porcentaje": 12.8,
        "nivel_impacto": "Impacto Alto"
      },
      {
        "feature": "pct_NSE_A",
        "label": "NSE A en zona",
        "shap_value": 0.082,
        "valor_feature": 1205.5,
        "impacto": "positivo",
        "peso_porcentaje": 7.7,
        "nivel_impacto": "Impacto Moderado"
      },
      {
        "feature": "Antiguedad",
        "label": "Antigüedad (8 años)",
        "shap_value": -0.063,
        "valor_feature": 8.0,
        "impacto": "negativo",
        "peso_porcentaje": 4.7,
        "nivel_impacto": "Impacto Leve"
      },
      "... [33 variables en total para persistencia en BD; la suma de peso_porcentaje es 100.0%]"
    ],
    "n_features_mostradas": 33,
    "n_features_total": 33,
    "nota": "'peso_porcentaje' suma exactamente 100.0% entre la totalidad de variables del modelo."
  },
  "modelo": {
    "version": "v2",
    "mape_test": 15.04,
    "r2_test": 0.7469,
    "ipc_factor": 169.18,
    "periodo_ipc": "2026-Q1",
    "train_periodo": "2016-2023",
    "test_periodo": "2024-2025"
  }
}
```

**Error 422 — Distrito no disponible:**
```json
{
  "error": "distrito_no_disponible",
  "message": "El distrito 'Baños de Chosica' no está disponible en el modelo actual.",
  "distritos_disponibles": ["Ate Vitarte", "Barranco", ...],
  "total_distritos": 22
}
```

---

## ¿Cómo usa el frontend los valores SHAP?

El campo `explicabilidad.contribuciones` está ordenado por `|shap_value|` descendente y contiene la **totalidad de las 33 variables del modelo** para persistencia íntegra en la base de datos del backend.

Cada item incluye:
- **`feature`**: Nombre técnico interno de la feature (ej: `distrito_encoded`, `Superficie`, `pct_NSE_A`).
- **`label`**: Texto legible para la interfaz de usuario (ej: `Distrito (San Miguel)`, `Superficie (75.0 m²)`).
- **`shap_value`**: Magnitud del impacto en escala logarítmica (escala interna de entrenamiento).
- **`valor_feature`**: Valor real de la característica para el inmueble evaluado.
- **`impacto`**: Dirección del impacto (`positivo` → empuja el precio hacia arriba; `negativo` → hacia abajo; `neutro`).
- **`peso_porcentaje`**: Peso relativo de la variable sobre el total del modelo. La suma de las 33 variables es **exactamente 100.0%**.
- **`nivel_impacto`**: Categoría cualitativa lista para renderizar en badges o chips de la UI:
  - $\ge 20\% \rightarrow$ **`Impacto Muy Alto`** (ej: Distrito, ~31%)
  - $10\% - 19.9\% \rightarrow$ **`Impacto Alto`** (ej: NSE A, ~13%)
  - $5\% - 9.9\% \rightarrow$ **`Impacto Moderado`** (ej: Superficie, Periodo, ~6-8%)
  - $< 5\% \rightarrow$ **`Impacto Leve`** (ej: Antigüedad, Cocheras, Pisos, <5%)

### Estrategia recomendada para Backend y Frontend:

```
                  ┌──────────────────────────────────────────────┐
                  │    POST /api/v1/prediccion/venta (o /alquiler)│
                  └──────────────────────┬───────────────────────┘
                                         │ Retorna 33 variables
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │              Backend Principal               │
                  │   - Guarda las 33 variables en PostgreSQL    │
                  └──────────────────────┬───────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
│     Vista Inicial (Top 10 variables)  │ │ Botón "Mostrar todas las variables"   │
│ - contribuciones.slice(0, 10)         │ │ - contribuciones.slice(10)            │
│ - Explica el 83.5% del impacto        │ │ - Completa el 16.5% restante (100.0%) │
│ - Barras de progreso limpias          │ │ - Despliegue en acordeón o modal      │
└───────────────────────────────────────┘ └───────────────────────────────────────┘
```

1. **Almacenamiento (Backend):** El backend almacena el array `contribuciones` completo (las 33 variables) directamente en la base de datos (columna `JSONB` de valuaciones o tabla hija).
2. **Vista Inicial (Top 10):** En la pantalla principal o tarjeta de resultados, el frontend muestra los primeros 10 elementos (`contribuciones.slice(0, 10)`), que representan aproximadamente el **83.5%** del impacto del precio.
3. **Botón "Mostrar todas las variables":** Un botón interactivo permite al usuario desplegar las 23 variables secundarias (`contribuciones.slice(10)`), completando el **16.5%** restante hasta el 100.0%.

El `valor_base_constantes` es el precio de partida (media global del modelo).
La suma `valor_base + Σshap_values` ≈ `log1p(precio_constantes)`.

---

## Análisis de Sensibilidad de Factores (Pantalla Principal / Home)

Para la tarjeta informativa o explicativa en la pantalla principal del frontend, el servicio expone:

**`GET /api/v1/modelos/sensibilidad-factores?tipo_operacion=venta`** (o `alquiler`)

**Response (200 OK):**
```json
{
  "modelo": "xgboost_venta",
  "tipo_operacion": "venta",
  "version": "v2",
  "metodo": "Análisis de Sensibilidad por Ganancia de Información (XGBoost Feature Attribution)",
  "periodo_referencia": "2016-2023",
  "factores_principales": [
    {
      "id": "ubicacion_entorno",
      "factor": "Ubicación y Entorno Urbano",
      "descripcion": "Distrito, estrato socioeconómico (NSE) y cercanía a vías principales, transporte, parques y servicios.",
      "peso_porcentaje": 53.1,
      "nivel_impacto": "Impacto Muy Alto",
      "variables_incluidas": ["distrito_encoded", "pct_NSE_A", "distancia_centro_km", "dist_estacion_transporte_km", "..."]
    },
    {
      "id": "superficie_espacio",
      "factor": "Área y Superficie Construida",
      "descripcion": "Metraje total en m², distribución espacial y número de ambientes (habitaciones y baños).",
      "peso_porcentaje": 33.4,
      "nivel_impacto": "Impacto Muy Alto",
      "variables_incluidas": ["Superficie", "Habitaciones", "Banios", "ratio_banio_hab"]
    },
    {
      "id": "estacionamiento_cocheras",
      "factor": "Estacionamiento y Accesibilidad",
      "descripcion": "Disponibilidad de cocheras y plazas de estacionamiento vehicular techado.",
      "peso_porcentaje": 13.1,
      "nivel_impacto": "Impacto Moderado",
      "variables_incluidas": ["tiene_garaje", "Garajes"]
    },
    {
      "id": "antiguedad_conservacion",
      "factor": "Antigüedad y Conservación",
      "descripcion": "Años de edificación del inmueble, depreciación física y condición de estreno.",
      "peso_porcentaje": 0.4,
      "nivel_impacto": "Impacto Leve",
      "variables_incluidas": ["Antiguedad", "es_estreno"]
    }
  ],
  "cobertura_macro_factores_pct": 100.0,
  "nota": "Los porcentajes representan la sensibilidad promedio de mercado en Lima Metropolitana..."
}
```

---

## Reentrenamiento Asíncrono (Admin)

Para consultar en profundidad la arquitectura, pasos del pipeline (1 a 8), ponderación temporal E1 para alquiler, hot-reload y ejemplos de integración con cURL y Python, consulta la guía dedicada:
- [Documentación Técnica de Reentrenamiento](DOCUMENTACION_REENTRENAMIENTO.md)

---

## Notas metodológicas para la tesis

- **Soles Constantes**: escala interna del modelo (Base dic 2009 = 100). Elimina distorsión inflacionaria.
- **Soles Nominales**: `Precio_Const × (IPC_2026-Q1 / 100)` = valor en moneda actual.
- **IPC 2026-Q1 = 169.18**: configurable en `config/model_config.json` sin recompilar ni redesplegar.
- **SHAP (SHapley Additive exPlanations)**: cada `shap_value` indica cuánto contribuyó esa variable (en escala logarítmica) a que el precio suba o baje respecto a la media de entrenamiento. Alineado con US-18 de explicabilidad del sistema.
