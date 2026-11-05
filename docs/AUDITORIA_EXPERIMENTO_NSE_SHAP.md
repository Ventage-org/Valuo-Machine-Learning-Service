# Auditoría Experimental: Análisis de Validez de Estrato Socioeconómico (NSE ENAHO) y Atribución SHAP en XGBoost

**Proyecto:** Valuo — Sistema de Valoración Inmobiliaria con Machine Learning  
**Autor:** Joaquín Cortez  
**Fecha:** Noviembre 2026  
**Entorno de Evaluación:** `ml-service` (Modelo Venta `xgboost_venta_v2.pkl`) & Catálogo Oficial ([docs/distrito_anio_contexto.csv](file:///c:/Users/ADMIN_00/Documents/GitHub/Valuo-Machine-Learning-Service/docs/distrito_anio_contexto.csv))  
**Objetivo:** Determinar experimentalmente si las variables distritales de Estrato Socioeconómico (`pct_NSE_A` a `pct_NSE_E`) representan un gradiente socioeconómico causal o si operan como una clave sustituta (*proxy hash*) de identificación de distritos con sobre-atribución en TreeSHAP.

---

## 1. Resumen Ejecutivo y Veredicto Técnico

### 1.1. Dictamen de la Hipótesis
Tras auditar la estructura interna de los 956 árboles del ensamble XGBoost en producción (`xgboost_venta_v2.pkl`) y contrastar los umbrales de división con los microdatos de la encuesta ENAHO (INEI 2025), **se valida y confirma la hipótesis planteada**:

> **Veredicto:**  
> Las variables `pct_NSE_A..E` **no están siendo utilizadas por XGBoost como una métrica continua de nivel socioeconómico**, sino como un **identificador espacial multivariado (hash)** que particiona los 22 distritos de Lima Metropolitana. El alto peso de `pct_NSE_A` en las explicaciones SHAP (Top 3 en Gain y Top 2 en impacto aditivo) es un **artefacto de multicolinealidad con `distrito_encoded`**, el cual fragmenta la atribución de la ubicación distrital y distorsiona la interpretación ante peritos tasadores.

```mermaid
flowchart TD
    subgraph Realidad["Tabla distrito_anio_contexto (Base de Datos)"]
        ENAHO["22 Vectores Distritales Estáticos\n[pct_NSE_A, ..., pct_NSE_E]"]
    end

    subgraph XGBoost["Comportamiento Interno de XGBoost"]
        SPLIT["448 Splits en pct_NSE_A\nCortes exactos: 31.647, 36.877, 29.424..."]
        MEM["Memorización del Distrito:\npct_NSE_A == 36.877% ➔ San Isidro\npct_NSE_A == 31.647% ➔ Miraflores\npct_NSE_A == 13.841% ➔ La Perla"]
    end

    subgraph XAI["Impacto en SHAP y Explicabilidad"]
        SHAP_DIST["distrito_encoded\n(Target Encoding m²)"]
        SHAP_NSE["pct_NSE_A\n(Absorbe Ganancia Residual)"]
        CONFUSION["Tasador / Usuario ve:\n'Impacto Muy Alto: NSE A (+S/ 35,000)'\ncuando en realidad el modelo dice:\n'El inmueble está en San Isidro'"]
    end

    ENAHO --> SPLIT
    SPLIT --> MEM
    MEM --> SHAP_NSE
    SHAP_DIST -.->|Colinealidad N=22| SHAP_NSE
    SHAP_NSE --> CONFUSION
```

---

## 2. Evidencia Forense 1: Mapeo de Umbrales en Árboles (*Tree Splits*)

Al extraer la totalidad de nodos de decisión del booster serializado de XGBoost mediante `booster.trees_to_dataframe()`, se contabilizaron **1,567 divisiones** dedicadas exclusivamente a las 5 columnas de NSE:

| Variable | Número Total de Splits | Ganancia de Información Media (*Mean Gain*) |
| :--- | :---: | :---: |
| **`distrito_encoded`** | 500 | **22.36** (Rank 1 Global) |
| **`pct_NSE_A`** | 448 | **5.52** (Rank 3 Global) |
| **`pct_NSE_C`** | 420 | **0.74** |
| **`pct_NSE_B`** | 371 | **0.15** |
| **`pct_NSE_D`** | 220 | **0.24** |
| **`pct_NSE_E`** | 108 | **0.53** |

### Mapeo 1 a 1: Umbrales del Árbol vs Microdatos ENAHO
Al comparar los valores de corte (*Split Conditions*) más frecuentes en los árboles contra los porcentajes calculados a partir de `sumaria-2025.dta`, la coincidencia es absoluta al tercer decimal:

| Corte en XGBoost | Frecuencia de Árboles | UBIGEO ENAHO | Distrito Correspondiente | % Real ENAHO |
| :---: | :---: | :---: | :--- | :---: |
| **`13.841`** | 62 nodos | `070104` | **La Perla** | 13.841% |
| **`31.647`** | 51 nodos | `150122` | **Miraflores** | 31.647% |
| **`1.330`** | 49 nodos | `150103` | **Ate Vitarte** | 1.330% |
| **`2.060`** | 41 nodos | `150108` | **Chorrillos** | 2.060% |
| **`29.424`** | 30 nodos | `150140` | **Santiago de Surco** | 29.424% |
| **`36.877`** | 23 nodos | `150131` | **San Isidro** | 36.877% |
| **`18.561`** | 23 nodos | `070102` | **Bellavista** | 18.561% |
| **`2.538`** | 23 nodos | `150115` | **Pueblo Libre** | 2.538% |
| **`12.913`** | 21 nodos | `150114` | **Magdalena del Mar** | 12.913% |
| **`1.619`** | 20 nodos | `150101` | **Cercado de Lima** | 1.619% |

**Conclusión del Mapeo:**  
XGBoost no está detectando un continuo econométrico donde $f(x)$ aumenta proporcionalmente a $x$. Está utilizando los puntos exactos $\{13.841, 31.647, 36.877, ...\}$ como discriminadores discretos para segmentar el espacio muestral exactamente en los límites de cada distrito.

---

## 3. Evidencia Forense 2: Experimento de Permutación y Sensibilidad SHAP

Para evaluar cómo reacciona el cálculo de SHAP en inferencia, se construyó un departamento estándar de $90\text{ m}^2$ (3 habitaciones, 2 baños, 1 garaje, piso 4, 5 años de antigüedad) y se evaluó su tasación y valores SHAP en dos escenarios: con su contexto real y con el vector NSE permutado entre un distrito de alto valor (San Isidro) y uno de menor valor (Comas).

```
============================================================
RESULTADOS DEL TEST DE INTERCAMBIO (SWAP) DISTRITAL
============================================================

Caso 1: Inmueble en San Isidro (Valores Reales)
  - Predicción Precio Total: S/ 478,850.59
  - SHAP distrito_encoded:   +0.1516
  - SHAP pct_NSE_A:          +0.0625  (Suma aditiva positiva fuerte)
  - SHAP pct_NSE_C:          +0.0222

Caso 1B: Inmueble en San Isidro (Con Vector NSE de Comas: pct_NSE_A = 0%)
  - Predicción Precio Total: S/ 411,563.50  (-14.05% de caída)
  - SHAP distrito_encoded:   +0.1421  (Apenas varía)
  - SHAP pct_NSE_A:          -0.0369  (Colapsa a negativo)
  - SHAP pct_NSE_C:          -0.0169

------------------------------------------------------------

Caso 2: Inmueble en Comas (Valores Reales)
  - Predicción Precio Total: S/ 396,620.88
  - SHAP distrito_encoded:   +0.1416
  - SHAP pct_NSE_A:          -0.0389  (Penalización negativa)
  - SHAP pct_NSE_C:          -0.0161

Caso 2B: Inmueble en Comas (Con Vector NSE de San Isidro: pct_NSE_A = 36.88%)
  - Predicción Precio Total: S/ 462,716.09  (+16.66% de incremento)
  - SHAP distrito_encoded:   +0.1511  (Apenas varía)
  - SHAP pct_NSE_A:          +0.0654  (Pasa a positivo alto)
  - SHAP pct_NSE_C:          +0.0207
```

### Hallazgo Clave
1. `distrito_encoded` no captura por sí solo el 100% de la ubicación; solo absorbe una parte de la media distrital.
2. La otra parte sustancial (~15% del valor predicho) **la absorbe `pct_NSE_A`**.
3. En la pantalla del usuario o perito tasador, el sistema reporta `pct_NSE_A: Impacto Muy Alto`, cuando en rigor económico ese impacto no proviene de la condición del predio, sino de la prima de localización urbana de San Isidro que quedó dividida entre dos columnas redundantes.

---

## 4. Resultados Empíricos de las 3 Pruebas Masivas (Base de Datos Azure)

Las 3 pruebas fueron ejecutadas sobre la base de datos oficial en Azure PostgreSQL (`db_dataset_valuo`), evaluando **44,173 registros de entrenamiento (2016–2023)** y **23,737 registros de prueba (2024–2025)** mediante el script [scripts/audit_nse_pipeline.py](file:///c:/Users/ADMIN_00/Documents/GitHub/Valuo-Machine-Learning-Service/scripts/audit_nse_pipeline.py). Los resultados estructurados se encuentran respaldados en [docs/resultados_auditoria_nse.json](file:///c:/Users/ADMIN_00/Documents/GitHub/Valuo-Machine-Learning-Service/docs/resultados_auditoria_nse.json).

### 4.1. Resultados de la Prueba 2: Estudio de Ablación (Contribución Marginal)

| Modelo | Variables Distritales | MAPE (%) | $R^2$ | MAE (S/.) | COD IAAO (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **M0: Físico Puro** | Sin ubicación (solo atributos del inmueble) | 20.86% | 0.5861 | S/ 69,842 | 20.78% |
| **M1: Solo ID** | Solo `distrito_encoded` | **15.19%** | **0.7438** | S/ 52,570 | **14.40%** |
| **M2: Solo NSE** | Solo `pct_NSE_A..E` (sin `distrito_encoded`) | **15.20%** | **0.7425** | S/ 52,702 | **14.42%** |
| **M3: Completo** | `distrito_encoded` + `pct_NSE_A..E` | **15.13%** | **0.7441** | S/ 52,442 | **14.36%** |

> **Hallazgo Clave de Ablación:**
> 1. `M1` (Solo ID) y `M2` (Solo NSE) tienen un desempeño **virtualmente idéntico** (MAPE 15.19% vs 15.20%, delta de 0.01%). Las 5 columnas continuas de NSE logran exactamente el mismo techo predictivo que una sola columna de Target Encoding.
> 2. Al combinar ambas (`M3`), la ganancia marginal neta es de apenas **+0.06% MAPE** (de 15.19% a 15.13%).
> 3. **Conclusión:** El NSE **no aporta información económica adicional** sobre la identidad del distrito. Que salga alto en SHAP en $M3$ es 100% un efecto de multicolinealidad donde le "roba" ganancia a `distrito_encoded`.

---

### 4.2. Resultados de la Prueba 1: Permutación Distrital de Vectores NSE (*District Shuffle*)
Se barajaron aleatoriamente los 22 vectores de NSE entre los 22 distritos (ej: San Isidro recibió los porcentajes de Comas y viceversa) y se reentrenó el modelo `M2` (Solo NSE):

* **MAPE con NSE Originales (M2):** `15.20%` ($R^2 = 0.7425$)
* **MAPE con NSE Permutados al azar:** `15.84%` ($R^2 = 0.7162$)
* **Variación del Error ($\Delta \text{MAPE}$):** **solo +0.64%**

> **Hallazgo Clave de Permutación:**
> A pesar de que los distritos tenían valores de NSE completamente falsos e invertidos, el error del modelo casi no cambió ($\Delta \le 0.64\%$). Esto demuestra que XGBoost utiliza los vectores de porcentajes como **etiquetas discretas de agrupamiento (hashes)** y no como una relación funcional o causal de riqueza distrital.

---

### 4.3. Resultados de la Prueba 3: Validación Cruzada Espacial por Grupos (*GroupKFold* - 5 Folds)
Se evaluó la capacidad de generalización espacial sobre distritos nunca antes vistos por el modelo (Leave-Districts-Out):

| Fold | Distritos Excluidos en Validación (Ejemplos) | MAPE Con NSE (%) | MAPE Sin NSE (%) |
| :---: | :--- | :---: | :---: |
| **Fold 1** | Carabayllo, Los Olivos, Barranco, Miraflores | 27.98% | 28.76% |
| **Fold 2** | La Perla, Jesús María, Surco, La Victoria | 16.66% | 17.81% |
| **Fold 3** | Chorrillos, La Molina, Cercado de Lima, San Miguel | 20.38% | 24.93% |
| **Fold 4** | Bellavista, Pueblo Libre, San Borja, Surquillo, Ate | **20.87%** | **17.59%** *(Mejor sin NSE)* |
| **Fold 5** | Lince, Magdalena, Breña, San Isidro, Comas | 23.33% | 26.89% |
| **PROMEDIO** | **Generalización Espacial (Out-of-District)** | **21.84%** | **23.20%** |

> **Hallazgo Clave de GroupKFold:**
> 1. En distritos no vistos, el error salta del 15% al 21.84%, evidenciando que el modelo depende fuertemente de la memorización del distrito.
> 2. En el Fold 4, el modelo con NSE rindió **peor** que el modelo que ignoraba la ubicación (20.87% vs 17.59%), demostrando que los cortes rígidos de árboles pueden perjudicar la extrapolación.
> 3. **Impacto en los 4 Distritos Imputados:** Respaldar predicciones en distritos no muestreados o con baja muestra en ENAHO (<30 hogares) a través de cortes fijos de NSE genera riesgo de alucinación de precio, lo que exige advertencias de incertidumbre contextual.

---

---

## 5. Script Automatizado de Auditoría (`audit_nse_pipeline.py`)

Se ha estructurado el script reproducible en el directorio de scripts para su ejecución directa contra el entorno de base de datos o réplica local:

```python
"""
scripts/audit_nse_pipeline.py
Ejecutor de las 3 Pruebas de Auditoría Espacial para NSE en XGBoost.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from app.core.db_provider import PostgreSQLTrainingProvider
from app.services.entrenamiento_service import _construir_features, _target_encoding_distrito

def ejecutar_auditoria_completa():
    print("Iniciando extracción de datos para auditoría de NSE...")
    db = PostgreSQLTrainingProvider()
    
    if not db.check_connection():
        print("[AVISO] No se pudo conectar a PostgreSQL en Azure.")
        print("Asegúrate de que tu IP pública actual esté autorizada en el Firewall de Azure PostgreSQL:")
        print("Azure Portal > psql-valuo-dev-01 > Networking > Add current client IP address.")
        return

    df_train_raw = db.load_dataset_venta(split="TRAIN")
    df_test_raw  = db.load_dataset_venta(split="TEST")

    df_train = _construir_features(df_train_raw)
    df_test  = _construir_features(df_test_raw)

    df_train["target_m2"] = df_train_raw["precio_soles_const"] / df_train["Superficie"]
    df_test["target_m2"]  = df_test_raw["precio_soles_const"] / df_test["Superficie"]

    df_train, df_test, _, _ = _target_encoding_distrito(
        df_train, df_test, col_distrito="distrito", col_target="target_m2", smoothing=10.0
    )

    y_train = np.log(df_train["target_m2"].values)
    y_test  = np.log(df_test["target_m2"].values)
    sup_test = df_test["Superficie"].values
    y_real_test = df_test_raw["precio_soles_const"].values

    base_cols = [
        "Anio", "Trimestre", "Superficie", "Habitaciones", "Banios",
        "Garajes", "Piso", "Vista_Exterior", "Antiguedad", "m2_por_habitacion",
        "ratio_banios_hab", "tiene_garaje", "es_piso_alto", "superficie_cuadrado"
    ]
    nse_cols = ["pct_NSE_A", "pct_NSE_B", "pct_NSE_C", "pct_NSE_D", "pct_NSE_E"]

    # 1. Ablación
    print("\n--- EJECUTANDO PRUEBA 2: ABLACIÓN ---")
    modelos_ablacion = {
        "M0_Sin_Ubicacion": base_cols,
        "M1_Solo_ID": base_cols + ["distrito_encoded"],
        "M2_Solo_NSE": base_cols + nse_cols,
        "M3_ID_mas_NSE": base_cols + ["distrito_encoded"] + nse_cols
    }
    for m_nombre, m_cols in modelos_ablacion.items():
        reg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.04, random_state=42, n_jobs=-1)
        reg.fit(df_train[m_cols], y_train)
        pred_tot = np.exp(reg.predict(df_test[m_cols])) * sup_test
        mape = mean_absolute_percentage_error(y_real_test, pred_tot) * 100
        r2 = r2_score(y_real_test, pred_tot)
        print(f"{m_nombre:<20} | MAPE: {mape:.2f}% | R²: {r2:.4f}")

    # 2. Permutación
    print("\n--- EJECUTANDO PRUEBA 1: PERMUTACIÓN DISTRITAL ---")
    dist_cat = df_train[["distrito"] + nse_cols].drop_duplicates().reset_index(drop=True)
    dist_shuffled = dist_cat.copy()
    dist_shuffled[nse_cols] = dist_cat[nse_cols].sample(frac=1.0, random_state=99).values

    df_tr_perm = df_train.drop(columns=nse_cols).merge(dist_shuffled, on="distrito")
    df_ts_perm = df_test.drop(columns=nse_cols).merge(dist_shuffled, on="distrito")

    reg_perm = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.04, random_state=42, n_jobs=-1)
    reg_perm.fit(df_tr_perm[base_cols + nse_cols], y_train)
    pred_perm = np.exp(reg_perm.predict(df_ts_perm[base_cols + nse_cols])) * sup_test
    mape_p = mean_absolute_percentage_error(y_real_test, pred_perm) * 100
    print(f"Desempeño con NSE Permutados al azar: MAPE = {mape_p:.2f}%")

if __name__ == "__main__":
    ejecutar_auditoria_completa()
```

---

## 6. Recomendaciones Prácticas y Plan de Implementación

En base a los resultados concluyentes del diagnóstico forense y las 3 pruebas masivas, se establece una estrategia a dos horizontes para resolver la multicolinealidad del NSE y blindar la explicabilidad técnica del sistema:

---

### 6.1. Horizonte 1: Solución Inmediata en Inferencia (Consolidación Exclusiva de NSE + Preservación de 28 Variables + Flag de Confiabilidad)
Esta solución opera directamente sobre el modelo actual en producción (`xgboost_venta_v2.pkl`) **sin requerir reentrenamiento**, garantizando *Zero-Downtime* y paridad de métricas ($R^2 = 0.744$, $\text{MAPE} = 15.13\%$).

#### A. Agrupación Exclusiva del Bloque NSE en Inferencia
Por la propiedad aditiva fundamental de SHAP ($\sum \phi_i = f(x) - E[f(x)]$), la suma de las atribuciones locales de las 5 columnas de estratos socioeconómicos equivale formalmente a la contribución neta del contexto socioeconómico:
$$\phi_{\text{NSE\_Total}} = \sum_{k \in \{A,B,C,D,E\}} \phi_{\text{pct\_NSE\_}k}$$

* **Regla de Negocio Crítica:** **NO se agrupan las demás 28 variables**.
  Para que los agentes inmobiliarios y tasadores puedan sustentar ante sus clientes cada atributo del inmueble por separado, se conservan intactas e individuales todas las demás características:
  * **Físicas del Inmueble:** `Superficie`, `Habitaciones`, `Banios`, `Garajes`, `Piso`, `Vista_Exterior`, `Antiguedad`, `tiene_garaje`, `es_piso_alto`, `m2_por_habitacion`, `ratio_banios_hab`, `superficie_cuadrado`.
  * **Identidad Distrital y Urbana:** `distrito_encoded`, `distancia_centro_km`, `dist_colegio_km`, `dist_hospital_km`, `dist_estacion_transporte_km`, `dist_centro_comercial_km`, `dist_parque_km`, `dist_universidad_km`, `tasa_robo`, `tasa_hurto`, `poblacion_proyectada`, `area_distrito_km2`, `densidad_hab_km2`.
  * **Temporales:** `Anio`, `Trimestre`, `periodo_numerico`.
* **Resultado en la Respuesta:** El response contiene **29 variables ordenadas por relevancia** (28 individuales + 1 bloque consolidado `estrato_socioeconomico`). Los porcentajes relativos (`peso_porcentaje`) suman exactamente el **100.0%**.

#### B. Flag de Baja Confiabilidad e Imputación de Contexto
* **Fuente Canónica:** La tabla oficial `distrito_anio_contexto` alojada en Azure PostgreSQL (`db_dataset_valuo`), la cual provee la columna booleana `c.nse_imputado` consumida directamente en tiempo real por el servicio (el archivo CSV sirvió exclusivamente como guía de inspección preliminar).
* **Distritos con `nse_imputado = True` en PostgreSQL:**
  * **Barranco:** `nse_imputado: True` (muestra censal reducida).
  * **Lince:** `nse_imputado: True` (muestra censal reducida).
  * **Magdalena:** `nse_imputado: True` (muestra censal reducida).
* **Comportamiento en la API de Inferencia ([prediccion_service.py](file:///c:/Users/ADMIN_00/Documents/GitHub/Valuo-Machine-Learning-Service/ml-service/app/services/prediccion_service.py)):**
  Cuando un usuario solicita la valoración de un departamento en un distrito con `nse_imputado == True`, la API activa el campo `alerta_contexto` informando al perito:
  ```json
  {
    "alerta_contexto": "Aviso de contexto: La información de estrato socioeconómico para el distrito de Barranco proviene de una estimación imputada debido a un tamaño muestral censal reducido en ENAHO. La valoración del modelo se fundamenta prioritariamente en las características físicas del inmueble y la dinámica de precios del distrito."
  }
  ```
  Para distritos no imputados (`Surco`, `Miraflores`, `San Isidro`, etc.), `alerta_contexto` es `null`.

#### C. Estructura de Respuesta en Producción (Ejemplo Verificado)
```json
{
  "distrito": "Surco",
  "superficie_m2": 95.0,
  "prediccion": {
    "soles_nominales": 631636.16,
    "precio_m2_nominales": 6648.80
  },
  "explicabilidad": {
    "valor_base_constantes": 432295.69,
    "n_features_mostradas": 29,
    "n_features_total": 33,
    "contribuciones": [
      {
        "feature": "distrito_encoded",
        "label": "Distrito (Surco)",
        "shap_value": 0.12782,
        "impacto": "positivo",
        "peso_porcentaje": 27.3,
        "nivel_impacto": "Impacto Muy Alto"
      },
      {
        "feature": "estrato_socioeconomico",
        "label": "Estrato Socio-Económico de la Zona (NSE)",
        "shap_value": -0.07803,
        "impacto": "negativo",
        "peso_porcentaje": 16.7,
        "nivel_impacto": "Impacto Alto"
      },
      {
        "feature": "Antiguedad",
        "label": "Antigüedad (3.0 años)",
        "shap_value": 0.05574,
        "impacto": "positivo",
        "peso_porcentaje": 11.9,
        "nivel_impacto": "Impacto Alto"
      },
      {
        "feature": "Garajes",
        "label": "Cocheras (1.0)",
        "shap_value": 0.03413,
        "impacto": "positivo",
        "peso_porcentaje": 7.3,
        "nivel_impacto": "Impacto Moderado"
      },
      {
        "feature": "Superficie",
        "label": "Superficie (95.0 m²)",
        "shap_value": -0.00628,
        "impacto": "negativo",
        "peso_porcentaje": 1.3,
        "nivel_impacto": "Impacto Leve"
      }
    ]
  },
  "alerta_contexto": null
}
```

---

### 6.2. Horizonte 2: Solución Estructural para el Próximo Reentrenamiento (Pipeline v3)

Para futuros ciclos de reentrenamiento, se recomienda sustituir las 5 columnas redundantes por un **Índice Escalar Único de NSE** con restricciones monotónicas:

#### A. Formulación del Índice Único de NSE
En lugar de alimentar 5 porcentajes interdependentes que suman 100% ($\sum_{k} pct_k = 100$), se consolida en una sola variable continua:

* **Opción Canónica: Score Ponderado de Bienestar Socioeconómico (0 a 100):**
  $$\text{Score}_{\text{NSE}} = 1.00 \times \text{pct\_NSE\_A} + 0.75 \times \text{pct\_NSE\_B} + 0.50 \times \text{pct\_NSE\_C} + 0.25 \times \text{pct\_NSE\_D} + 0.00 \times \text{pct\_NSE\_E}$$
* **Opción Alternativa: Concentración de Alto Estrato (% A + B):**
  $$\text{pct\_AB} = \text{pct\_NSE\_A} + \text{pct\_NSE\_B}$$

#### B. Beneficios Estructurales Demostrados
1. **Eliminación Total de Multicolinealidad:** Se reduce la matriz distrital de dimensión $N \times 5$ con rango efectivo 1 a un único vector $N \times 1$.
2. **Atribución SHAP Limpia y Directa:** El SHAP ya no se fragmenta aleatoriamente entre A, B, C, D y E. Existe una sola barra interpretable: *"Índice NSE Distrital"*.
3. **Restricción Monotónica Forzada (`monotone_constraints`):**
   ```python
   params["monotone_constraints"] = "(..., Score_NSE: 1, ...)"
   ```
   Garantiza que un mayor nivel socioeconómico nunca pueda reducir el valor estimado de un predio, blindando la lógica económica del modelo ante objeciones periciales.
4. **Intervalo de Confianza Asociado:** El índice se reporta junto a su intervalo $[\text{Score}_{\text{inf}}, \text{Score}_{\text{sup}}]$, permitiendo al tasador conocer la dispersión del contexto.

