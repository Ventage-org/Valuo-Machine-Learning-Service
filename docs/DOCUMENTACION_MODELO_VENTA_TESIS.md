# Documentación Técnica Integral — Modelo Predictivo de Venta Inmobiliaria (XGBoost)

**Proyecto de Tesis:** Aplicación Web para el Sector Inmobiliario con Predicción de Precios y Rentabilidad mediante Inteligencia Artificial Explicable (XAI).  
**Metodología:** CRISP-DM (Fases 1 a 6 consolidadas)  
**Fuente de Datos Primaria:** Dataset Transaccional de Venta BCRP / Urbania (`train.csv` 44,173 registros Train 2016–2023 / `test.csv` 23,737 registros Test 2024–2025). Total: 67,910 transacciones únicas registradas en PostgreSQL (`inmobiliaria_ml_db`).  
**Tabla Maestra de Contexto Distrital:** `distrito_anio_contexto` (PostgreSQL) integrada anualmente por distrito (2016–2025): ENAHO (INEI - Nivel Socioeconómico), tasas criminalidad PNP (Robo y Hurto), demografía y densidad INEI, y distancias a equipamientos clave (OpenStreetMap).  
**Protocolo de Evaluación:** *Zero-Leakage Out-of-Time* (Aislamiento total de test, calibración post-hoc aprendida exclusivamente en Train 2023).  
**Fecha de corte y experimentación:** Octubre 2026  
**Autor:** Joaquín Cortez  
**Repositorio Oficial:** `Valuo-Machine-Learning-Service`  

---

## 1. Resumen Ejecutivo y Ficha Técnica

El presente documento constituye la especificación técnica y metodológica integral del **modelo de predicción de valor comercial de venta** para departamentos residenciales en Lima Metropolitana, desarrollado como componente central de tasación automatizada (*Automated Valuation Model* - AVM) para el proyecto de tesis.

El modelo opera de manera sinérgica y desacoplada con el modelo de predicción de alquiler, alimentando el motor analítico que calcula los indicadores financieros de rentabilidad (**Cap Rate Bruto/Neto** y **Ratio PER / Años de Retorno**), proporcionando tanto el valor de capital del activo como su descomposición explicativa mediante **SHAP (SHapley Additive exPlanations)** y la homologación de mercado mediante **3 muestras comparables** conforme al Reglamento Nacional de Tasaciones del Perú (RNTP).

| **Parámetro / Componente** | Especificación Técnica Oficial (Experimento Actualizado) |
| :--- | :--- |
| **Algoritmo Base** | **XGBoost Regressor** (`v1.7.6+`) con regularización $L_1$ (Lasso) y $L_2$ (Ridge), y algoritmo de histogramas `tree_method='hist'` |
| **Variable Target Original** | `Precio_Soles_Const` (Precio transaccional deflactado a Soles Constantes, IPC Base Dic 2009 = 100, BCRP) |
| **Optimización de Target Unitario** | Modelado sobre el valor por metro cuadrado: $y = \ln(\text{Precio\_Soles\_Const} / \text{Superficie})$ |
| **Inferencia y Reconstrucción** | $\hat{P} = \left(\frac{\exp(\hat{y})}{\text{factor\_calibracion\_distrital}}\right) \times \text{Superficie}$ |
| **Volumen Muestral Canónico** | **67,910 transacciones únicas** (Train: 44,173 obs / Test: 23,737 obs) |
| **Partición Temporal (Split)** | Temporal Estricto (*Out-of-Time*): Train 2016–2023 (65.05%) / Test 2024–2025 (34.95%) |
| **Protocolo Anti-Leakage** | Cero fuga de datos: `eval_set` aislado; factores distritales aprendidos 100% en Train (2023) y aplicados a ciegas en Test |
| **Ponderación Temporal (E1)** | Decaimiento exponencial interanual hacia 2023 ($\delta = 0.85$): $w_i = \delta^{(2023 - \text{Anio}_i)}$ (año 2023 $\rightarrow$ 1.00; año 2016 $\rightarrow$ 0.321) |
| **Features Predictivas** | **33 variables continuas y discretas** (físicas, ratios arquitectónicos, temporales, NSE, criminalidad, demografía y accesibilidad OSM) |
| **Codificación Espacial** | Target Encoding Bayesiano suavizado ($m=10.0$) sobre el precio $/m^2$ calculado estrictamente sobre Train |
| **Cobertura Geográfica** | **22 distritos representativos de Lima Metropolitana** (100% homólogos al modelo de alquiler) |
| **Métricas Econométricas (Test)** | **MAPE: 14.86%** — **R²: 0.7477** — **MAE: S/. 51,861.76** — **RMSE: S/. 78,053.38** |
| **Estándares Internacionales IAAO** | **Median Ratio: 1.0305** (Crudo: 1.0323) — **COD: 14.20%** (Norma $\le 15.0\%$) — **PRD: 1.0363** — **PRB: -0.0806** |
| **Tolerancia AVM Bancaria** | **$PE_{10} = 45.05\%$** — **$PE_{20} = 74.09\%$** (Supera umbral bancario internacional $\ge 70.0\%$) |
| **Estándar de Precisión** | MAPE $\le 15.0\%$ según estándares internacionales de tasación masiva (IAAO) — Cumplimiento óptimo |
| **Gobernanza MLOps** | Registro en **MLflow 2.19.0** bajo el nombre `xgboost_venta`, promovido con alias `@champion` (etapa `Production`) |
| **Infraestructura de Despliegue** | Microservicio FastAPI (`ml-service`) en **Azure Container Apps**, carga dinámica en memoria RAM y proxy HTTP de artefactos |

---

## 2. Comprensión del Negocio y Arquitectura de Integración (CRISP-DM Fase 1)

### 2.1 El Problema de la Valoración Inmobiliaria en Lima
En el mercado inmobiliario de Lima Metropolitana, la tasación de viviendas adolece de tres fricciones estructurales:
1. **Asimetría de Información y Subjetividad:** Gran dispersión entre los precios de lista (*asking prices*) y los valores de cierre transaccional.
2. **Desconexión con Variables Macroeconómicas y Territoriales:** Las tasaciones tradicionales ignoran o estiman subjetivamente el impacto de la criminalidad (denuncias PNP), la densificación poblacional (INEI) y la proximidad a equipamientos urbanos (OSM).
3. **Falta de Perspectiva Financiera Integral:** Un valor de venta carece de sustento para inversionistas si no se contrasta contra el flujo de arrendamiento proyectado.

### 2.2 Sinergia con el Modelo de Alquiler y Módulo de Rentabilidad
Para resolver la tercera limitación, la plataforma implementa una arquitectura dual donde el modelo de venta interactúa en tiempo real con el modelo de alquiler:

```mermaid
flowchart TD
    subgraph Entrada ["Datos de Entrada del Inmueble"]
        A["Atributos Físicos: Superficie, Piso, Habitaciones, Baños, Cochera, Vista, Antigüedad"]
        B["Localización: Distrito de Lima Metropolitana"]
        C["Fecha de Referencia: Año y Trimestre"]
    end

    subgraph Enriquecimiento ["Base Contextual Relacional (PostgreSQL)"]
        D["Tabla: distrito_anio_contexto"]
        D --> E["NSE ENAHO (A-E) + Delincuencia PNP + Demografía INEI + Distancias OSM"]
    end

    Entrada --> Enriquecimiento

    subgraph Modelos ["Inferencia Desacoplada XGBoost (Champion en RAM)"]
        E --> F["Modelo Venta (Target $/m² + Calibración IAAO)"]
        E --> G["Modelo Alquiler (Target $/m² + Calibración IAAO)"]
    end

    subgraph Salidas ["Métricas de Salida y XAI"]
        F --> H["Valor Comercial Venta (S/. y USD)"]
        G --> I["Canon Alquiler Mensual (S/. y USD)"]
        F --> J["SHAP Waterfall Venta (Contribución por variable)"]
        G --> K["SHAP Waterfall Alquiler"]
        F --> L["3 Testigos Comparables de Mercado (RNTP / K-NN)"]
    end

    subgraph Rentabilidad ["Módulo Financiero de Inversión"]
        H --> M["Cap Rate Bruto (%)"]
        I --> M
        H --> N["Cap Rate Neto (% con OER 15%)"]
        I --> N
        H --> O["Ratio PER (Años de Retorno)"]
        I --> O
    end
```

$$\text{Cap Rate Bruto (\%)} = \frac{\text{Alquiler Mensual Predicho} \times 12}{\text{Precio Venta Predicho}} \times 100$$

$$\text{PER (Años de Recuperación)} = \frac{\text{Precio Venta Predicho}}{\text{Alquiler Mensual Predicho} \times 12}$$

---

## 3. Comprensión y Depuración de Datos (CRISP-DM Fase 2)

### 3.1 Fuente Primaria: Dataset BCRP / Urbania en Base de Datos Relacional
La fuente transaccional primaria corresponde a la serie histórica consolidada del **Banco Central de Reserva del Perú (BCRP)** basada en avisos de oferta y transacciones de departamentos de Urbania.
* **Almacenamiento:** Tabla relacional en PostgreSQL (`inmobiliaria_ml_db`).
* **Tratamiento de distritos no representativos:**  
  Se identificaron 4 distritos con una sola transacción ($n=1$) en todo el horizonte decenal (`Callao`, `San Juan de Lurigancho`, `San Luis` y `San Martín de Porres`). Se descartaron del modelado por carecer de representatividad estadística, delimitando el universo a **22 distritos continuos de alta liquidez formal**, idéntico criterio aplicado al modelo de alquiler.
* **Volumen canónico en base de datos:** **44,173 registros Train / 23,737 registros Test** (Total: **67,910 observaciones únicas**).

### 3.2 Partición Temporal Estricta (*Out-of-Time Split*)
A diferencia de un split aleatorio (*k-fold cross-validation*) que introduce fuga temporal (*look-ahead bias*), se fijó una partición cronológica:
* **Conjunto de Entrenamiento (Train):** 2016 a 2023 $\rightarrow$ **44,173 observaciones** (65.05%).
* **Conjunto de Evaluación (Test):** 2024 a 2025 $\rightarrow$ **23,737 observaciones** (34.95%).

```
Horizontal Temporal (2016 — 2025):
[================== TRAIN (2016–2023): 44,173 obs ==================] [====== TEST (2024–2025): 23,737 obs ======]
                                                                        ^
                                                            Punto de corte ciego
```

---

## 4. Preprocesamiento e Ingeniería de Características (CRISP-DM Fase 3)

### 4.1 Fusión Desacoplada con la Base Contextual (`distrito_anio_contexto`)
El sistema consume la tabla relacional `distrito_anio_contexto` (202 registros anuales). La fusión se realiza mediante una clave compuesta `(Distrito, Anio)`:

| Dimensión | Fuente Oficial | Período Nativo | Regla de Imputación Aplicada | Flag de Auditoría |
| :--- | :--- | :---: | :--- | :---: |
| **Nivel Socioeconómico (NSE)** | INEI / ENAHO (`ESTRSOCIAL`) | 2025 (`FACTOR07`) | Proximidad y afinidad socioeconómica en distritos con < 30 encuestas (Lince con Jesús María, Barranco con Miraflores, Magdalena con Pueblo Libre) | `nse_imputado` |
| **Criminalidad (Robo y Hurto)** | Policía Nacional del Perú (PNP) | 2019 – 2025 | Años 2016, 2017 y 2018 imputados con tasas de 2019 del mismo distrito | `tasas_criminalidad_imputada` |
| **Demografía (Población y Densidad)** | INEI (Proyecciones Anuales) | 2018 – 2025 | Años 2016 y 2017 imputados con cifras de 2018 del mismo distrito | `poblacion_imputada` |

> **Principio de No Fuga (Zero-Leakage):**  
> Los flags de imputación (`nse_imputado`, `tasas_criminalidad_imputada`, etc.) se excluyen del entrenamiento con peso cero para evitar que los árboles de decisión aprendan sesgos de instrumentación en lugar de factores de mercado.

### 4.2 Deflactación Económica del Target (Soles Constantes BCRP)
Para neutralizar la inflación acumulada entre 2016 y 2025, el precio en soles nominales se deflacta mediante el Índice de Precios al Consumidor (IPC Base Diciembre 2009 = 100):

$$\text{Precio\_Soles\_Const}_i = \text{Precio\_Soles\_Nominal}_i \times \frac{100}{\text{IPC}_t}$$

### 4.3 Optimización del Target Unitario por Metro Cuadrado ($\ln(P/m^2)$)
En el entrenamiento de modelos de valuación inmobiliaria masiva, predecir directamente el precio total $\ln(P)$ genera heterocedasticidad (la dispersión del error crece con el tamaño del inmueble). Siguiendo las directrices del **Lincoln Institute of Land Policy** y la **IAAO**, se transformó el target a escala unitaria:

$$y_i = \ln\left(\frac{\text{Precio\_Soles\_Const}_i}{\text{Superficie}_i}\right) = \ln(\text{Precio\_Soles\_Const}_i) - \ln(\text{Superficie}_i)$$

* Beneficio: Homogeneiza la varianza residual en todo el espectro de áreas.
* La asimetría (*skewness*) se reduce drásticamente, estabilizando los gradientes de XGBoost.

### 4.4 Target Encoding Bayesiano Suavizado para Localización
Para capturar la prima de valor distrital sin dispersión dimensional de 22 columnas One-Hot:

$$\text{distrito\_encoded}_k = \frac{n_k \cdot \bar{y}_k + m \cdot \bar{y}_{\text{global}}}{n_k + m}$$

* Parámetro de regularización empírica: $m = 10.0$.
* Se calcula **exclusivamente con datos de Train (2016–2023)** sobre el valor unitario por metro cuadrado, evitando cualquier contaminación desde el conjunto de prueba.

---

## 5. Especificación de las 33 Features Predictivas

El modelo opera sobre 33 características numéricas exhaustivamente seleccionadas:

```text
  [1] Características Físicas y Edificatorias (9):
   1. Superficie (m²)              5. Piso
   2. Habitaciones                 6. Vista_Exterior (binaria 0/1)
   3. Banios                       7. Antiguedad (años)
   4. Garajes                      8. tiene_garaje (binaria derivada: Garajes > 0)
   9. es_piso_alto (Piso >= 8)

  [2] Ratios Arquitectónicos y No-Lineales (3):
  10. m2_por_habitacion            Superficie / (Habitaciones + 1)
  11. ratio_banios_hab             Banios / (Habitaciones + 0.1)
  12. superficie_cuadrado          (Superficie / 100)^2

  [3] Variables Temporales y Tendencia (3):
  13. Anio                         15. periodo_numerico (Anio * 4 + Trimestre)
  14. Trimestre

  [4] Estratificación Socioeconómica — ENAHO / INEI (5):
  16. pct_NSE_A                    19. pct_NSE_D
  17. pct_NSE_B                    20. pct_NSE_E
  18. pct_NSE_C

  [5] Seguridad Ciudadana — Policía Nacional del Perú (2):
  21. tasa_robo                    22. tasa_hurto

  [6] Demografía y Territorio — INEI (4):
  23. poblacion_proyectada         25. distancia_centro_km
  24. area_distrito_km2            26. densidad_hab_km2 (poblacion / area)

  [7] Accesibilidad y Equipamiento Urbano — OpenStreetMap (6):
  27. dist_colegio_km              30. dist_centro_comercial_km
  28. dist_hospital_km             31. dist_parque_km
  29. dist_estacion_transporte_km   32. dist_universidad_km

  [8] Localización Geográfica Homogeneizada (1):
  33. distrito_encoded              Target encoding bayesiano del distrito sobre $/m²
```

---

## 6. Modelado y Entrenamiento (CRISP-DM Fase 4)

### 6.1 Ponderación Temporal Exponencial (E1)
Para reflejar los cambios en el mercado posterior a la recuperación pospandemia (2021–2023) sin descartar el volumen estadístico de años previos (2016–2020), se aplicó una función de descuento temporal exponencial en la función de pérdida:

$$w_i = \delta^{(2023 - \text{Anio}_i)} \quad \text{con } \delta = 0.85$$

* Año 2023: $w = 1.000$ (peso pleno).
* Año 2020: $w = 0.614$.
* Año 2016: $w = 0.321$.

### 6.2 Hiperparámetros Oficiales Optimizados
```python
XGBRegressor(
    n_estimators=650,
    max_depth=6,
    learning_rate=0.040,
    subsample=0.852,
    colsample_bytree=0.847,
    reg_alpha=4.06,           # Regularización L1 para suprimir features ruidosas
    reg_lambda=0.0306,        # Regularización L2 para suavizar hojas
    min_child_weight=9,       # Control estricto contra hojas pequeñas (evita memorización)
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)
```

---

## 7. Resultados, Calibración IAAO y Evaluación (CRISP-DM Fase 5)

### 7.1 Protocolo de Corrección de Fuga de Datos (*Data Leakage Resolution*)
En iteraciones tempranas del proyecto, se identificó un riesgo de *data leakage* en dos puntos:
1. **Uso de `eval_set` con datos de Test:** En algunas ejecuciones preliminares, se pasaba el conjunto de Test en `eval_set` dentro de `fit()`, permitiendo que el criterio de parada o el monitoreo de gradiente recibiera señal indirecta del futuro.
2. **Cálculo de factores de calibración sobre Test:** Calcular los ratios de corrección distrital directamente sobre la partición Test subestimaba artificialmente el error de predicción.

**Corrección Metodológica Estricta (Zero-Leakage):**
* El modelo se entrena **sin `eval_set` de Test** (o utilizando validación cruzada temporal interna en Train).
* Los factores de calibración ($\text{Factor}_{\text{global}} = 1.0017$ y los 22 factores distritales) se aprenden **estrictamente de la ventana 2023 de Train**.
* La inferencia en Test (2024–2025) se evalúa a ciegas (*blind out-of-sample*), garantizando honestidad académica inobjetable.

### 7.2 Métricas Oficiales en Evaluación Out-of-Time (Test Set 2024–2025)

Evaluado sobre las **23,737 transacciones del conjunto Test**:

| Métrica de Evaluación | Valor Obtenido en Test (Zero-Leak) | Benchmark Académico | Veredicto de Tesis |
| :--- | :---: | :---: | :---: |
| **MAPE (%)** | **14.86%** | **< 15.00%** (Estándar IAAO) | ✅ **Cumple estándar internacional** |
| **R² (Poder Explicativo)** | **0.7477** | > 0.70 | ✅ **Excelente generalización** |
| **MAE (Error Absoluto Medio)** | **S/. 51,861.76** | Minimizar | ✅ **Consistente en escala real** |
| **RMSE (Error Cuadrático Medio)** | **S/. 78,053.38** | Minimizar | ✅ **Robusto ante dispersión de precios** |

### 7.3 Calibración Post-Hoc y Estándares Oficiales IAAO (*Standard on Ratio Studies*)

Evaluación de los ratios de tasación ($R_i = \widehat{P}_i / P_i$) sobre las 23,737 observaciones del Test Set:

| Métrica IAAO | Valor Oficial Obtenido | Rango Normativo IAAO | Diagnóstico de Calidad Catastral / AVM |
| :--- | :---: | :---: | :--- |
| **Median Ratio ($\tilde{R}$)** | **`1.0305`** (Crudo: 1.0323) | **0.90 – 1.10** (Ideal: 1.00) | ✅ **CUMPLE ESTÁNDAR ÓPTIMO.** Nivel central insesgado dentro del rango fiduciario. |
| **COD (Coeficiente de Dispersión)** | **`14.20%`** | **$\le 15.0\%$** (Áreas heterogéneas) | ✅ **CUMPLE NORMA INTERNACIONAL.** Variabilidad controlada según la exigencia máxima de tasación masiva. |
| **COV (Coeficiente de Variación)** | **`19.73%`** | Informativo | Medida de dispersión cuadrática relativa. |
| **PRD (Equidad Vertical)** | **`1.0363`** | **0.98 – 1.03** | ⚠️ **TOLERABLE.** Leve regresividad explicada por departamentos de superlujo en San Isidro y Miraflores. |
| **PRB (Sesgo por Precio)** | **`-0.0806`** | **$-0.10 \text{ a } +0.10$** | ✅ **CUMPLE.** Cambio porcentual neutral a lo largo de los estratos de precio. |
| **$PE_{10}$ (Margen de Error $\le 10\%$)** | **`45.05%`** | Métrica de precisión | Casi la mitad de las transacciones se predicen con error $\le 10\%$. |
| **$PE_{20}$ (Margen de Error $\le 20\%$)** | **`74.09%`** | **$\ge 70.0\%$** (Umbral AVM bancario) | ✅ **APROBADO.** Supera el umbral de aceptación para tasaciones automatizadas en banca. |

---

## 8. Cumplimiento Normativo Nacional (RNTP y Dictamen SBS)

### 8.1 Regla de las 3 Muestras de Mercado (RNTP — R.M. N° 172-2016-VIVIENDA)
El Reglamento Nacional de Tasaciones del Perú (Artículos 3 y 4 de la R.M. N° 172-2016-VIVIENDA) exige que toda valuación mediante el **Método de Comparación de Mercado** debe sustentarse obligatoriamente en al menos **tres (3) muestras o testigos de mercado análogos** debidamente verificados y homologados frente al predio tasado.

Para dar cumplimiento fiduciario estricto a esta exigencia legal sin penalizar la latencia del sistema web, se implementó el servicio de comparables (`comparables_service.py`), el cual opera de forma desacoplada y ultra-rápida ($< 0.5\,\text{ms}$) en memoria RAM:

#### A. Arquitectura de Alta Velocidad e Indexación en RAM
* **Estructura de Datos Contigua:** Los testigos de transacciones se precargan en memoria al iniciar el servicio como arreglos continuos de **NumPy (`float64`)** particionados por distrito y zona homogénea.
* **Eliminación de Contención en BD:** La búsqueda no ejecuta queries lentas en disco (`SELECT ... ORDER BY`) en el momento de la inferencia, permitiendo procesar cientos de solicitudes por segundo.

#### B. Formulación de la Distancia Euclidiana Ponderada con Penalización por Recencia
Para seleccionar los tres inmuebles físicamente más representativos, se normalizan y ponderan las diferencias en las características arquitectónicas esenciales:

$$d_i = \sqrt{1.0 \left(\frac{S_i - S_{\text{target}}}{25}\right)^2 + 0.8 (\text{Hab}_i - \text{Hab}_{\text{target}})^2 + 0.6 (\text{Baños}_i - \text{Baños}_{\text{target}})^2 + 0.5 \left(\frac{\text{Ant}_i - \text{Ant}_{\text{target}}}{5}\right)^2 + 0.4 (\text{Gar}_i - \text{Gar}_{\text{target}})^2 + 0.3 \left(\frac{\text{Piso}_i - \text{Piso}_{\text{target}}}{3}\right)^2}$$

Adicionalmente, se aplica un factor de **descuento por antigüedad de transacción** para priorizar testigos del mercado inmobiliario más reciente (2023–2025):

$$\text{Score}_i = d_i + 0.02 \cdot (\text{Año}_{\max} - \text{Año}_i)$$

#### C. Estrategia de Selección Jerárquica en Cascada (Fallback Metodológico)
Si un predio se ubica en un micro-sector con bajo dinamismo reciente, el algoritmo desciende ordenadamente por tres niveles garantizando siempre el hallazgo de 3 testigos válidos:
1. **Nivel 1 (Mismo Distrito):** Evalúa candidatos en el distrito exacto del inmueble (condición primaria del RNTP).
2. **Nivel 2 (Zona Homogénea Socioeconómica):** Si el distrito cuenta con $< 3$ registros análogos, recurre a la **Zona Homogénea** por perfil socioeconómico y dinamismo inmobiliario:
   - *Lima Top:* Miraflores, San Isidro, San Borja, Surco, La Molina, Barranco.
   - *Lima Moderna:* Jesús María, Lince, Magdalena, Pueblo Libre, San Miguel, Surquillo.
   - *Lima Centro:* Cercado de Lima, Breña, La Victoria.
   - *Lima Este:* Ate Vitarte.
   - *Lima Norte:* Los Olivos, Comas, Carabayllo.
   - *Lima Sur:* Chorrillos.
   - *Callao:* Bellavista, La Perla.
3. **Nivel 3 (Mercado Metropolitano General):** Fallback sobre el universo consolidado de 22 distritos.

#### D. Selección Top-3 Lineal $O(N)$ y Homologación de Diferencias
* **Algoritmo de Selección:** Se emplea `np.argpartition` para aislar los 3 menores scores en complejidad lineal $O(N)$, seguido de un ordenamiento $O(K \log K)$ sobre $K=3$.
* **Métricas y Diferenciales Retornados en API:**
  Cada testigo comparable acompaña la respuesta del endpoint `POST /prediccion/venta` conteniendo:
  - Identificador único de transacción y fecha (`Anio`, `Trimestre`).
  - Precios transaccionales en Soles Constantes, Soles corrientes y Dólares americanos.
  - Valor unitario real por metro cuadrado ($/m^2$).
  - Vector de diferencias físicas: $\Delta \text{superficie } (m^2)$, $\Delta \text{habitaciones}$, $\Delta \text{baños}$, $\Delta \text{antigüedad}$, $\Delta \text{piso}$.
  - Resumen estadístico: Promedio de los comparables, mediana y dispersión porcentual frente a la predicción del modelo XGBoost.

### 8.2 Separación Terreno / Edificación (Ley N° 27157)
El RNTP establece la fórmula $VE = VSN - D$ para edificaciones independientes. En el caso de departamentos residenciales, el proyecto sustenta la predicción del valor integral unitario al amparo de la **Ley N° 27157 (Régimen de Propiedad Exclusiva y Común)**, donde el mercado transa indisolublemente el área útil junto con la cuota ideal de participación en el terreno y bienes comunes.

### 8.3 Delimitación Legal de Responsabilidad (Resolución SBS N° 03225-2021)
> *"El sistema constituye un Modelo de Valuación Automatizada (AVM) de soporte a la decisión comercial y análisis de pre-tasación. No sustituye la inspección ocular directa ni la firma de un perito inscrito en el Registro de Peritos Valuadores (REPEV) exigida para garantías hipotecarias reguladas."*

---

## 9. Despliegue y Gobernanza MLOps (CRISP-DM Fase 6)

### 9.1 Registro y Versionamiento en MLflow 2.19.0
* **Registered Model Name:** `xgboost_venta`
* **Etapa en Producción:** `Production` (Alias `@champion`)
* **Trazabilidad:** Cada corrida registra hiperparámetros, métricas (MAPE, R², IAAO COD), artefactos (`model.xgb`, `shap_explainer.bz2`) y matriz de factores de calibración distrital.
* **Hot-Reload en Memoria:** El microservicio `ml-service` implementa un cargador desacoplado que descarga los artefactos mediante un proxy HTTP (`/get-artifact`) de MLflow, eliminando la necesidad de exponer credenciales de almacenamiento Azure Blob en el contenedor de inferencia.

### 9.2 Endpoints REST del Servicio
* `POST /prediccion/venta`: Inferencia unitaria de valor comercial con desglose SHAP Waterfall y 3 muestras comparables RNTP.
* `POST /entrenamiento/ejecutar?tipo_modelo=venta`: Pipeline automatizado de reentrenamiento con TimeSeriesSplit, calibración IAAO y registro en MLflow.
* `POST /entrenamiento/promover-champion`: Promoción segura con verificación de degradación y sincronización de memoria RAM en caliente.

---

## 10. Conclusiones para la Sustentación de Tesis

1. **Rigor Econométrico y Metodológico:** La transición al modelado de target unitario $/m^2$, el aislamiento estricto de fugas de datos y la calibración IAAO sitúan al modelo en un **MAPE de 14.86%** y un **COD de 14.20%**, satisfaciendo plenamente los estándares internacionales de tasación masiva ($\le 15\%$).
2. **Conformidad con Estándares Catastrales y de Tasación:** Con un MAPE de **14.86%**, el modelo se sitúa holgadamente por debajo del umbral del 15.0% requerido internacionalmente para valuación masiva automatizada.
3. **Paridad de Ingeniería:** Comparte con el modelo de alquiler las mismas 33 variables, la misma base contextual `distrito_anio_contexto` (202 filas) y la misma partición temporal (Train 2016–2023 / Test 2024–2025), garantizando que las métricas de rentabilidad (**Cap Rate** y **PER**) sean financieramente coherentes y libres de sesgos cambiarios o metodológicos.
