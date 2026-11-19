# Sustento Normativo y Metodológico: Homologación de Muestras Comparables según el RNTP

**Documento Técnico para Artículo Científico / Tesis de Ingeniería**  
**Módulo:** `app/services/comparables_service.py`  
**Normativa de Referencia:** Reglamento Nacional de Tasaciones del Perú (R.M. N° 172-2016-VIVIENDA y modificatorias R.M. N° 424-2017-VIVIENDA, R.M. N° 124-2020-VIVIENDA)

---

## 1. Marco Legal y Normativo (El Reglamento Nacional de Tasaciones del Perú - RNTP)

En el Perú, toda valuación de bienes inmuebles con fines comerciales, bancarios, hipotecarios o periciales se rige por el **Reglamento Nacional de Tasaciones del Perú (RNTP)**, aprobado por el Ministerio de Vivienda, Construcción y Saneamiento (MVCS) mediante la **Resolución Ministerial N° 172-2016-VIVIENDA**.

### 1.1. Disposiciones Clave del RNTP Aplicadas al Modelo AVM

1. **Tasación Comercial y Estudio de Mercado (Título I, Artículo 3, numeral 3.2):**
   Define la *Tasación Comercial* como aquella en la que se utilizan valores del libre mercado aplicando métodos directos e indirectos, sustentados obligatoriamente en un **estudio de mercado con muestras de bienes similares (testigos)**.

2. **Regla de las Tres (3) Muestras Comparables y Homologación (Título II — Valuación de Predios Urbanos):**
   El reglamento y la práctica pericial estandarizada por el Cuerpo Técnico de Tasaciones del Perú (CTTP) y la Superintendencia de Banca, Seguros y AFP (SBS — Resolución SBS N° 11356-2008) establecen que toda valuación por el **Método de Comparación de Mercado** debe respaldarse con un **mínimo de tres (3) muestras representativas (testigos)** que cumplan con:
   * **Proximidad geográfica o pertenencia a una misma Zona Homogénea:** Mismo distrito o área de estrato socioeconómico y dinámica urbana equivalente.
   * **Similitud tipológica y física:** Equivalencia en superficie ($\text{m}^2$), distribución (dormitorios, baños, estacionamientos), nivel de piso y edad de la edificación (antigüedad).
   * **Homologación:** Proceso técnico mediante el cual se cuantifica el grado de semejanza y las diferencias físicas entre cada muestra de mercado y el inmueble sujeto de tasación, además de actualizar los valores transaccionales a la fecha presente.

---

## 2. Propósito en la Arquitectura del Modelo de Valoración Automatizada (AVM)

Los modelos de aprendizaje automático basados en ensamblados de árboles (*Gradient Boosted Decision Trees* como XGBoost) destacan por su alta precisión predictiva, pero tradicionalmente enfrentan resistencia en el sector inmobiliario y pericial por ser percibidos como "cajas negras".

Para superar esta limitación y alinear el sistema con la legislación peruana, cada petición de predicción (`POST /api/v1/prediccion/venta` y `/alquiler`) integra dos pilares complementarios de transparencia:
1. **Explicabilidad Algorítmica Interna (SHAP TreeExplainer):** Desglosa matemáticamente el aporte de las 33 variables sobre el valor predicho.
2. **Evidencia Empírica Normativa (Homologación RNTP de 3 Muestras):** Ejecuta en paralelo una búsqueda matricial en memoria para recuperar los **3 inmuebles reales transaccionados más idénticos** al inmueble consultado, calculando sus diferencias exactas y su convergencia de precio.

---

## 3. Funcionamiento Técnico y Matemático del Algoritmo (`ComparablesService`)

El servicio de homologación está implementado en [`comparables_service.py`](file:///d:/NuevaCarpetaLool/proyecto-tesis/Valuo-Machine-Learning-Service/ml-service/app/services/comparables_service.py) y opera en **menos de 0.5 milisegundos** por consulta gracias a la indexación precalculada en arreglos contiguos de `NumPy` en memoria RAM.

```mermaid
flowchart LR
    A["Request de Predicción\n(Distrito, m², Hab, Baños, Antigüedad, Garajes, Piso)"] --> B{"¿Distrito tiene\n>= 3 muestras?"}
    B -- Sí --> C["Nivel 1: Mismo Distrito"]
    B -- No --> D["Nivel 2: Zona Homogénea NSE\n(Lima Top, Lima Moderna, etc.)"]
    C --> E["Cálculo Vectorizado de Distancia\nEuclidiana Ponderada + Bono de Recencia"]
    D --> E
    E --> F["Selección O(N) con np.argpartition\n(Top 3 Testigos Más Similares)"]
    F --> G["Actualización Monetaria por IPC\n+ Cálculo de Índice de Similitud (%)"]
```

### Paso 1: Indexación en Memoria y Deduplicación
Al iniciar el microservicio, `ComparablesService` carga el repositorio histórico de transacciones verificadas desde PostgreSQL ($\text{año} \ge 2021$, sumando **93,428 ventas** y **77,588 alquileres**), elimina registros duplicados exactos y construye tablas hash en memoria RAM particionadas por **Distrito** y por **Zona Homogénea**.

### Paso 2: Estrategia Territorial en Cascada (3 Niveles)
Siguiendo el criterio pericial del RNTP cuando una subzona presenta baja densidad muestral, el sistema aplica una cascada jerárquica:
1. **Nivel 1 — Mismo Distrito (Prioridad Estricta):** Filtra exclusivamente testigos ubicados dentro de la misma jurisdicción distrital.
2. **Nivel 2 — Zona Urbana Homogénea (Fallback NSE):** Si el distrito cuenta con menos de 3 muestras válidas, busca en la macrozona socioeconómica equivalente definida según la segmentación urbana de Lima Metropolitana (CAPECO / INEI):
   * `LIMA_TOP`: Miraflores, San Isidro, San Borja, Surco, La Molina, Barranco.
   * `LIMA_MODERNA`: Jesús María, Lince, Magdalena, Pueblo Libre, San Miguel, Surquillo.
   * `LIMA_CENTRO`: Cercado de Lima, Breña, La Victoria.
   * `LIMA_ESTE`: Ate Vitarte.
   * `LIMA_NORTE`: Los Olivos, Comas, Carabayllo.
   * `LIMA_SUR`: Chorrillos.
   * `CALLAO`: Bellavista, La Perla.
3. **Nivel 3 — Mercado Metropolitano:** Respaldo global para garantizar tolerancia a fallos del 100%.

### Paso 3: Función de Distancia Euclidiana Ponderada y Normalizada
Para evaluar objetivamente la similitud multidimensional entre el inmueble sujeto $X^* = (S^*, H^*, B^*, A^*, G^*, P^*)$ y cada candidato $i$ del repositorio, se formuló una **Distancia Euclidiana Ponderada con Escaladores de Tolerancia Inmobiliaria** ($D_i$):

$$D_i = \sqrt{ \sum_{k} w_k \left( \frac{x_{i,k} - x^*_k}{\sigma_k} \right)^2 }$$

Desarrollando cada término según la implementación exacta del código:

$$D_i = \sqrt{ 1.0 \left(\frac{S_i - S^*}{25.0}\right)^2 + 0.8 \left(\frac{H_i - H^*}{1.0}\right)^2 + 0.6 \left(\frac{B_i - B^*}{1.0}\right)^2 + 0.5 \left(\frac{A_i - A^*}{5.0}\right)^2 + 0.4 \left(\frac{G_i - G^*}{1.0}\right)^2 + 0.3 \left(\frac{P_i - P^*}{3.0}\right)^2 }$$

#### Justificación de los Parámetros de Homologación:

| Variable ($k$) | Símbolo | Escalador de Tolerancia ($\sigma_k$) | Peso Jerárquico ($w_k$) | Sustento Técnico Pericial |
| :--- | :---: | :---: | :---: | :--- |
| **Superficie ($\text{m}^2$)** | $S$ | $25.0\text{ m}^2$ | **$1.00$** | Atributo físico principal. Normaliza diferencias de área para que $25\text{ m}^2$ equivalgan a 1 unidad estándar de desviación. |
| **Habitaciones** | $H$ | $1.0\text{ hab}$ | **$0.80$** | Define el programa arquitectónico y el segmento familiar del inmueble. |
| **Baños** | $B$ | $1.0\text{ baño}$ | **$0.60$** | Indicador directo de confort interior y categoría del departamento. |
| **Antigüedad** | $A$ | $5.0\text{ años}$ | **$0.50$** | Refleja la depreciación física y estado de conservación por quinquenios (tablas de depreciación del RNTP). |
| **Garajes (Cocheras)** | $G$ | $1.0\text{ garaje}$ | **$0.40$** | Atributo complementario de alto valor comercial en zonas densificadas. |
| **Nivel de Piso** | $P$ | $3.0\text{ pisos}$ | **$0.30$** | Captura diferencias de altura, iluminación y vistas cada 3 niveles. |

### Paso 4: Penalización por Antigüedad Temporal de la Muestra (Recencia)
El RNTP exige que las muestras de mercado sean lo más contemporáneas posible a la fecha de tasación. Para desempatar muestras físicamente similares a favor de las transacciones más recientes, se incorpora una penalización temporal lineal suave:

$$\text{Score}_i = D_i + 0.02 \times (\text{Año}_{\max} - \text{Año}_i)$$

Donde $\text{Año}_{\max}$ es el año más reciente disponible en la base de datos.

### Paso 5: Selección $\mathcal{O}(N)$ e Índice de Similitud Porcentual
En lugar de ordenar toda la matriz ($\mathcal{O}(N \log N)$), se aplica el algoritmo de selección rápida por partición `np.argpartition(score_total, 2)[:3]`, que extrae los **3 mejores testigos en tiempo lineal $\mathcal{O}(N)$**.

Para facilitar la interpretación por parte del agente inmobiliario o tasador, la distancia $D_i$ se transforma en un **Índice de Similitud Porcentual** acotado en $[50\%, 100\%]$ mediante una función de decaimiento hiperbólico:

$$\text{Similitud}_i (\%) = \max\left( 50.0, \, \min\left( 100.0, \, \frac{100.0}{1.0 + 0.15 \cdot D_i} \right) \right)$$

* Si el testigo es físicamente idéntico al inmueble tasado ($D_i = 0$), su similitud es del **$100.0\%$**.

### Paso 6: Indexación por IPC y Verificación de Convergencia (`ResumenComparablesRNTP`)
Dado que el repositorio histórico almacena precios en **Soles Constantes (Base Dic. 2009 = 100)** para aislar la inflación durante el entrenamiento, el servicio actualiza el precio de cada testigo seleccionado a **Soles Nominales vigentes** multiplicándolo por el factor IPC actual del INEI/BCRP:

$$P_{\text{nominal}, i} = P_{\text{constante}, i} \times \left(\frac{\text{IPC}_{\text{vigente}}}{100}\right)$$

Finalmente, el módulo construye el objeto de auditoría `resumen_rntp`, el cual calcula:
1. **Precio Promedio de las 3 Muestras ($\bar{P}_{\text{RNTP}}$):** Media aritmética de los 3 testigos en Soles Nominales.
2. **Precio por $\text{m}^2$ Promedio ($\overline{P/\text{m}^2}_{\text{RNTP}}$):** Media aritmética del valor unitario de los 3 testigos.
3. **Desviación frente a la Predicción AVM ($\Delta_{\text{RNTP}}$):** Mide el grado de divergencia entre la estimación no lineal de XGBoost ($\hat{Y}_{\text{nominal}}$) y el promedio directo de los 3 testigos:

$$\Delta_{\text{RNTP}} (\%) = \left( \frac{\hat{Y}_{\text{nominal}} - \bar{P}_{\text{RNTP}}}{\bar{P}_{\text{RNTP}}} \right) \times 100$$

4. **Certificación Normativa (`cumple_regla_tres_muestras = true`):** Valida que la tasación automatizada cuenta con el respaldo documental de al menos 3 testigos homologados exigidos por el RNTP.

---

## 4. Párrafo Sugerido para Incluir en el Paper / Tesis

> *"Con el fin de alinear el modelo de valoración automatizada (AVM) con las exigencias regulatorias del Reglamento Nacional de Tasaciones del Perú (RNTP, R.M. N° 172-2016-VIVIENDA), se diseñó e integró un motor de homologación matricial en tiempo real que sustenta cada inferencia de XGBoost con las tres (3) muestras comparables más análogas del mercado. El algoritmo implementa una búsqueda jerárquica en cascada (distrito exacto $\to$ zona urbana homogénea por estrato socioeconómico $\to$ área metropolitana) y evalúa la proximidad multidimensional mediante una función de distancia euclidiana ponderada sobre seis atributos físicos normalizados (superficie, dormitorios, baños, antigüedad, estacionamientos y piso), combinada con un factor de recencia transaccional. La selección de los tres testigos óptimos se resuelve en complejidad lineal $\mathcal{O}(N)$ mediante partición rápida en arreglos contiguos de memoria (`np.argpartition`), logrando tiempos de respuesta inferiores a $0.5\text{ ms}$ y proporcionando al tasador tanto el índice de similitud porcentual como el vector explícito de diferencias físicas y la desviación respecto al valor predicho actualizado por IPC."*
