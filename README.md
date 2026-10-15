# Diagnóstico de marca empleadora con reseñas de Glassdoor

Análisis de NLP sobre reseñas públicas de Glassdoor para saber qué valoran y qué critican los empleados de **Deutsche Bank** y cómo se compara con tres competidores: **Barclays, Citi y HSBC Holdings**.
El objetivo es ayudar a Recursos Humanos a identificar fortalezas para employer branding y áreas de mejora que dañan la reputación o provocan salidas, bajo una regla: **un problema solo es prioritario si estamos peor que el mercado**.

Trabajo del módulo de Deep Learning del Máster en Data Science & IA (Evolve Academy).

## Pregunta de negocio

1. ¿Qué puntos fuertes de Deutsche Bank se pueden explotar en selección y employer branding?
2. ¿Qué áreas de mejora le dejan peor que sus competidores?
3. ¿Se quejan de lo mismo los empleados actuales y los antiguos (posibles motivos de salida)?

## Datos

- **Fuente:** *Glassdoor Job Reviews 1 y 2* (David Gauthier, Kaggle), unidos y preparados por el profesor en un único Parquet.
- **Ventana:** julio 2018 – julio 2023. Solo empresas con al menos 300 reseñas.
- **Variables usadas:** `firm`, `date_review`, `current`, `overall_rating`, `recommend`, `pros`, `cons`.
- **Filtro:** se cargan solo las 4 empresas elegidas (36.350 reseñas: Citi 14.609, Barclays 9.863, HSBC 6.404 y Deutsche Bank 5.474).
- **Muestra para NLP:** 1.000 reseñas por banco (4.000 en total, semilla 42), para que ningún banco domine los modelos y los tiempos sean razonables.

> El fichero de datos **no se versiona** (`data/` está en `.gitignore`). Hay que colocarlo en `data/glassdoor_reviews_hr.parquet`.

## Estructura del repositorio

```
glassdoor-employer-branding/
├── notebooks/
│   └── 01_analisis_glassdoor.ipynb   # análisis completo y conclusiones
├── data/                              # Parquet de reseñas (no versionado)
├── outputs/                           # materiales generados (no versionado)
├── requirements.txt
└── README.md
```

## Qué hace el cuaderno

1. **Preparación y EDA.** Separa `current` en situación (actual / antiguo) y antigüedad; rating, % que recomienda y reparto actuales/antiguos por empresa; evolución del rating por año.
2. **Sentimiento (control de calidad).** DistilBERT sobre pros y cons por separado, con `truncation=True`. Se contrasta con SiEBERT y con lectura manual; se analizan los casos discordantes.
3. **Topic modeling con BERTopic.** Un único modelo para los cuatro bancos (uno para pros y otro para cons), para que los temas sean comparables.
   - Con los ajustes por defecto, el 93 % de los comentarios cae en un solo tema.
   - Con HDBSCAN afinado, el 40,6 % de las reseñas queda sin tema.
   - Se usa KMeans con 12 grupos para que todas las reseñas cuenten; se eliminan los nombres de los bancos de los textos y se etiquetan los grupos a mano leyendo ejemplos.
4. **Categorías de negocio.** Los 12 temas se agrupan en 6 categorías comunes a pros y cons: compensación, conciliación y horarios, dirección y gestión, carrera y aprendizaje, cultura y personas, y procesos y herramientas.
5. **Comparación con peers.** Balance neto por categoría (% en pros − % en cons) y gap frente a la media de los peers; matriz de priorización (peso en los cons frente a desventaja); cruce de cada categoría con `overall_rating` y `recommend`.
6. **Actuales frente a antiguos.** Temas de los cons de los antiguos como posibles motivos de salida, también frente a los antiguos de los peers.
7. **Recomendaciones, limitaciones, próximos pasos y conclusión.**

## Resultados principales

- Deutsche Bank tiene un rating medio de **3,73** y es el banco con **menos reseñas que recomiendan** la empresa (43,4 %, frente a un 45,7-48,0 % en los demás). La mejora de rating desde 2019 se da también en los peers.
- **Fortaleza:** «Cultura y personas» (36,0 % de sus pros, la más alta de los cuatro bancos; +7,7 puntos de balance neto sobre la media de los peers).
- **Única categoría prioritaria:** «Compensación» (17,1 % de sus cons y −9,2 puntos de balance frente a los peers). Aun así, las reseñas que se quejan de ella no puntúan peor (3,80 frente a 3,68 de media): es una desventaja frente al mercado más que la causa del descontento.
- **Dirección y gestión** es la queja más frecuente y la que se asocia a peor valoración (3,21), pero ocurre igual en los cuatro bancos, por lo que no es una desventaja propia de Deutsche Bank.
- Entre los **antiguos**, «Compensación» pesa más que entre los antiguos de los peers (13,3 % frente a 8,8 %).

Las conclusiones son descriptivas y se basan en reseñas voluntarias: ver el apartado de limitaciones del cuaderno.

## Cómo ejecutarlo

Requiere **Python 3.12 o superior** (el cuaderno usa f-strings con comillas anidadas).

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Después, coloca el Parquet en `data/` y abre `notebooks/01_analisis_glassdoor.ipynb` con Jupyter o VS Code. Ejecútalo de arriba abajo.

- La primera ejecución **descarga modelos de Hugging Face** (DistilBERT, SiEBERT y un modelo de embeddings), así que necesita conexión; SiEBERT es el más grande.
- No hace falta GPU. En CPU, el cuaderno completo tarda unos 10 minutos.
- Las semillas están fijadas (`random_state=42`), por lo que los resultados son reproducibles.

## Referencias

- Gauthier, D. *Glassdoor Job Reviews* y *Glassdoor Job Reviews 2* (Kaggle).
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794.
- Sanh, V., Debut, L., Chaumond, J. & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108.
- Modelo SiEBERT: `siebert/sentiment-roberta-large-english` (Hugging Face).
