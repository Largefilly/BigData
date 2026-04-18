# Universidad Peruana de Ciencias Aplicadas (UPC)
## Curso: Big Data
## Proyecto: Educational Philanthropy & Donor Matching Engine
### Entregable Semana 3: Propuesta de Dataset y Dataset Procesado V1

**Integrantes del Equipo:**
1. ✅ `[Tu Nombre Completo Aquí]` - *Data Engineering Lead*
2. ✅ `[Nombre de Tu Hermanito Aquí]` - *Modeling and Evaluation Lead*
3. ✅ `[Nombre de un posible Tercero]` - *Reporting and Presentation Lead (Opcional)*

---

## 1. Propuesta de Proyecto

**Dominio**
Filantropía Educativa, Crowdfunding e Impacto Social.

**Planteamiento del Problema**
Muchos proyectos de aulas de escuelas públicas en zonas vulnerables no logran asegurar el financiamiento adecuado porque los donantes se enfrentan a una "sobrecarga de información" y tienen dificultades para encontrar causas que resuenen profundamente con sus valores específicos. Al mismo tiempo, los fondos no siempre se distribuyen equitativamente entre las escuelas con los mayores niveles de pobreza.

**Pregunta Esperada del Producto**
¿Cómo podemos agrupar las necesidades educativas ocultas basándonos en las solicitudes de proyectos escritas por los profesores (mediante procesamiento de lenguaje natural), y posteriormente construir un motor de recomendación que conecte a los donantes históricos con los proyectos de aula más urgentes para maximizar un financiamiento equitativo? Además, ¿cómo revela el grafo subyacente entre donantes y escuelas los "desiertos" de financiamiento?

**Por qué este dataset es adecuado para el curso**
El dataset cumple con todos los requisitos técnicos para la segunda mitad del curso:
1. **Capa de Catálogo (Catalog Layer):** Una vasta taxonomía de proyectos de aula, completa con materias categorizadas e índices de pobreza.
2. **Capa de Features (Feature Layer):** Texto no estructurado (ensayos de los profesores) perfecto para el Procesamiento de Lenguaje Natural (TF-IDF/Embeddings) y análisis de reducción de dimensionalidad (PCA/SVD).
3. **Capa de Interacción (Interaction Layer):** Registros explícitos de donaciones de donantes a proyectos, proporcionando la matriz vital para el Filtrado Colaborativo.
4. **Capa de Grafos (Graph Layer):** Un grafo bipartito que conecta donantes y escuelas, lo que permite un análisis de centralidad utilizando algoritmos de grafos para detectar estructuras de comunidades y nodos aislados.

## 2. Inventario de Fuentes

* **URL de Origen:** [DonorsChoose Dataset Archive (Kaggle)](https://www.kaggle.com/datasets/hanselhansel/donorschoose/data) 
* **Licencias o Condiciones de Acceso:** El dataset es abierto para uso educativo y de ciencia de datos bajo la estructura de licencias de datos abiertos de Kaggle, proporcionado originalmente por DonorsChoose.
* **Formatos de Archivo Crudo:** CSV (Valores Separados por Comas)
* **Tamaño Estimado:** Varios gigabytes sin comprimir (Solo los proyectos contienen más de 1 millón de filas; las donaciones superan los 4.6 millones de filas).

## 3. Borrador del Esquema (Schema Draft)

**Tablas de Entidades y Llaves:**
* `projects`: `Project_ID` (PK), `School_ID` (FK), `Teacher_ID` (FK), `Project_Subject_Category_Tree`, `Project_Title`, `Project_Essay`, `Project_Cost`
* `donations`: `Donation_ID` (PK), `Project_ID` (FK), `Donor_ID` (FK), `Donation_Amount`, `Donation_Included_Optional_Donation`
* `donors`: `Donor_ID` (PK), `Donor_City`, `Donor_State`, `Donor_Is_Teacher`
* `schools`: `School_ID` (PK), `School_Metro_Type`, `School_Percentage_Free_Lunch` (Proxy para nivel de pobreza)

**Uniones Esperadas (Joins):**
* `donations` INNER JOIN `projects` ON `donations.Project_ID = projects.Project_ID`
* `projects` INNER JOIN `schools` ON `projects.School_ID = schools.School_ID`

## 4. Dataset Procesado V1 (Script de Ingesta)

El siguiente script en Python constituye nuestra primera tubería reproducible de Ingeniería de Datos para construir el `Dataset V1`. Debido a las limitaciones computacionales, filtramos el dataset a un subconjunto cronológico reciente de proyectos con ensayos no nulos. *(Ver archivo `notebooks/ingesta.ipynb` para la ejecución completa).*

```python
import pandas as pd
import os

def build_dataset_v1(raw_data_path, output_path):
    """
    Ingiere los archivos crudos de DonorsChoose, limpia las capas de catálogo e interacción, 
    y exporta un dataset V1 manejable para el modelado posterior.
    """
    print("Leyendo archivos CSV crudos...")
    
    projects_df = pd.read_csv(os.path.join(raw_data_path, 'Projects.csv'), on_bad_lines='skip', low_memory=False)
    donations_df = pd.read_csv(os.path.join(raw_data_path, 'Donations.csv'), on_bad_lines='skip', low_memory=False)
    
    # 1. Limpieza y Filtrado del Catálogo (Proyectos)
    projects_clean = projects_df.dropna(subset=['Project Essay', 'Project Subject Category Tree', 'Project Cost']).copy()
    
    # Filtro cronológico: Solo proyectos a partir de 2017
    projects_clean['Project Posted Date'] = pd.to_datetime(projects_clean['Project Posted Date'], errors='coerce')
    projects_clean = projects_clean.dropna(subset=['Project Posted Date']).copy()
    projects_clean = projects_clean[projects_clean['Project Posted Date'].dt.year >= 2017].copy()
    
    # Crear feature de texto normalizado uniendo Título y Ensayo
    projects_clean['Text_Feature'] = projects_clean['Project Title'].astype(str) + " " + projects_clean['Project Essay'].astype(str)
    
    catalog_cols = ['Project ID', 'School ID', 'Teacher ID', 'Project Posted Date', 
                    'Project Subject Category Tree', 'Project Cost', 'Project Current Status', 'Text_Feature']
    catalog_v1 = projects_clean[catalog_cols]
    
    # 2. Limpieza de Capa de Interacciones (Donaciones)
    valid_project_ids = set(catalog_v1['Project ID'])
    donations_clean = donations_df[donations_df['Project ID'].isin(valid_project_ids)].copy()
    
    donations_clean['Donation Amount'] = donations_clean['Donation Amount'].fillna(donations_clean['Donation Amount'].median())
    donations_clean = donations_clean.dropna(subset=['Donor ID'])
    
    # Exportar a carpeta processed
    os.makedirs(output_path, exist_ok=True)
    catalog_v1.to_csv(os.path.join(output_path, 'catalog_projects_v1.csv'), index=False)
    donations_clean.to_csv(os.path.join(output_path, 'interactions_donations_v1.csv'), index=False)
    print("Dataset V1 construido y guardado en disco con éxito.")
```

## 5. Borrador del Diccionario de Datos

| Nombre de Feature | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `Project ID` | String | Identificador alfanumérico único para la solicitud del proyecto del aula. |
| `School ID` | String | Identificador alfanumérico único para la escuela pública. |
| `Project Subject Category Tree` | Categórico | La materia académica relevante para el proyecto (ej., 'Math & Science'). Se realizará "one-hot encoding" en la Semana 5. |
| `Text_Feature` | Texto | La cadena de texto concatenada del título del proyecto y el ensayo del maestro explicando la necesidad. |
| `Project Cost` | Numérico | El monto total de financiamiento objetivo solicitado (en USD). |
| `Donor ID` | String | Identificador alfanumérico único para el donante que realiza la contribución. |
| `Donation Amount` | Numérico | El monto monetario otorgado por el donante al proyecto específico (en USD). |

## 6. Análisis de Escala

* **Filas y Columnas:** El dataset crudo de `proyectos` contiene ~1.11 millones de filas y 18 columnas. El dataset crudo de `donaciones` contiene ~4.68 millones de filas y 7 columnas.
* **Valores Nulos (Missingness):** Los valores faltantes se encuentran predominantemente en campos de metadatos opcionales. Las llaves primarias (`Project ID`, `Donor ID`) tienen una tasa de valores nulos casi del 0%. Los campos de texto principal (`Project Essay`) tienen valores nulos extremadamente bajos.
* **Esparsidad (Sparsity):** La matriz de interacción Donante-Proyecto es altamente dispersa ("sparse"). Millones de donantes solo realizan 1 o 2 interacciones (Distribución de larga cola). 
* **Estrategia del Subconjunto de Trabajo:** Para gestionar las restricciones de memoria y la huella de esparsidad, el `Dataset V1` aplica un estricto corte temporal (2017 en adelante). Esto reduce exitosamente el dataset activo a **308,399 proyectos** y **1,202,425 donaciones (interacciones)**, que es una escala ideal para los próximos algoritmos. Filtraremos aún más a los donantes de "inicio frío" (aquellos con menos de 3 interacciones) durante la fase del Motor de Recomendación (Semana 10) para asegurar matrices densas en el Filtrado Colaborativo.

## 7. Nota de Ética y Acceso

* **De dónde provienen los datos:** Los datos son una exportación pública de código abierto de la organización DonorsChoose, alojada en la plataforma Kaggle.
* **Por qué el equipo tiene permitido usarlo:** El dataset se publica explícitamente para fomentar la investigación en ciencia de datos y el modelado analítico destinado a mejorar los resultados de la filantropía educativa. 
* **Qué riesgos de datos personales existen:** Aunque los datos son públicos, involucran actores humanos (profesores escribiendo ensayos, escuelas en ubicaciones geográficas específicas y donantes). Existe un riesgo teórico de elaboración de perfiles de comportamiento ("behavioral profiling").
* **Cómo se redujeron esos riesgos:** DonorsChoose y Kaggle han ocultado (mediante hashing) y anonimizado completamente todas las Identidades Únicas (`Teacher_ID`, `Donor_ID`). Los nombres reales exactos y la información de contacto han sido depurados de raíz. Nuestro equipo se compromete a usar estos datos estrictamente de forma agregada. No utilizaremos técnicas de referencia cruzada (por ejemplo, buscar fragmentos de ensayos en la web) para des-anonimizar a los profesores, escuelas o estudiantes. Los resultados finales enmascararán de forma segura los indicadores geográficos sensibles.
