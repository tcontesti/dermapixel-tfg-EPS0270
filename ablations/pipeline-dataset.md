# Anexo C — Pipeline de construcción y caracterización del dataset DermapixelAI 1.0

> Material complementario del TFG EPS0270. Corresponde al antiguo Anexo C de la memoria. Documenta el pipeline de construcción y la caracterización cuantitativa completa del dataset DermapixelAI 1.0, descrito de forma resumida en el capítulo de materiales y métodos. El dataset se publica como contribución independiente del trabajo.

## C.1 Origen y materia prima

El dataset DermapixelAI se construye a partir del archivo del blog Dermapixel. La materia prima es el conjunto de casos publicados durante más de quince años por la Dra. R. Taberner, cada uno acompañado de imágenes (fotografía clínica y, cuando procede, imagen dermatoscópica), historia clínica abreviada en castellano y diagnóstico final con razonamiento.

## C.2 Pipeline de construcción

Proceso iterativo de seis pasos:

**(i) Extracción.** Recogida de las publicaciones del blog y de los metadatos asociados por caso (título, fecha, texto narrativo, imágenes, diagnóstico final). Scripts deterministas a partir de la fuente HTML del blog.

**(ii) Deduplicación.** Cálculo de hash MD5 sobre cada imagen y eliminación de duplicados exactos. Identificación y exclusión documentada de imágenes asociadas a la pista de solución de los casos del blog (susceptibles de inducir *leakage*).

**(iii) Clasificación de modalidad.** Asignación a cada imagen de su modalidad (*clinical*, *dermoscopy*, *histology*, *ultrasound* o *wood_lamp*) mediante heurísticas sobre el nombre del fichero y revisión manual. Incluye auditoría sistemática y exclusión de imágenes no dermatológicas (a `images/_excluded/`, eliminándose del fichero maestro `dataset.csv` y manteniéndose en `cases.csv` con `num_images_in_dataset = 0`).

**(iv) Mapeo ontológico.** La etiqueta cruda de cada caso se mapea, mediante revisión experta de la Dra. Taberner sobre la totalidad del dataset, a una entrada L3 de la ontología jerárquica. Genera tres campos: `label_source`, `diagnosis_source` y `rosa_verified`.

**(v) Particionado.** Asignación a `train`, `val` o `test` bajo regla *case-aware*: todas las imágenes de un mismo caso clínico van al mismo split.

**(vi) Auditoría de integridad.** Verificación final de hashes MD5, ausencia de imágenes corruptas, consistencia carpetas/`dataset.csv`. Toda modificación se acompaña de *backup* y registro de auditoría (`audit_log.txt` y `excluded_images.csv`).

La versión 1.0 contiene 1 089 imágenes vinculadas a 669 casos clínicos con identificador único resoluble, sobre un total de 698 casos catalogados en el archivo original (672 disponen de imagen efectiva antes de la deduplicación MD5; los tres casos restantes se pierden por duplicado exacto de imagen entre entradas distintas del blog).

## C.3 Cardinalidades del dataset y splits experimentales

Coexisten tres particiones distintas, todas *case-aware* (ningún caso se reparte entre splits). El test del estudio principal sobre el dataset de producción es deliberadamente reducido para reservar masa de validación interna a la futura cohorte prospectiva del HUSLL.

| Configuración | Imágenes | Casos | Train | Val/Test | Uso |
|---|---:|---:|---:|---|---|
| Producción (`dataset.csv`) | 1 089 | 669 | 891 | 157 / 41 | Estudio principal |
| Filtrada (`dataset_filtered.csv`) | 1 062 | 653 | 874 | 152 / 36 | Restricción a *clinical+dermoscopy* con `label_source=ontology` |
| Dermapixel R0 contrastivo | 1 062 | 653 | 854 | 107 / 101 | Experimento dual-encoder con *manifest* SHA-256 propio |

La cifra base 1 089 es el resultado del pipeline completo; el subconjunto filtrado de 1 062 excluye 27 imágenes que pierden trazabilidad ontológica tras la auditoría de mapeo; los splits de la rama contrastiva (854/107/101) son reproducibles a partir del *manifest* SHA-256 pinned en cada checkpoint.

## C.4 Distribución por modalidad

| Modalidad | N |
|---|---:|
| clinical | 1 036 |
| dermoscopy | 49 |
| histology | 2 |
| ultrasound | 1 |
| wood_lamp | 1 |
| **Total en `dataset.csv`** | **1 089** |

## C.5 Distribución por categoría etiológica L1

La fila «(sin asignar)» recoge las 19 imágenes cuyo mapeo ontológico no se ha completado (todas corresponden a casos con `label_source = raw`).

| Categoría L1 | N | % |
|---|---:|---:|
| Patología inflamatoria | 544 | 49,9 |
| Patología tumoral | 276 | 25,3 |
| Patología infecciosa | 259 | 23,8 |
| Genodermatosis | 11 | 1,0 |
| (sin asignar) | 19 | — |
| **Total** | **1 089** | **100,0** |

## C.6 Distribución por subcategoría L2 y por diagnóstico L3

Sobre el subconjunto filtrado (1 062 imágenes), Patología inflamatoria concentra el 50,1 %; Genodermatosis es la cola corta con 8 imágenes (0,8 %), lo que explica su ausencia en el split de `test` (figura `fig_eda_l1.png`).

De las 43 entradas L2 del vocabulario, 38 aparecen representadas con al menos una imagen. La extracción inicial observa 39, pero dos corresponden a variantes ortográficas de la misma subcategoría (*Trastornos de la queratinización* y *Trastornos queratinización*); la normalización reduce la cifra efectiva a 38 (top-20 en figura `fig_eda_l2.png`).

A nivel L3, 250 entradas están representadas sobre las 367 declaradas (cobertura efectiva del 68,1 %). El L3 más frecuente es *Psoriasis en placas* (38 imágenes), seguido por *Tiña corporis* (32), *Liquen plano* (28) y *Nevo melanocítico adquirido* (25).

### Top-10 diagnósticos L3 más frecuentes

| Diagnóstico L3 | N |
|---|---:|
| Psoriasis en placas | 38 |
| Tiña corporis | 32 |
| Liquen plano | 28 |
| Nevo melanocítico adquirido | 25 |
| Carcinoma basocelular | 23 |
| Queratosis actínica | 21 |
| Dermatitis alérgica de contacto | 21 |
| Acné | 16 |
| Dermatofibroma | 16 |
| Melanoma | 14 |
| **Total top-10** | **234** |

Estos diez diagnósticos concentran 234 imágenes, equivalentes al 22,0 % del dataset filtrado.

## C.7 Cola larga diagnóstica

Reparto de las 250 clases L3 efectivas según número de imágenes por clase (figuras `fig_eda_l3_cola.png`, `fig_eda_l3_rank.png`).

| Rango imgs/clase | Nº clases L3 | % del total |
|---|---:|---:|
| 1 img | 64 | 25,6 |
| 2–5 imgs | 129 | 51,6 |
| 6–10 imgs | 39 | 15,6 |
| 11–20 imgs | 11 | 4,4 |
| 21+ imgs | 7 | 2,8 |
| **Total** | **250** | **100,0** |

El 77,2 % de las 250 clases L3 dispone de cinco imágenes o menos, y un cuarto (25,6 %) cuenta con una única imagen. Esto condiciona la viabilidad de un clasificador con cabeza directa a 250 clases L3 y justifica la estrategia de cabeza L2 con ranking L3 por prototipos.

## C.8 Particionado en train, validación y test

Cada caso clínico (identificado por `case_id`) está íntegro en un único split. Media 1,63 imágenes por caso, mediana 2, máximo 5 (figura `fig_eda_casos.png`).

| Split | N |
|---|---:|
| train | 891 |
| val | 157 |
| test | 41 |
| **Total** | **1 089** |

La regla *case-aware* se mantiene íntegra: ningún `case_id` aparece en más de un split. La composición L1 dentro de cada split no es perfectamente estratificada: Genodermatosis carece de representación en `test`, once entradas L2 carecen de imágenes en `val` y veintidós L2 carecen de imágenes en `test` (figura `fig_eda_splits.png`).

### Cobertura ontológica por split sobre el subconjunto filtrado (1 062 imágenes)

Nº de entradas únicas observadas en cada nivel.

| Split | Imágenes | Casos | L1 | L2 | L3 |
|---|---:|---:|---:|---:|---:|
| train | 874 | 527 | 4 | 38 | 224 |
| val | 152 | 98 | 3 | 28 | 72 |
| test | 36 | 28 | 3 | 17 | 27 |

El tamaño del split `test` (36 imágenes, 28 casos, 17 L2 únicas, 27 L3 únicas) constituye una limitación intrínseca para la evaluación a niveles ontológicos profundos.

## C.9 Texto narrativo asociado al caso

Cada caso lleva un campo `case_text` en castellano (motivo de consulta, antecedentes, evolución y, en ocasiones, razonamiento inicial). Longitud mediana 223 palabras por caso, con primer y tercer cuartil en 175 y 271 palabras.

Un subconjunto reducido contiene la palabra «diagnóstico»/«diagnosticar» en construcciones que podrían anticipar el resultado: 32 casos (4,6 % del dataset). Suficiente para justificar su exclusión en escenarios de evaluación zero-shot textual donde el modelo recibe `case_text` como entrada.

## C.10 Validación experta y campos de calidad

Tres campos de validación experta, **no intercambiables**:

- **`label_source = ontology`** (97,89 % del dataset): la etiqueta diagnóstica L3 final está mapeada a la ontología jerárquica validada con revisión experta.
- **`diagnosis_source = expert_v3`** (98,38 % del dataset): el razonamiento textual asociado al diagnóstico está respaldado por la tercera revisión experta de la Dra. Taberner.
- **`rosa_verified = True`** (84,39 % del dataset): *flag* operativo que indica que la imagen ha pasado revisión visual explícita como ejemplo representativo del diagnóstico asignado.

La cifra que el TFG y los reportes técnicos asocian a «validación experta del dataset» es la primera (97,89 %), **no** la tercera (figura `fig_eda_calidad.png`).

## C.11 Caracterización temporal y de exclusiones

### Distribución temporal

Rango 2011–2026 (más de quince años de publicación continuada). Año de mayor aportación: 2017 con 87 imágenes. El perfil agregado sugiere agrupación natural por décadas (2011–2019 y 2020–2026), útil como estratificación temporal de robustez (figura `fig_eda_temporal.png`).

### Razones de exclusión

El pipeline descartó 80 imágenes del candidato original (`excluded_images.csv`).

| Razón | N |
|---|---:|
| `not_found` | 35 |
| `not_derm_user_confirmed` | 19 |
| `md5_matches_solution` | 10 |
| `dermoscopy_pending` | 9 |
| `dedup_same_md5` | 6 |
| `not_derm_visual_review` | 1 |
| **Total** | **80** |

La razón más frecuente es `not_found` (35 imágenes, páginas no recuperables). Le siguen `not_derm_user_confirmed` (19), `md5_matches_solution` (10, fuga directa con el panel de solución, susceptible de *leakage* diagnóstico), `dermoscopy_pending` (9), `dedup_same_md5` (6) y `not_derm_visual_review` (1). El registro íntegro permite la reproducción exacta del filtrado.
