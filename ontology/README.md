# Ontología dermatológica jerárquica (Dra. Rosa Taberner)

Material complementario del TFG EPS0270. Este documento describe en detalle la **ontología clínica de tres niveles** que la **Dra. Rosa Taberner (Hospital Universitari Son Llàtzer)** diseñó y validó, y que vertebra todo el trabajo: es la **lengua común** con la que se armonizan datasets heterogéneos y con la que el clasificador jerárquico (M7) y la adaptación Dermapixel R0 razonan por niveles.

Artefactos en esta carpeta:
- [`ontology_en.csv`](ontology_en.csv) — la taxonomía completa (un diagnóstico por fila: `level_1, level_2, level_3_diagnosis`).
- [`dataset_to_ontology_mapping.csv`](dataset_to_ontology_mapping.csv) — el mapeo de cada clase original de los datasets públicos a la ontología, con nivel de confianza.

---

## 1. Por qué una ontología

Los datasets públicos de dermatología nombran las mismas enfermedades de formas distintas (`nv`, `nevus`, `melanocytic nevus`…), con granularidades incompatibles y solapamientos. Para entrenar y evaluar un clasificador **unificado** hace falta un vocabulario común. En lugar de una lista plana de etiquetas, la Dra. Taberner diseñó una **taxonomía jerárquica de tres niveles** que refleja el razonamiento clínico real: primero la gran familia, luego el grupo y por último el diagnóstico concreto.

Esto permite tres cosas: (i) **armonizar** clases de orígenes distintos; (ii) **evaluar y predecir a distintos niveles de especificidad** (es clínicamente útil acertar la categoría aunque el diagnóstico exacto sea incierto); y (iii) imponer **coherencia *top‑down*** (si L1 es «Tumoral» y L3 es «Melanoma», L2 no puede ser una categoría benigna).

---

## 2. Los tres niveles

| Nivel | Qué es | Cardinalidad |
|---|---|---|
| **L1 — Categoría** | Las cuatro grandes familias dermatológicas | **4** |
| **L2 — Subcategoría** | Grupo clínico-patológico dentro de cada L1 | **42** |
| **L3 — Diagnóstico** | El diagnóstico específico | **372** |

> **Nota sobre conteos (taxonomía viva).** La memoria cita la escala de **diseño** validada con la Dra. Taberner (revisión v3, abril de 2026) como **4 / 43 / 367**. Esta versión congelada que se publica aquí (`ontology_en.csv`) enumera **4 / 42 / 372** tras ajustes posteriores. En el dataset propio **DermapixelAI 1.0** quedan **efectivamente representadas 4 / 38 / 250** clases (con una cola larga: la mayoría de los L3 tienen muy pocas imágenes). Las pequeñas diferencias provienen de revisiones sucesivas de la taxonomía, no de un error.

---

## 3. Las cuatro categorías L1

| L1 | L2 | L3 | Descripción |
|---|---:|---:|---|
| **Infectious** | 5 | 92 | Infecciones e infestaciones (bacterianas, víricas, fúngicas, parasitarias, picaduras). |
| **Inflammatory** | 22 | 173 | Dermatosis inflamatorias y reactivas; es la familia más amplia y heterogénea. |
| **Tumoral** | 9 | 100 | Tumores benignos, premalignos y malignos (cutáneos y anexiales). |
| **Genodermatosis** | 6 | 7 | Genodermatosis (enfermedades cutáneas hereditarias). |

---

## 4. Subcategorías L2 (agrupadas por L1)

### Infectious (5)
Bacterial infections (34 dx) · Viral infections (25) · Fungal infections (20) · Infestations and bites (10) · Parasitic infections (3).

### Inflammatory (22)
Systemic and metabolic (22 dx) · Other inflammatory dermatoses (16) · Exogenous agents (15) · Pigmentation disorders (14) · Eczema and dermatitis (13) · Drug eruptions (10) · Vascular disorders (10) · Psoriasis (9) · Blistering diseases (8) · Hair follicle disorders (8) · Eccrine apocrine glands (6) · Subcutaneous tissue (6) · Neutrophilic dermatoses (5) · Connective tissue (5) · Eosinophilic dermatoses (4) · Humoral reactivity (4) · Sebaceous glands (4) · Acantholytic disorders (4) · Granulomatous dermatoses (3) · Atrophic disorders (3) · Paraneoplastic syndromes (3) · Keratinization disorders (1).

### Tumoral (9)
Benign epithelial tumors (24 dx) · Malignant tumors (18) · Hematologic tumors (12) · Benign tumors (12) · Benign melanocytic neoplasms (11) · Vascular tumors (8) · Benign mesenchymal tumors (7) · Vascular malformations (5) · Precancerous tumors (3).

### Genodermatosis (6)
Epidermal differentiation disorders (2 dx) · Tuberous sclerosis (1) · Neurofibromatosis type 1 (1) · Incontinentia pigmenti (1) · Hereditary palmoplantar keratoderma (1) · Pseudoxanthoma elasticum (1).

*(El listado completo de los 372 diagnósticos L3 está en [`ontology_en.csv`](ontology_en.csv).)*

---

## 5. Ejemplos de la jerarquía completa (L1 → L2 → L3)

```
Tumoral        → Malignant tumors              → Melanoma
Tumoral        → Malignant tumors              → Basal cell carcinoma
Tumoral        → Benign melanocytic neoplasms  → Acquired melanocytic nevus
Tumoral        → Benign epithelial tumors      → Seborrheic keratosis
Tumoral        → Precancerous tumors           → Actinic keratosis
Inflammatory   → Psoriasis                     → Plaque psoriasis
Inflammatory   → Eczema and dermatitis         → Atopic dermatitis
Infectious     → Fungal infections             → Pityriasis versicolor
Infectious     → Parasitic infections          → Cutaneous leishmaniasis
Infectious     → Infestations and bites        → Scabies
```

---

## 6. Cómo se usó para armonizar los datasets

Las clases originales de **11 datasets públicos** se mapearon a esta ontología y el resultado se **revisó con la Dra. Taberner** (documento de revisión, 2026‑04‑06). Resumen del mapeo ([`dataset_to_ontology_mapping.csv`](dataset_to_ontology_mapping.csv)):

| Confianza del mapeo | Clases | % |
|---|---:|---:|
| `exact` (nombre idéntico) | 120 | 42 % |
| `high` (sinónimo claro) | 107 | 37 % |
| `medium` (interpretación razonable) | 46 | 16 % |
| `low` (ambiguo) | 2 | 0,7 % |
| Excluidas (no mapeables, p. ej. tejido normal histopatológico) | 12 | 4 % |
| **Total** | **287** | **100 %** |

Distribución por L1 del corpus armonizado (≈73.900 imágenes): **Tumoral 74,3 %**, **Inflammatory 16,7 %**, **Infectious 8,8 %**, **Genodermatosis 0,2 %**. El fuerte sesgo hacia lo tumoral refleja la composición de los datasets dermatoscópicos públicos (dominados por nevus y melanoma), un sesgo que se documenta y se tiene en cuenta en la evaluación.

---

## 7. Papel en el sistema

- **Evaluación multi-nivel**: las métricas (L1/L2/L3) permiten medir el rendimiento a distinta granularidad.
- **Clasificador jerárquico M7** y **coherencia *top‑down*** del veredicto del prototipo.
- **Dermapixel R0**: la cabeza supervisada en castellano (head L2) y el *ranking* L3 por prototipos coseno operan sobre esta jerarquía.
- Base para conectar, en el futuro, el vocabulario **dermatoscópico** típico/atípico de la Dra. Taberner (ver [`../sae/ground-truth-dermatoscopia-rosa.md`](../sae/ground-truth-dermatoscopia-rosa.md)) con la taxonomía diagnóstica.

La ontología es una **contribución clínica** de la Dra. Rosa Taberner; este repositorio la documenta y la pone a disposición para reproducibilidad. Todos los derechos sobre el diseño taxonómico corresponden a su autora.
