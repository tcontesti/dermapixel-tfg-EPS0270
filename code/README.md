# code/ — Scripts de los experimentos

Código de los experimentos del TFG EPS0270, **sanitizado para repositorio público** y organizado por tarea. Cada script reproduce una de las cifras/tablas del cuerpo de la memoria; la correspondencia experimento→script está también en [`../repro/`](../repro/README.md) y las cifras que produce, en [`../tables/`](../tables/README.md), [`../ablations/`](../ablations/README.md), [`../sae/`](../sae/README.md) y [`../datasheet/`](../datasheet/README.md).

> **Material de terceros no incluido.** Los pesos de los modelos base (PanDerm, DermLIP, SAM2/MedSAM2, SigLIP, BiomedCLIP, DINOv2, MedGemma) y los datasets dermatológicos de terceros **no se redistribuyen**: obtenlos de sus publicaciones y repositorios oficiales según su licencia. El dataset propio DermapixelAI 1.0 se rige por su [datasheet](../datasheet/README.md) (CC BY-NC-SA 4.0).

## Entorno

- **Python** 3.10+ (probado en aarch64 sobre servidor GPU on-prem; el código es agnóstico de plataforma).
- **Dependencias:** ver [`requirements.txt`](requirements.txt) (`pip install -r requirements.txt`). No todos los scripts necesitan todos los paquetes.
- **Variables de entorno (rutas configurables):**

  | Variable | Por defecto | Significado |
  |---|---|---|
  | `DERMAPIXEL_ROOT` | `./data` | Raíz de datos. Los scripts esperan bajo ella: `datasets/` (datasets y DermapixelAI 1.0), `weights/` (pesos base), `output/` (resultados y embeddings precalculados). Las salidas se escriben en `output/`. |

  ```bash
  export DERMAPIXEL_ROOT=/ruta/a/tus/datos
  python frozen_eval/dermapixel_lp_eval.py
  ```

  Ningún script contiene rutas absolutas internas, hostnames, IPs ni credenciales: la infraestructura (p. ej. `RABBITMQ_URL`, `CORS_ORIGINS`, claves de API de proveedores LLM) se lee de variables de entorno sin valores por defecto reales.

## Organización y mapa script → cifra

### `frozen_eval/` — Evaluación sobre el codificador congelado
| Script | Qué hace | Entradas | Salida (`$DERMAPIXEL_ROOT/output/`) | Produce |
|---|---|---|---|---|
| `dermapixel_lp_eval.py` | Sondeo lineal (LP) de PanDerm Base/Large sobre DermapixelAI 1.0 (L1/L2/L3) | embeddings/dataset DermapixelAI 1.0, pesos PanDerm | `dermapixel_v1_lp/` | tabla LP de R0 ([ablations](../ablations/ablaciones-complementarias.md)) |
| `dermapixel_faiss_knn.py` | Clasificación por *k*-NN sobre índice FAISS de embeddings | embeddings DermapixelAI 1.0 | `dermapixel_v1_faiss/` | ablación *k*-NN/FAISS (J.7) |
| `dermapixel_abe_eval.py` | Evaluación A·B·E sobre los embeddings congelados: k-NN (*k*∈{1,5,10}, coseno), MLP de 2 capas con *class weighting*, y CV 5-fold *case-aware* sobre el LP | `dermapixel_v1_lp/…_embeddings.npy` | `dermapixel_v1_abe/` | material de [ablations](../ablations/README.md) |
| `dermapixel_dinov2_clipl.py` | Baselines con codificadores generalistas (DINOv2, CLIP-L) | dataset DermapixelAI 1.0, pesos DINOv2/CLIP-L | `dermapixel_v1_extra/` | ablación de encoders generalistas (J.6) |

### `fine_tuning/` — Ajuste fino
| Script | Qué hace | Entradas | Salida | Produce |
|---|---|---|---|---|
| `dermapixel_ft_l1.py` | *Fine-tuning* de la cabeza L1 sobre PanDerm | dataset, pesos PanDerm | `dermapixel_v1_ft_l1/` | FT multinivel (cap. DermapixelAI) |
| `dermapixel_ft_l2l3.py` | *Fine-tuning* de las cabezas L2 y L3 | dataset, pesos PanDerm | `dermapixel_v1_ft_{l2,l3}/` | FT multinivel |
| `dermapixel_focal_loss.py` | Ablación de función de pérdida (*focal* vs CE) | dataset, embeddings | `dermapixel_v1_focal/` | ablación de pérdida (J.4) |

### `tta_ensemble/` — Aumento en test y ensembles
| Script | Qué hace | Entradas | Salida | Produce |
|---|---|---|---|---|
| `dermapixel_tta_eval.py` | Evaluación con *test-time augmentation* | dataset, modelo entrenado | `dermapixel_v1_tta/` | ablación TTA (J.5) |
| `dermapixel_ensemble_eval.py` | Ensemble por complementariedad de modelos | salidas de varios modelos | `dermapixel_v1_ensemble/` | ensemble (J.N.5 / módulo M11) |

### `zero_shot/` — Clasificación guiada por texto
| Script | Qué hace | Entradas | Salida | Produce |
|---|---|---|---|---|
| `dermapixel_zs_eval.py` | *Zero-shot* DermLIP v2 (clase abierta, glosario ES→EN) | dataset, pesos DermLIP v2 | `dermapixel_v1_zs/` | zero-shot (cap. DermapixelAI / módulo M6) |
| `dermapixel_hierarchical_zs.py` | *Zero-shot* jerárquico (plano vs cascada vs condicional) | dataset, pesos DermLIP v2 | `dermapixel_v1_hierzs/` | ablación *zero-shot* jerárquico (J.8) |

### `dermapixel_r0/` — Dermapixel R0 (adaptación al castellano)
| Script | Qué hace | Entradas | Salida | Produce |
|---|---|---|---|---|
| `dermapixel_spanderm_v0.py` | Adaptación supervisada LoRA L2 castellana (rama desplegada como M9) | dataset, pesos PanDerm | `dermapixel_v1_spanderm_v0/` | [Dermapixel R0](../ablations/ablacion-dermlip-en-y-r0-contrastiva.md) |
| `dermapixel_spanderm_v0_multiseed.py` | Repetición multi-semilla {42,43,44} para reporte con desviación | dataset, pesos PanDerm | `dermapixel_v1_spanderm_v0_multiseed/` | tabla L2 de R0 (media ± desv.) |

### `retrieval/` — Recuperación visual del prototipo
| Script | Qué hace | Entradas | Salida | Produce |
|---|---|---|---|---|
| `build_m4bis_faiss.py` | Construye el índice FAISS (`IndexFlatIP`, L2-norm) de M4-bis, imagen→imagen sobre el archivo DermapixelAI con PanDerm Large | embeddings PanDerm Large del archivo | `m4bis_faiss_dermapixel/` | búsqueda visual en producción (M4-bis) |

### `dataset_audit/` — Construcción y auditoría del dataset
| Script | Qué hace | Entradas | Salida | Produce |
|---|---|---|---|---|
| `dermapixel_eda.py` | Análisis exploratorio (modalidad, distribución L1/L2/L3, cola larga, temporal) | DermapixelAI 1.0 | `dermapixel_v1_eda/` | [datasheet](../datasheet/README.md) / [pipeline](../ablations/pipeline-dataset.md) |
| `dermapixel_rosa_verified.py` | Cruce con la validación experta (`rosa_verified`) | dataset + campos de validación | `dermapixel_v1_rosa/` | ablación `rosa_verified` (J.1) |
| `dermapixel_md5_audit.py` | Auditoría de solapamiento por hash MD5 frente a datasets de referencia | DermapixelAI 1.0 + catálogos | `dermapixel_v1_md5_audit/` | originalidad (0 coincidencias) ([pipeline](../ablations/pipeline-dataset.md)) |

### `sae_derm7pt/` — Sparse Autoencoders y conceptos
| Script | Qué hace | Entradas | Salida | Produce |
|---|---|---|---|---|
| `derm7pt_sae_e1.py` | Extracción de *features* del SAE y AUROC por concepto | embeddings PanDerm, SkinCon/Derm7pt | `derm7pt_sae_e1/` | [SAE](../sae/README.md) (E1) |
| `derm7pt_sae_e2.py` | Pesos meta-LP de conceptos | features SAE (E1) | `derm7pt_sae_e2/` | SAE (E2) |
| `derm7pt_sae_e3.py` | Cruce con los conceptos del Grupo A (*Seven-Point Checklist*) | features SAE (E1), Derm7pt | `derm7pt_sae_e3/` | SAE (E3) |
| `derm7pt_sae_e4.py` | LP separado vs multitarea | features SAE | `derm7pt_sae_e4/` | SAE (E4) |

### `prototype/` — Módulos e infraestructura de inferencia
| Script | Qué hace |
|---|---|
| `prototype_m4bis_faiss.py` | Módulo M4-bis: recuperación imagen→imagen sobre el archivo en castellano |
| `prototype_m9_spanderm.py` | Módulo M9: cabeza L2 castellana (PanDerm + LoRA) |
| `prototype_m10_concepts.py` | Módulo M10: *Seven-Point Checklist* + cabeza de melanoma |
| `prototype_m11_ensemble.py` | Módulo M11: ensemble de consenso (banner de varios clasificadores) |
| `pipeline_remote.py` | Motor de inferencia que integra los módulos M1–M11 |
| `server_remote.py` | Servidor FastAPI que expone el pipeline (HTTP + *worker* RabbitMQ opcional) |

La arquitectura del prototipo en producción (dermapixel.eu) se describe en [`../prototype/README.md`](../prototype/README.md). Los scripts de esta carpeta son el material reproducible de los módulos; la infraestructura real (colas, proxies, autenticación) se configura por entorno.

### `utils/` — Utilidades
| Script | Qué hace |
|---|---|
| `persist_models_for_prototype.py` | Persiste los modelos M9/M10 (mejor validación) para servir el prototipo |
| `slim_checkpoints.py` | Adelgaza los *checkpoints* (elimina estados de optimizador) |

## Nota de reproducibilidad

Solo se han modificado, respecto a los scripts de ejecución, las **rutas, la infraestructura y las credenciales** (parametrizadas por entorno); los **algoritmos, hiperparámetros, semillas y métricas son idénticos**. Cada script documenta en su *docstring* sus entradas y salidas. Para el mapa completo de tareas, recetas, semillas y normalizaciones, ver [`../repro/`](../repro/README.md).
