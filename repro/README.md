# Anexo A — Mapa de tareas experimentales y estructura del repositorio

> Material complementario del TFG EPS0270 «Evaluación de modelos fundacionales para Dermatología Clínica» (A. Contestí Coll, UIB EPS, 2025–2026). Corresponde al antiguo Anexo A de la memoria; se publica aquí como parte del repositorio reproducible `dermapixel-tfg-EPS0270`.

Este documento recoge el conjunto de tareas experimentales reproducibles ejecutadas durante el trabajo y la estructura del repositorio asociado. La finalidad es habilitar la reproducción independiente de las cifras del capítulo de resultados a partir del código publicado y de los *checkpoints* de modelo.

## A.1 Estructura del repositorio

El repositorio se organiza en directorios que encapsulan responsabilidades disjuntas.

| Directorio | Contenido |
|---|---|
| `classification/` | Pipeline de sondeo lineal y ajuste fino. Carga de datasets, modelos, métricas. |
| `classification/panderm_model/` | *Wrappers* de encoders fundacionales (PanDerm, DINOv2, DermLIP). |
| `classification/panderm_model/downstream/eval_features/` | Sondeo lineal, regresión logística L-BFGS, *bootstrap*. |
| `classification/datasets/` | Adaptadores por dataset, *splits* canónicos. |
| `classification/models/` | *Builder* de modelos, *timm wrappers*. |
| `segmentation/` | Pipeline de segmentación. SAM2.1 (FT del decodificador), CAE-seg, dataset ISIC2018. |
| `ontology/` | Vocabulario L1/L2/L3, mapeos por dataset, scripts de armonización. |
| `data/` | Índices locales con *splits*, embeddings precomputados. |
| `datasets/` | Datasets y dataset DermapixelAI 1.0. |
| `results/` | Reportes técnicos por experimento (formato Markdown). |
| `output/` | Salidas de evaluación: predicciones, máscaras, *checkpoints*. |
| `tfg_figures/` | Figuras del documento, generadas por scripts reproducibles. |

## A.2 Mapa de tareas experimentales

Las cifras del capítulo de resultados pueden reconstruirse íntegramente a partir de estos *outputs* sin reejecutar los experimentos.

| Tarea | Referencia | *Output* principal |
|---|---|---|
| Sondeo lineal PanDerm Base/Large (10 ds) | §LP PanDerm | `RESULTADOS_TFG.md` |
| Benchmark CNN vs encoders (3 ds) | §Benchmark | `results/CNN_BASELINE_RESULTS.md` |
| Ajuste fino PanDerm Base (4 ds) | §FT | `RESULTADOS_TFG.md` |
| Eficiencia en etiquetas (HAM, PAD) | §Label-efficiency | `RESULTADOS_TFG.md` |
| Segmentación SAM2.1 decoder FT (ISIC2018) | §Segmentación | `results/SEG_GENERALIZATION_RESULTS.md` |
| Zero-shot DermLIP (3 prompts, 2 tok.) | §Zero-shot | `results/zs_concepts_results.md` |
| LLM multimodales (12 modelos, 3 ds) | §LLM | `results/{GPT4O,GEMINI,MEDGEMMA,BLIP2}_RESULTS.md` |
| MedGemma 27B + LoRA (HAM, PAD, DDI) | §LLM | `results/MEDGEMMA_FT_RESULTS.md` |
| Equidad por fototipo (5 enc., 6 FST) | §Equidad | `results/FITZPATRICK_FULL_RESULTS.md` |
| Equidad cruzada Light/Dark | §Equidad | `tfg_figures/fairness_5models/summary_5models.md` |
| Contraste DeLong AUROC malignidad (5 enc.) | §Equidad DeLong | `output/fitzpatrick17k_fairness/delong_malignancy_results.json` |
| Ensemble seguridad melanoma | §Ensemble | `output/ensemble_eval/ENSEMBLE_REPORT.md` |
| SAE Large + CBM SkinCon | Anexo F (SAE) | `results/{SAE_LARGE,SKINCON_CBM}_RESULTS.md` |
| Auditoría leakage MD5 (12 ds vs Derm1M) | §Auditoría | `output/leakage_audit_12ds_vs_derm1m/report.md` |

## A.3 Comandos de ejecución

Todos los comandos se ejecutan desde la raíz del repositorio sobre el entorno declarado en la memoria (Python 3.12.3, PyTorch 2.11.0, `timm` 0.9.16, CUDA 13.0).

**Extracción de *embeddings* (paso previo al sondeo lineal):**

```bash
python -m classification.panderm_model.downstream.extract_features \
    --model panderm_large \
    --dataset ham10000 \
    --output data/embeddings/ham10000_panderm_large.npz
```

**Sondeo lineal sobre *embeddings* extraídos:**

```bash
python -m classification.panderm_model.downstream.eval_features.linear_probe \
    --embeddings data/embeddings/ham10000_panderm_large.npz \
    --C 1.0 --max_iter 5000 --random_state 42
```

**Ajuste fino supervisado:**

```bash
python -m classification.run_class_finetuning \
    --model panderm_base --dataset ham10000 \
    --opt adamw --weight_decay 0.05 --lr 5e-4 \
    --warmup_epochs 10 --epochs 50 \
    --layer_decay 0.65 --drop_path 0.2 --smoothing 0.1 \
    --mixup 0.8 --cutmix 1.0 --tta 5 --seed 0
```

**Segmentación SAM2.1 (fine-tuning del decodificador):**

> Comando ilustrativo; el script real vive en `segmentation/workers/train.py`. El régimen es FT denso del decoder + prompt encoder (image encoder congelado), sin LoRA.

```bash
python -m segmentation.train_sam2_decoder \
    --dataset isic2018 --lr 1e-4 \
    --weight_decay 0.05 --epochs 50 --batch_size 4 \
    --norm uniform_05 --seed 0
```

**Evaluación zero-shot multimodal:**

```bash
python -m classification.zs_eval \
    --model dermlip_original --dataset ham10000 \
    --prompts derm1m_a3 --tokenizer gpt2_clip
```

## A.4 Configuraciones de entrenamiento por experimento

### Configuración del protocolo Dermapixel R0 (cabeza supervisada L2) sobre PanDerm Large con adaptación LoRA

Los 524 288 parámetros entrenables corresponden a 39 K en la cabeza FC 1024 → 38 y 485 K en los ocho módulos LoRA insertados en las capas lineales de los dos últimos bloques de transformer.

| Componente | Configuración |
|---|---|
| Encoder base | PanDerm Large (1 024 dim, 303,85 M params, congelado) |
| LoRA | r = 16, α = 32, *dropout* 0,1 |
| Capas adaptadas | `blocks.22` y `blocks.23`: `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` (8 capas) |
| Cabeza supervisada | FC 1024 → 38 (L2 con queratinización consolidada) |
| Parámetros entrenables | 524 288 (0,17 % del total) |
| Optimizador | AdamW, lr_head = 10⁻³, lr_LoRA = 5 × 10⁻⁴, wd = 10⁻⁴ |
| *Scheduler* | Cosine con *warmup* de una época |
| Pérdida | *Cross-entropy* con `class_weight=balanced` |
| *Batch size* | 16 |
| Épocas | 15 |
| Augmentaciones *train* | RandomResizedCrop(0,7–1,0), hflip, vflip (p = 0,3), ColorJitter (0,1) |
| Selección | Mejor BAcc de validación; reporte adicional de la media de las últimas cinco épocas |
| Semillas | {42, 43, 44} |

### Configuración de las cinco iteraciones de la rama contrastiva de Dermapixel R0

Entrenadas sobre DermapixelAI 1.0. Las versiones v1–v4 comparten el mismo *split* per-case (854 *train* / 107 *val* / 101 *test*) y el mismo manifiesto SHA-256. La versión v5 amplía el conjunto de entrenamiento con 15 128 imágenes externas (PAD-UFES-20 y DermNet) reponderadas mediante *Weighted Random Sampler* con ratio efectivo 6× en favor del subconjunto castellano original, manteniendo los conjuntos de validación y test idénticos.

| Versión | Image encoder | Text encoder | Train |
|---|---|---|---|
| v1 | PanDerm-Large frozen | E5-base *full FT* | 854 castellano |
| v2 | PanDerm-Large frozen | E5-base + LoRA r = 16, α = 32 | 854 castellano |
| v3 | PanDerm-Large + LoRA r = 16 (últimos 4/24 bloques) | E5-base + LoRA r = 16, α = 32 | 854 castellano |
| v4 | PanDerm-Large + LoRA r = 32 (últimos 12/24 bloques) | E5-base + LoRA r = 32, α = 64 | 854 castellano |
| v5 | PanDerm-Large + LoRA r = 32 (últimos 12/24 bloques) | E5-base + LoRA r = 32, α = 64 | 854 cast. + 15 128 ext. |

### Receta de ajuste fino denso parcial sobre DermapixelAI 1.0

PanDerm Large con las dos últimas capas descongeladas (25,2 M parámetros, 8,3 % del encoder) más cabeza *fully-connected*.

| Componente | Configuración |
|---|---|
| Optimizador | AdamW, lr_cabeza = 10⁻³, lr_encoder = 10⁻⁵, wd = 10⁻⁴ |
| *Scheduler* | Cosine con *warmup* de una época |
| *Batch size* | 16 |
| Épocas | 10 |
| Pérdida | *Cross-entropy* con `class_weight='balanced'` |
| Augmentaciones *train* | RandomResizedCrop, hflip, vflip (p = 0,3), ColorJitter |
| Selección | Mejor BAcc de validación |

## A.5 Versiones de software y semillas

El entorno de ejecución se fija explícitamente. Las semillas de `torch`, `numpy` y `random` se inicializan a 0 en los protocolos de *fine-tuning* y segmentación, y el parámetro `random_state` se fija a 42 en los basados en `scikit-learn`. La variable `WANDB_MODE` se fija a `disabled` y `CUDA_VISIBLE_DEVICES` a `0`. La especificación completa del entorno se documenta en el archivo `environment.yml` del repositorio.

Para los experimentos sobre DermapixelAI 1.0 se distinguen dos usos de semilla. El protocolo Dermapixel R0 (cabeza L2 con LoRA) reporta sobre el conjunto `{42, 43, 44}` (media y desviación entre semillas); la persistencia de *checkpoints* para el prototipo congela únicamente la semilla 42 con el fin de fijar un artefacto reproducible.

**Normalización de imagen.** Los módulos basados en PanDerm comparten la misma transformación de evaluación: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → `Normalize` con la media y desviación estándar de ImageNet (`mean = (0,485, 0,456, 0,406)`, `std = (0,229, 0,224, 0,225)`). En *train* se sustituye el `Resize`/`CenterCrop` por `RandomResizedCrop` más volteos horizontal/vertical y `ColorJitter`, según la receta de cada experimento (sección A.4). La segmentación SAM2.1 emplea la normalización `uniform_05` declarada en su comando de la sección A.3.

## A.6 Checkpoints publicados

| *Checkpoint* | Modelo base | Tamaño | Métrica de referencia |
|---|---|---|---|
| `panderm_base_ft_ham10000.pth` | PanDerm Base + TTA | ~350 MB | Acc HAM10000 0,920 |
| `panderm_large_ft_ham10000.pth` | PanDerm Large | 1,2 GB | Acc HAM10000 0,919 |
| `panderm_large_ft_padufes.pth` | PanDerm Large | 1,2 GB | Acc PAD-UFES 0,755 |
| `sam2_seg_isic2018.pth` | SAM2.1-L (decoder FT) | ~17 MB | Dice 0,947 |
| `sae_large_panderm.pth` | PanDerm Large + SAE | ~200 MB | Esparsidad 16,5 % |
| `medgemma_27b_lora_ham10000.pth` | MedGemma 27B + LoRA | ~320 MB | Acc HAM10000 0,802 |

El acceso a los *checkpoints* se realiza mediante el repositorio de releases del proyecto, sujeto a la licencia declarada en el documento de entrega (ver [`datasheet/`](../datasheet/README.md)). La verificación de integridad se realiza por hash SHA-256 publicado junto a cada *checkpoint*.

## A.7 Artefactos del prototipo y recetas de construcción

Esta sección documenta los artefactos reproducibles que alimentan el prototipo **DermApIxel** ([dermapixel.eu](https://dermapixel.eu)). El prototipo integra los módulos **M1–M11 más M4-bis**; su descripción funcional completa figura en [`prototype/README.md`](../prototype/README.md) y en el capítulo correspondiente de [`MemoriaTFG.pdf`](../MemoriaTFG.pdf). Aquí se recogen únicamente las recetas necesarias para reconstruir los artefactos de inferencia a partir del código publicado, sobre el servidor GPU on-prem (NVIDIA DGX Spark, chip GB10, aarch64). Todas las rutas se expresan de forma genérica relativa al directorio de proyecto `~/panderm/`.

### A.7.1 Módulo M4-bis: índice FAISS de recuperación visual en castellano

M4-bis es la **búsqueda visual desplegada en producción**: opera **imagen→imagen** sobre los *embeddings* de imagen de PanDerm Large indexados con FAISS. Sustituye, para el caso de uso real, a la rama contrastiva texto→imagen de Dermapixel R0 (SpanDerm-CLIP), que **no se desplegó** porque su métrica favorable no generaliza a régimen de consulta real (la *split* es limpia, disjunta por caso e imagen, lo que descarta contaminación como causa).

Receta de construcción (script de evidencia: `build_m4bis_faiss.py`):

| Componente | Configuración |
|---|---|
| Encoder de imagen | PanDerm Large (1 024 dim), *embeddings* precomputados |
| Normalización del vector | L2-normalize por fila (similitud coseno vía producto interno) |
| Índice | FAISS `IndexFlatIP` (búsqueda exacta por producto interno) |
| Conjunto indexado | 874 *embeddings* de entrenamiento de DermapixelAI 1.0 (más 36 de test, para validación interna) |
| Metadatos por vector | Información clínica del caso y texto clínico en castellano (`case_text`) |
| Salida | `~/panderm/output/m4bis_faiss_dermapixel/` (`faiss_index_train.bin`, `metadata_train.json`, e índices/metadatos de test) |
| Latencia *warm* | ~0,5 ms por consulta *top-k* (orden de magnitud reportado en la memoria) |

### A.7.2 Persistencia de *checkpoints* del prototipo

Los módulos M9 (cabeza L2 castellana de Dermapixel R0) y M10 (multitarea *Seven-Point Checklist* + melanoma) se persisten para el servicio de inferencia mediante `persist_models_for_prototype.py`, que reejecuta la lógica de entrenamiento con semilla 42 y guarda el *best state* seleccionado por mejor BAcc de validación. Cada artefacto se acompaña de su mapeo de etiquetas y de sus métricas de test:

- M9: `best_seed42.pth` (`state_dict` del encoder con LoRA más cabeza FC L2), `best_seed42_l2_mapping.json`, `best_seed42_metrics.json`. El *checkpoint slim* resultante (sólo pesos LoRA y cabeza, ≈524 K parámetros) ocupa ~2,3 MB tras `slim_checkpoints.py`.
- M10: `best_model.pth` (ocho cabezas más LoRA), `best_concept_mapping.json`, `best_metrics.json`.

La política general de *checkpoints* (selección por mejor validación, verificación SHA-256, releases) es la descrita en la sección A.6; esta subsección sólo añade la variante de semilla única para los artefactos en producción.

## A.8 Inventario de scripts de evidencia (DermapixelAI 1.0)

La tabla siguiente traza cada tarea experimental sobre DermapixelAI 1.0 y cada artefacto del prototipo a su **script real** y a su salida principal verificada en el repositorio de pipeline. Los nombres de fichero se conservan tal cual para facilitar la trazabilidad.

| Tarea / artefacto | Script | Salida principal |
|---|---|---|
| Análisis exploratorio del dataset | `dermapixel_eda.py` | `eda/eda_report.md` |
| Sondeo lineal (L1/L2/L3) | `dermapixel_lp_eval.py` | `dermapixel_v1_lp_summary.csv` |
| Zero-shot DermLIP | `dermapixel_zs_eval.py` | `dermapixel_v1_zs_summary.csv` |
| Zero-shot jerárquico | `dermapixel_hierarchical_zs.py` | `dermapixel_v1_hierzs_summary.csv` |
| *k*-NN sobre FAISS | `dermapixel_faiss_knn.py` | `dermapixel_v1_faiss_summary.csv` |
| Baselines DINOv2 / CLIP-L | `dermapixel_dinov2_clipl.py` | `dermapixel_v1_extra_summary.csv` |
| Ajuste fino L1 | `dermapixel_ft_l1.py` | `dermapixel_v1_ft_l1_summary.csv` |
| Ajuste fino L2 / L3 | `dermapixel_ft_l2l3.py` | `dermapixel_v1_ft_l2_summary.csv`, `dermapixel_v1_ft_l3_summary.csv` |
| Ajuste fino con *focal loss* | `dermapixel_focal_loss.py` | `dermapixel_v1_focal_summary.csv` |
| TTA | `dermapixel_tta_eval.py` | `dermapixel_v1_tta_summary.csv` |
| Ensemble | `dermapixel_ensemble_eval.py` | `dermapixel_v1_ensemble_summary.csv` |
| Evaluación A·B·E (k-NN *k*∈{1,5,10}; MLP 2 capas con *class weighting*; CV 5-fold *case-aware* sobre el LP) | `dermapixel_abe_eval.py` | `dermapixel_v1_abe_summary.csv` |
| Dermapixel R0, cabeza L2 LoRA (multisemilla) | `dermapixel_spanderm_v0_multiseed.py` | `dermapixel_v1_spanderm_v0_multiseed_summary.csv` |
| Auditoría de *leakage* MD5 | `dermapixel_md5_audit.py` | `dermapixel_v1_md5_report.md` |
| *Seven-Point Checklist* + SAE (E1–E4) | `derm7pt_sae_e1.py` … `derm7pt_sae_e4.py` | `q4_derm7pt/{report,e2_report,e3_report}.md`, `q4_derm7pt/e4_summary.csv` |
| Construcción del índice M4-bis | `build_m4bis_faiss.py` | `~/panderm/output/m4bis_faiss_dermapixel/` |
| Persistencia de M9 y M10 para el prototipo | `persist_models_for_prototype.py` | `best_seed42.pth`, `derm7pt_sae_e4/best_model.pth` |
| Adelgazamiento de *checkpoints* | `slim_checkpoints.py` | *checkpoints slim* (M9 ~2,3 MB) |
| Integración de módulos (M4-bis, M9, M10, M11) | `prototype_m4bis_faiss.py`, `prototype_m9_spanderm.py`, `prototype_m10_concepts.py`, `prototype_m11_ensemble.py` | módulos del servicio de inferencia |
| Motor de inferencia y servidor | `pipeline_remote.py`, `server_remote.py` | servicio de inferencia (M1–M11 + M4-bis) |

El ensemble M11 (`prototype_m11_ensemble.py`) combina las probabilidades de M1, M7, M9 y M4-bis por niveles L1/L2/L3 con pesos derivados de las AUROC reportadas en el capítulo de resultados; el detalle de los pesos y del *banner* de consenso figura en [`prototype/README.md`](../prototype/README.md). Las cifras agregadas y sus intervalos de confianza están en [`tables/`](../tables/README.md).
