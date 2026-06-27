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
