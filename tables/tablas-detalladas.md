# Anexo B — Tablas detalladas de resultados

> Material complementario del TFG EPS0270. Corresponde al antiguo Anexo B de la memoria. Recoge desgloses y tablas adicionales al capítulo de resultados: cifras por clase de sondeo lineal y ajuste fino sobre HAM10000 y PAD-UFES-20, caracterización por fototipo cutáneo en la formulación de 114 patologías sobre Fitzpatrick17k, y la tabla extendida de evaluación cruzada Light↔Dark.

## B.1 Desglose por clase sobre HAM10000 (GPT-4o zero-shot)

La columna «Predicciones» indica el número total de imágenes asignadas a cada clase por el modelo, revelando el sesgo léxico. *Recall*: proporción de imágenes de la clase correctamente identificadas. *Predicciones*: número de imágenes asignadas a la clase por el modelo (debe contrastarse con N_test para detectar sesgos sistemáticos).

| Clase | N_test | Recall (GPT-4o) | Predicciones (GPT-4o) | Recall (GPT-4o-mini) |
|---|---:|---:|---:|---:|
| actinic keratosis | 35 | 0,000 | 2 | 0,086 |
| basal cell carcinoma | 44 | 0,477 | — | 0,136 |
| seborrheic keratosis | 107 | 0,308 | — | 0,654 |
| dermatofibroma | 8 | 0,750 | 284 | 0,000 |
| melanoma | 70 | 0,686 | 299 | 0,571 |
| melanocytic nevus | 951 | 0,505 | — | 0,196 |
| vascular lesion | 17 | 0,529 | — | 0,588 |

La asimetría entre N_test y predicciones es máxima en dermatofibroma (N_test = 8, predicciones 284: factor 35,5×) y melanoma (N_test = 70, predicciones 299: factor 4,3×). El comportamiento es diagnóstico de un sesgo léxico hacia categorías con presencia elevada en el *prompt*, no hacia categorías mayoritarias en el dataset.

## B.2 Desglose por clase del ajuste fino MedGemma 27B sobre HAM10000

| Clase | N test | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| actinic keratosis | 35 | 0,32 | 0,26 | 0,29 |
| basal cell carcinoma | 44 | 0,34 | 0,25 | 0,29 |
| dermatofibroma | 8 | 0,00 | 0,00 | 0,00 |
| melanocytic nevus | 951 | 0,85 | 0,99 | 0,92 |
| melanoma | 70 | 0,38 | 0,36 | 0,37 |
| seborrheic keratosis | 107 | 0,67 | 0,04 | 0,07 |
| vascular lesion | 17 | 0,00 | 0,00 | 0,00 |

Tras la adaptación con LoRA, el modelo concentra el rendimiento en la clase mayoritaria (nevus melanocítico: *recall* 0,99) a costa de las clases menos representadas (dermatofibroma y vascular lesion: F1 0,00). Coherente con la accuracy de 0,802 pese a una BAcc de solo 0,270.

## B.3 Equidad sobre Fitzpatrick17k: formulación de 114 patologías

Rendimiento por fototipo Fitzpatrick de PanDerm Large sobre la formulación de 114 patologías (figura `fig_fairness_fst.png`). El rendimiento crece de FST I a FST V y desciende bruscamente en FST VI.

| Fototipo (FST) | N test | Acc | BAcc | W-F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| I | 299 | 0,515 | 0,522 | 0,509 | — |
| II | 470 | 0,519 | 0,524 | 0,513 | — |
| III | 325 | 0,640 | 0,590 | 0,636 | — |
| IV | 261 | 0,670 | 0,602 | 0,654 | — |
| V | 169 | 0,692 | 0,614 | 0,680 | — |
| VI | 73 | 0,479 | 0,474 | 0,462 | — |
| **Global** | | 0,581 | 0,549 | 0,576 | 0,968 |

El *gap* entre FST V (mejor) y FST VI (peor) es de 0,213 pp accuracy. El grano fino de 114 patologías expone una capa de información clínica donde la representación visual de las lesiones sobre fototipos intermedios y altos (III, IV, V) supera a la de fototipos extremos (I, II y VI).

## B.4 Evaluación cruzada Light/Dark sobre 114 patologías

Experimento cruzado entrenamiento/evaluación entre subgrupos *Light* (FST I-III) y *Dark* (FST IV-VI) sobre 114 patologías con PanDerm Large. *All* agrupa el conjunto de entrenamiento completo. C: número de clases efectivas en la evaluación.

| Configuración | N train | N test | C | Acc | BAcc | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Light → Light | 8 845 | 1 094 | 114 | 0,541 | 0,528 | 0,962 |
| Light → Dark | 8 028 | 503 | 100 | 0,473 | 0,453 | 0,944 |
| Dark → Dark | 3 807 | 503 | 100 | 0,616 | 0,519 | 0,948 |
| Dark → Light | 3 958 | 1 094 | 114 | 0,351 | 0,325 | 0,906 |
| All → Light | 12 803 | 1 094 | 114 | 0,559 | 0,547 | 0,965 |
| All → Dark | 11 835 | 503 | 100 | 0,658 | 0,614 | 0,970 |

La asimetría más notable se observa en Dark → Light (BAcc 0,325, AUROC 0,906). La accuracy desciende desde 0,616 en Dark → Dark a 0,351 en Dark → Light, lo que cuantifica la pérdida sistemática asociada al cambio de distribución de fototipo a la inversa de la dirección habitual reportada en la literatura.

## B.5 Resumen comparativo por paradigma sobre HAM10000

Datos etiquetados: tamaño del conjunto de entrenamiento empleado. Coste indicativo en EUR para los modelos de acceso comercial (cifras 2026); 0 EUR para los modelos locales.

| Paradigma | Modelo | Datos etiq. | Acc | AUROC | Coste eval. |
|---|---|---:|---:|---:|---|
| ZS (*prompts* genéricos) | DermLIP v1/v2 | 0 | 0,086 | 0,366 | 0 |
| ZS (*prompts* A3 Derm1M) | DermLIP original | 0 | 0,427 | 0,854 | 0 |
| ZS LLM multimodal | GPT-4o | 0 | 0,485 | — | ~$2,71 |
| ZS LLM multimodal | MedGemma 27B | 0 | 0,665 | — | 0 |
| LP (1% entrenam.) | PanDerm Large | 82 | 0,850 | 0,918 | 0 |
| LP (100% entrenam.) | PanDerm Base | 8 207 | 0,853 | 0,892 | 0 |
| LP (100% entrenam.) | PanDerm Large | 8 207 | 0,888 | 0,954 | 0 |
| LP (100% entrenam.) | SigLIP-Large | 8 207 | 0,900 | 0,971 | 0 |
| FT LoRA LLM | MedGemma 27B + LoRA | 8 207 | 0,802 | — | 0 |
| FT (receta paper) + TTA | PanDerm Base | 8 207 | 0,920 | 0,978 | 0 |

Figuras asociadas al dataset armonizado de 72 654 imágenes (clasificador unificado): `tradeoff_classes_accuracy.png` (compromiso clases/accuracy; el nivel L2 de 43 clases es el punto operativo más equilibrado), `unified_l1_distribution.png` (distribución por categoría L1) y `unified_l2_perclass_accuracy.png` (accuracy por clase L2, 43 entradas).

## B.6 Composición del conjunto de entrenamiento del SAE Large

El SAE Large (1 024 → 16 384 *features*), descrito en [`sae/`](../sae/README.md), se entrena sobre la concatenación de siete datasets dermatológicos.

| Dataset | Imágenes |
|---|---:|
| HAM10000 | 10 015 |
| BCN20000 | 12 413 |
| Dermnet | 19 559 |
| PAD-UFES-20 | 2 298 |
| Fitzpatrick17k | 3 887 |
| DDI | 647 |
| HIBA | 1 635 |
| **Total** | **50 454** |

## B.7 Segmentación: generalización fuera de dominio (Dice e IoU)

Dice e IoU (media ± desv. estándar) sobre ISIC2017 (N = 600) y PH2 (N = 200).

| Modelo | Dataset | Dice | IoU |
|---|---|---|---|
| SAM2 *zero-shot* (bbox) | ISIC2017 | 0,797 ± 0,158 | 0,686 ± 0,183 |
| SAM2 *zero-shot* (bbox) | PH2 | 0,886 ± 0,199 | 0,830 ± 0,198 |
| SAM2.1-L (decoder FT) ISIC2018 | ISIC2017 | 0,945 ± 0,046 | 0,898 ± 0,065 |
| SAM2.1-L (decoder FT) ISIC2018 | PH2 | 0,960 ± 0,022 | 0,925 ± 0,039 |
| CAE-seg (PanDerm-L) FT ISIC2018 | ISIC2017 | 0,879 ± 0,139 | 0,803 ± 0,166 |
| CAE-seg (PanDerm-L) FT ISIC2018 | PH2 | 0,924 ± 0,066 | 0,865 ± 0,098 |

## B.8 Equidad por fototipo: tablas complementarias

### Equidad sobre Fitzpatrick17k completo (rendimiento global, dos formulaciones)

Mejor valor por columna en **negrita**.

| Modelo | Acc (114) | BAcc (114) | AUROC (114) | W-F1 (114) | Acc (3 mal.) | BAcc (3 mal.) | AUROC (3 mal.) | W-F1 (3 mal.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PanDerm Large | **0,581** | 0,549 | 0,968 | **0,576** | 0,835 | 0,699 | 0,908 | 0,828 |
| PanDerm Base | 0,500 | 0,478 | 0,955 | 0,491 | 0,819 | 0,657 | 0,882 | 0,807 |
| DermLIP v2 (visual) | 0,572 | **0,549** | **0,971** | 0,569 | **0,852** | **0,721** | **0,909** | **0,846** |
| DINOv2 ViT-L/14 | 0,519 | 0,485 | 0,947 | 0,516 | 0,809 | 0,656 | 0,854 | 0,801 |
| BiomedCLIP | 0,334 | 0,321 | 0,910 | 0,323 | 0,799 | 0,590 | 0,830 | 0,776 |

### Evaluación cruzada Light↔Dark (BAcc, formulación de 3 clases)

L: *Light*; D: *Dark*; All: ambos grupos.

| Modelo | L→L | L→D | D→D | D→L | All→L | All→D |
|---|---:|---:|---:|---:|---:|---:|
| PanDerm Large | 0,699 | 0,589 | 0,701 | 0,667 | 0,726 | 0,627 |
| PanDerm Base | 0,673 | 0,557 | 0,621 | 0,629 | 0,673 | 0,594 |
| DermLIP v2 (visual) | **0,729** | **0,651** | 0,667 | **0,683** | **0,745** | **0,682** |
| DINOv2 ViT-L/14 | 0,655 | 0,523 | 0,625 | 0,569 | 0,682 | 0,589 |
| BiomedCLIP | 0,595 | 0,532 | 0,571 | 0,551 | 0,594 | 0,565 |

### Test de DeLong pareado entre encoders (AUROC binaria de malignidad)

*maligno* frente al resto, test de Fitzpatrick17k: p-valor a dos colas. Valores no significativos al nivel nominal 0,05 *en cursiva*. *Provenance*: `delong_malignancy_results.json`.

| | DermLIP v2 | PanDerm Base | DINOv2 | BiomedCLIP |
|---|---|---|---|---|
| PanDerm Large | *0,55* | 1,2×10⁻³ | 4,2×10⁻⁵ | 8,7×10⁻⁹ |
| DermLIP v2 | — | 7,3×10⁻³ | 2,4×10⁻³ | 6,0×10⁻⁷ |
| PanDerm Base | — | — | *0,12* | 1,2×10⁻⁴ |
| DINOv2 | — | — | — | *0,066* |

## B.N Tablas desplazadas del cuerpo (condensación paper-style v6)

Las siguientes tablas se reportan aquí para no fragmentar la argumentación del capítulo de resultados; sus cifras-ancla figuran en el cuerpo de la memoria.

### B.N.1 Segmentación: control automático frente a régimen *promptable* (ISIC2018, N=1000)

| Modelo | Régimen | Dice | IoU | Precisión | Recall |
|---|---|---|---|---|---|
| MedSAM2-tiny | *promptable* | **0,9556** | 0,916 | — | — |
| U-Net (ResNet-101) | automático | 0,8811 | 0,804 | 0,856 | 0,942 |

Retirar la caja-*prompt* cuesta 7,5 pp de Dice, entre seis y siete veces más que la elección de arquitectura.

### B.N.2 Zero-shot binario melanoma/no-melanoma con DermLIP v1/v2 (prompts genéricos)

| Dataset | N_test | Acc | BAcc | AUROC | F1-macro |
|---|---|---|---|---|---|
| DDI | 137 | 0,766 | 0,499 | 0,436 | 0,463 |
| MSKCC | 1664 | 0,716 | 0,500 | 0,430 | 0,417 |
| HIBA | 334 | 0,859 | 0,497 | 0,502 | 0,462 |
| Derm7pt clín. | 168 | 0,589 | 0,472 | 0,337 | 0,439 |

### B.N.3 Zero-shot multimodal sobre HAM10000 (N_test=1232, 7 clases)

| Modelo | Tokenizador | Prompts | Acc | AUROC | F1-macro |
|---|---|---|---|---|---|
| DermLIP v1/v2 | PubMedBERT | Genéricos | 0,086 | 0,366 | 0,023 |
| DermLIP v1/v2 | PubMedBERT | Derm1M A3 | — | 0,516 | — |
| DermLIP original | GPT-2/CLIP | Derm1M A3 | **0,427** | **0,854** | — |

### B.N.4 PanDerm Large frente a la mejor CNN supervisada (ConvNeXt-L / EfficientNetV2-L)

| Dataset | CNN Acc | PanDerm Acc | ΔAcc (pp) | CNN AUROC | PanDerm AUROC | ΔAUROC (pp) |
|---|---|---|---|---|---|---|
| HAM10000 | 0,883 | 0,888 | +0,5 | 0,967 | 0,954 | -1,3 |
| PAD-UFES-20 | 0,690 | 0,772 | +8,2 | 0,895 | 0,949 | +5,4 |
| DDI | 0,818 | 0,847 | +2,9 | 0,774 | 0,860 | +8,6 |
