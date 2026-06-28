# Anexo B — Tablas detalladas de resultados

> Material complementario del TFG EPS0270. Corresponde al antiguo Anexo B de la memoria. Recoge desgloses y tablas adicionales al capítulo de resultados: el sondeo lineal completo por dataset y modelo (incluido Kappa de Cohen), el contraste sondeo lineal frente a ajuste fino y la curva de eficiencia en etiquetas, cifras por clase sobre HAM10000 y PAD-UFES-20, la caracterización por fototipo cutáneo en la formulación de 114 patologías sobre Fitzpatrick17k, la tabla extendida de evaluación cruzada Light↔Dark, la comparativa de segmentación por arquitectura *promptable*, el test de DeLong desglosado por *endpoint* y el clasificador unificado jerárquico M7. Todas las cifras trazan al capítulo de Resultados y a los anexos de la memoria ([`MemoriaTFG.pdf`](../MemoriaTFG.pdf)); no se introduce ningún resultado adicional.

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

## B.9 Sondeo lineal de PanDerm sobre los diez datasets externos (tabla completa)

Sondeo lineal (sin reajustar el codificador) de PanDerm Base y Large sobre los diez datasets externos, con las cinco métricas del cuerpo (`tab:lp-panderm`): *accuracy* (Acc), *balanced accuracy* (BAcc), AUROC, F1 ponderada (W-F1) y coeficiente kappa de Cohen. Mejor valor por dataset y métrica en **negrita**. El asterisco (*) marca Dermnet, único dataset con solapamiento exacto frente a Derm1M (100 % por hash MD5; HIBA y MSKCC presentan 0 coincidencias). Los intervalos de confianza celda a celda figuran en [`bootstrap-ci.md`](./bootstrap-ci.md).

| Dataset | Modelo | Acc | BAcc | AUROC | W-F1 | Kappa |
|---|---|---:|---:|---:|---:|---:|
| HAM10000 | Base | 0,853 | 0,381 | 0,892 | 0,829 | 0,709 |
| HAM10000 | Large | **0,888** | **0,575** | **0,954** | **0,880** | **0,814** |
| BCN20000 | Base | 0,659 | 0,292 | 0,855 | 0,615 | 0,447 |
| BCN20000 | Large | **0,702** | **0,382** | **0,903** | **0,676** | **0,527** |
| PAD-UFES-20 | Base | 0,725 | 0,510 | 0,913 | 0,692 | 0,369 |
| PAD-UFES-20 | Large | **0,772** | **0,642** | **0,949** | **0,760** | **0,527** |
| Dermnet* | Base | 0,450 | 0,381 | 0,888 | 0,432 | 0,437 |
| Dermnet* | Large | **0,550** | **0,495** | **0,931** | **0,540** | **0,555** |
| WSI patches | Base | 0,781 | 0,640 | 0,976 | 0,774 | 0,842 |
| WSI patches | Large | **0,868** | **0,792** | **0,991** | **0,871** | **0,896** |
| DDI | Base | **0,847** | 0,714 | 0,827 | 0,836 | 0,482 |
| DDI | Large | **0,847** | **0,764** | **0,860** | **0,846** | **0,535** |
| Derm7pt clín. | Base | 0,762 | 0,675 | 0,808 | 0,736 | 0,397 |
| Derm7pt clín. | Large | **0,798** | **0,732** | **0,869** | **0,784** | **0,506** |
| Derm7pt dermo. | Base | 0,818 | 0,749 | 0,841 | 0,811 | 0,528 |
| Derm7pt dermo. | Large | **0,836** | **0,766** | **0,858** | **0,829** | **0,570** |
| HIBA | Base | 0,883 | 0,595 | 0,894 | 0,853 | 0,275 |
| HIBA | Large | **0,904** | **0,701** | **0,942** | **0,892** | **0,494** |
| MSKCC | Base | 0,721 | **0,654** | 0,746 | 0,720 | 0,309 |
| MSKCC | Large | **0,746** | 0,644 | **0,751** | **0,732** | **0,316** |
| **Media** | Base | 0,750 | 0,559 | 0,865 | 0,730 | — |
| **Media** | Large | **0,791** | **0,649** | **0,901** | **0,781** | — |

PanDerm Large supera a Base en cuatro de las cinco métricas (mejoras medias de +4,1 pp Acc, +9,0 pp BAcc y +3,6 pp AUROC); la única excepción es la BAcc de MSKCC (−1,0 pp). La media agregada resume la tendencia, pero no constituye un *ranking* absoluto: cada dataset es un problema distinto.

## B.10 Sondeo lineal frente a ajuste fino (PanDerm Base)

Comparación del sondeo lineal (LP) con el ajuste fino supervisado (FT) de PanDerm Base sobre cuatro datasets de tamaño dispar (`tab:ft-panderm`). Sobre HAM10000 el FT incorpora *test-time augmentation* con cinco augmentaciones deterministas. ΔBAcc en puntos porcentuales respecto al LP. Mejor valor por dataset en **negrita**.

| Dataset | Modo | Acc | BAcc | AUROC | W-F1 | ΔBAcc |
|---|---|---:|---:|---:|---:|---:|
| HAM10000 | LP | 0,853 | 0,381 | 0,892 | 0,829 | — |
| HAM10000 | FT | **0,920** | **0,852** | **0,978** | **0,923** | +47,1 |
| PAD-UFES-20 | LP | 0,725 | 0,510 | 0,913 | 0,692 | — |
| PAD-UFES-20 | FT | **0,755** | **0,704** | **0,935** | **0,757** | +19,4 |
| DDI | LP | **0,847** | **0,714** | **0,827** | **0,836** | — |
| DDI | FT | 0,774 | 0,693 | 0,682 | 0,780 | −2,1 |
| MSKCC | LP | 0,721 | 0,654 | **0,746** | 0,720 | — |
| MSKCC | FT | **0,731** | **0,684** | 0,719 | **0,735** | +3,0 |

El ajuste fino mejora con datos abundantes (HAM10000 +47,1 pp BAcc; PAD-UFES-20 +19,4 pp) y es contraproducente con pocos (DDI −2,1 pp, compatible con sobreajuste; MSKCC marginal). Regla operativa: *dataset* grande → ajuste fino; pocos datos → codificador congelado.

## B.11 Eficiencia en etiquetas (sondeo lineal con PanDerm Large)

Rendimiento bajo submuestreos estratificados crecientes del conjunto de entrenamiento (`tab:label-eff`). N_train: tamaño efectivo del subconjunto. Mejor valor por columna y dataset en **negrita**.

| % entrenam. | N_train | Acc | BAcc | AUROC | W-F1 |
|---|---:|---:|---:|---:|---:|
| *HAM10000* | | | | | |
| 1 % | 82 | 0,850 | 0,406 | 0,918 | 0,828 |
| 5 % | 410 | 0,870 | 0,481 | 0,932 | 0,855 |
| 10 % | 820 | 0,875 | 0,480 | 0,939 | 0,863 |
| 20 % | 1 641 | 0,882 | 0,547 | 0,951 | 0,873 |
| 50 % | 4 103 | **0,894** | **0,589** | 0,952 | **0,887** |
| 100 % | 8 207 | 0,888 | 0,575 | **0,954** | 0,880 |
| *PAD-UFES-20* | | | | | |
| 1 % | 14 | 0,740 | 0,599 | 0,907 | 0,727 |
| 5 % | 74 | 0,751 | 0,632 | 0,919 | 0,742 |
| 10 % | 149 | 0,738 | 0,593 | 0,923 | 0,724 |
| 20 % | 298 | 0,748 | 0,587 | 0,931 | 0,728 |
| 50 % | 746 | 0,757 | 0,611 | 0,941 | 0,742 |
| 100 % | 1 493 | **0,772** | **0,642** | **0,949** | **0,760** |

La AUROC satura muy pronto (con el 1 % de HAM10000, 82 imágenes, ya alcanza 0,918; en PAD-UFES-20 bastan 14 imágenes para el 95,5 % de la AUROC final), mientras que la BAcc sigue mejorando con el volumen por ser más sensible a las clases minoritarias. Cifras de una única partición y semilla por nivel.

## B.12 Segmentación: comparativa por arquitectura *promptable* (ISIC2018, N=1000)

Cinco arquitecturas *promptable* evaluadas de forma pareada sobre el test canónico de ISIC2018 (N = 1 000, blindado) bajo régimen común (ajuste fino del *decoder* y *promptable encoder*, codificador visual congelado, *prompt* de *bounding box* de referencia; `tab:seg-arquitecturas`). Todas las diferencias son significativas al test de Wilcoxon pareado (p < 10⁻⁶). Mejor Dice en **negrita**.

| Modelo | Arquitectura | Preentr. | Params | Dice | Latencia |
|---|---|---|---:|---:|---:|
| MedSAM2-tiny | SAM2 | médico | 39 M | **0,9556** | 26 ms |
| SAM2.1-tiny | SAM2 | general | 39 M | 0,9517 | 26 ms |
| SAM2.1-Large | SAM2 | general | 224 M | 0,9503 | 96 ms |
| SAM-Med2D | SAM | médico | 271 M | 0,9465 | 20 ms |
| MedSAM v1 | SAM | médico | 94 M | 0,9458 | 74 ms |

Tres conclusiones: (1) la generación SAM2 aporta más que el preentrenamiento médico (los tres SAM2 superan a los SAM v1); (2) el preentrenamiento médico añade una mejora menor, solo dentro de la misma generación; (3) en régimen *box-prompted* el tamaño del *backbone* aporta poco: SAM2.1-tiny iguala a SAM2.1-Large con 5,7× menos parámetros. La mejor configuración (MedSAM2-tiny, 0,9556) combina generación SAM2 y preentrenamiento médico en el *backbone* más compacto. La arquitectura mueve ~1,0 pp y el preentrenamiento médico ~0,4 pp, frente a los 7,5 pp que cuesta retirar la caja-*prompt* (ver B.N.1).

## B.13 Test de DeLong por *endpoint* (AUROC binaria de detección de malignidad)

El contraste de DeLong se aplica a tres *endpoints* binarios. La potencia estadística suficiente para separar modelos solo se alcanza en **Fitzpatrick17k** (226 positivos); en HAM10000 (70 melanomas) y DermapixelAI 1.0 (58 malignos) la falta de potencia impide concluir, lo que **no** equivale a ausencia de efecto. La matriz completa de p-valores pareados del *endpoint* 1 figura en B.8. IC95 % por la varianza de DeLong. Mejor valor por tabla en **negrita**.

### B.13.1 Endpoint 1 — malignidad sobre Fitzpatrick17k (N=1658, 226 malignos)

| Modelo | AUROC | IC95 % (DeLong) |
|---|---:|---|
| PanDerm Large | **0,9488** | [0,9350, 0,9626] |
| DermLIP v2 | 0,9452 | [0,9307, 0,9598] |
| PanDerm Base | 0,9296 | [0,9130, 0,9463] |
| DINOv2 ViT-L/14 | 0,9150 | [0,8932, 0,9367] |
| BiomedCLIP | 0,8915 | [0,8672, 0,9158] |

PanDerm Large y DermLIP v2 son estadísticamente indistinguibles (ΔAUROC = 0,0036, p = 0,55), pero ambos superan de forma significativa a los generalistas y a PanDerm Base (p ≤ 7,3×10⁻³ en los seis contrastes). Tras la corrección de Holm, siete de los diez contrastes se mantienen significativos.

### B.13.2 Endpoint 2 — melanoma sobre HAM10000 (N=1232, 70 melanomas)

| Modelo | AUROC | IC95 % (DeLong) |
|---|---:|---|
| PanDerm Large | **0,9523** | [0,9314, 0,9733] |
| SigLIP-Large | 0,9495 | [0,9245, 0,9745] |
| DermLIP v2 | 0,9463 | [0,9246, 0,9681] |

Los tres codificadores alcanzan AUROC ~0,95 con diferencias < 0,006; ningún contraste pareado resulta significativo tras la corrección de Holm (p_Holm = 1,0 en los tres). El detalle de calibración de estas mismas probabilidades (ECE y *Brier*) está en [`calibracion-melanoma-ham.md`](./calibracion-melanoma-ham.md).

### B.13.3 Endpoint 3 — malignidad sobre DermapixelAI 1.0 (validación cruzada *out-of-fold*, N=1062, 58 malignos)

Protocolo distinto del de los *endpoints* train→test: el *split* de test canónico es de tamaño limitado para inferencia binaria (N = 36, solo 3 positivos), por lo que se estima por validación cruzada *out-of-fold* de cinco pliegues sobre el dataset completo, con DeLong pareado sobre las predicciones agrupadas y corrección de Holm.

| Modelo | AUROC (OOF) | IC95 % (DeLong) |
|---|---:|---|
| DermLIP v2 | **0,8946** | [0,8472, 0,9421] |
| PanDerm Large | 0,8632 | [0,8075, 0,9189] |
| PanDerm Base | 0,8630 | [0,8097, 0,9163] |
| CLIP-L | 0,8536 | [0,7926, 0,9146] |
| DINOv2 | 0,8469 | [0,7980, 0,8958] |

DermLIP v2 obtiene el mejor valor puntual, pero con 58 positivos ningún par alcanza significación tras Holm. La conclusión sólida no es coronar un modelo, sino constatar que toda la familia de codificadores dermatológicos supera de forma consistente a los generalistas.

## B.14 Clasificador unificado jerárquico M7 (DermapixelAI 1.0)

M7 es PanDerm Large *fine-tuned* sobre las 43 clases L3 de la ontología unificada multi-dataset (*merged43*), entrenado con *weighted sampling* sobre los siete datasets viables del *mapping* ontológico. La inferencia aplica *Test-Time Augmentation* (TTA) con cinco augmentaciones (original, simetría horizontal, simetría vertical, rotación 90° y 270°) promediando probabilidades *softmax*; la salida se proyecta jerárquicamente a L1 y L2 garantizando consistencia *top-down*. Cifras sobre el conjunto de test fijo de DermapixelAI 1.0.

| Nivel | Métrica | Valor |
|---|---|---:|
| L1 | Accuracy | 0,947 |
| L2 | Accuracy | 0,819 |
| L3 | Accuracy | 0,797 |
| L3 | BAcc | 0,818 |

La TTA aporta +4,72 pp de BAcc L3 respecto a la inferencia simple; la latencia *warm* es de ~80 ms. Este módulo es la base del *tab* `UNIF` del frontend del prototipo, con vista jerárquica L1/L2/L3 y *badge* de melanoma ([`../prototype/README.md`](../prototype/README.md)).
