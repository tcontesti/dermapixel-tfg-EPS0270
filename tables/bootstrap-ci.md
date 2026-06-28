# Intervalos de confianza por bootstrap (B=1000)

<a id="sec:repo-bootstrap-ci"></a>

> Material complementario del TFG EPS0270. Amplía el Anexo B (tablas detalladas de resultados) con los intervalos de confianza al 95 % por *bootstrap*, celda a celda, de las cuatro tablas *headline* del cuerpo de la memoria que se reportan en él como estimaciones puntuales: el sondeo lineal de PanDerm sobre los conjuntos externos (`tab:lp-panderm`), la comparativa de paradigmas de razonamiento clínico (`tab:llm-comparativa`), el desglose de equidad por fototipo (`tab:equidad-fototipo`) y el *benchmark* frente a líneas base convolucionales (`tab:benchmark`). El cuerpo de la memoria enlaza a esta sección mediante el ancla `sec:repo-bootstrap-ci` desde la sección de pruebas estadísticas (§3.7) y desde el anexo-puntero de material complementario.

## Nota de método

Bootstrap **estratificado por clase**, **B = 1000** remuestreos, **IC95 % por percentil** (cuantiles 2,5 / 97,5), **semilla 42**. Cada remuestreo extrae, con reemplazo y preservando el tamaño por clase, índices dentro de cada clase verdadera del conjunto de evaluación; las métricas se recalculan sobre cada remuestreo y el intervalo se toma como `[percentil 2,5, percentil 97,5]` de la distribución *bootstrap*.

Dos mecanismos según la disponibilidad de la salida por muestra:

- **Reajuste de clasificador lineal sobre *embeddings* congelados** (donde no había probabilidad por muestra guardada): se cargan los *embeddings* ya extraídos por el codificador correspondiente y se reajusta una regresión logística con la configuración fija `LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000, random_state=42)` sobre el *split* de entrenamiento; se obtiene `predict_proba` sobre el *split* de test y se bootstrapea sobre esas predicciones. El reajuste de la head lineal es determinista y reproduce el valor puntual publicado (no constituye reentrenamiento del codificador).
- **Bootstrap directo sobre las predicciones guardadas** (donde sí existían `y_true` / `y_pred` / probabilidad por muestra): se remuestrea directamente sobre el CSV de predicciones por imagen.

**Verificación (*sanity check*).** Para cada celda se contrasta el valor puntual publicado en el cuerpo frente a la media *bootstrap*: el criterio de aceptación exige `|media_boot − publicado| ≤ 0,005` y que el IC contenga el valor publicado (con tolerancia de redondeo a 3 decimales). Las **177 celdas con cifra publicada cumplen el criterio** (cero ajustes, cero paradas); las **8 celdas en *gap*** (ver abajo) no se admiten y no se les fabrica intervalo. El valor mostrado en cada celda es el publicado en el cuerpo; el intervalo `[lo, hi]` es el IC95 % *bootstrap* a su alrededor.

## Nota de *caveat* (declarada abiertamente)

**(a) N mixto en el DDI.** En `tab:benchmark` las filas de líneas base convolucionales (CNN) se publicaron sobre **N = 220** —el *script* de extracción CNN fusiona el *split* de validación con el de test— mientras que PanDerm Large y SigLIP se publicaron sobre **N = 137** (solo test). En `tab:llm-comparativa`, **Gemini 2.5 Pro** y **Gemini 2.5 Flash** se evaluaron sobre el DDI con **N = 220**, mientras que el resto de filas LLM-DDI usan **N = 137** (el *caption* del cuerpo declara N = 137). Cada celda se ha bootstrapeado **sobre su propio N**, el que reproduce su valor publicado; el N efectivo de cada celda se indica en la columna `N` de las tablas siguientes. *No se ha recalculado ni homogeneizado ninguna cifra del cuerpo: esta nota documenta la inconsistencia de N existente, no la corrige.*

**(b) Convención de AUROC por fila.** La convención de promediado de AUROC multiclase no es homogénea entre filas y se ha verificado por celda contra los `metrics_results.csv` guardados (reproduce el valor publicado al 4.º decimal):

- líneas base **CNN** → **OvR-macro**;
- **SigLIP** → **OvR-weighted**;
- **PanDerm / sondeo lineal** → **OvO-macro** (con la única excepción de la celda PAD-UFES-20 AUROC del *benchmark*, que reproduce con **OvR-macro**);
- celdas binarias (DDI, Derm7pt, HIBA, MSKCC) → AUROC binaria directa.

## Celdas en *gap* (8)

No se publican IC para las siguientes celdas, por las razones indicadas; **no se inventa ningún intervalo** para ellas:

1. **DINOv2 ViT-L/14 — HAM10000 (Acc y AUROC): 2 celdas.** Los valores publicados (0,744 / 0,847) no son específicos de HAM10000 sino el **promedio del sondeo lineal sobre 10 datasets** (marcado con asterisco en el cuerpo); un IC por-dataset no procede.
2. **Gemini 2.5 Pro y Gemini 2.5 Flash — BAcc en HAM10000, PAD-UFES-20 y DDI: 6 celdas.** El cuerpo omite la BAcc de estas filas (`---`). El IC *bootstrap* de la BAcc es computable a partir de las predicciones guardadas, pero, al **no figurar la cifra puntual en el cuerpo**, se marca como **«inf.»** (informativo) y se aporta solo a título orientativo.

---

## tab:lp-panderm — Sondeo lineal PanDerm Base / Large sobre 10 datasets externos

Mecanismo: *bootstrap* directo sobre las predicciones por muestra guardadas. AUROC multiclase con convención **OvO-macro**; DDI, Derm7pt, HIBA y MSKCC son binarias (AUROC directa). Formato de celda: `valor [lo, hi]` (coma decimal); el `valor` es el publicado en el cuerpo.

### PanDerm Base

| Dataset | N | Acc | BAcc | AUROC | W-F1 |
|---|---:|---|---|---|---|
| HAM10000 | 1232 | 0,853 [0,840, 0,867] | 0,381 [0,344, 0,420] | 0,892 [0,854, 0,925] | 0,829 [0,813, 0,845] |
| BCN20000 | 1242 | 0,659 [0,639, 0,680] | 0,292 [0,277, 0,308] | 0,855 [0,838, 0,872] | 0,615 [0,593, 0,636] |
| PAD-UFES-20 | 461 | 0,725 [0,690, 0,759] | 0,510 [0,461, 0,569] | 0,913 [0,894, 0,931] | 0,692 [0,657, 0,725] |
| Dermnet | 4002 | 0,450 [0,436, 0,464] | 0,381 [0,367, 0,397] | 0,888 [0,882, 0,894] | 0,432 [0,418, 0,446] |
| WSI patches | 12354 | 0,781 [0,775, 0,788] | 0,640 [0,631, 0,649] | 0,976 [0,974, 0,977] | 0,774 [0,767, 0,780] |
| DDI | 137 | 0,847 [0,796, 0,891] | 0,714 [0,615, 0,804] | 0,827 [0,731, 0,916] | 0,836 [0,774, 0,888] |
| Derm7pt clínico | 168 | 0,762 [0,714, 0,810] | 0,675 [0,613, 0,741] | 0,808 [0,738, 0,877] | 0,736 [0,674, 0,796] |
| Derm7pt dermoscópico | 225 | 0,818 [0,769, 0,862] | 0,749 [0,682, 0,814] | 0,841 [0,781, 0,901] | 0,811 [0,758, 0,860] |
| HIBA | 334 | 0,883 [0,865, 0,901] | 0,595 [0,541, 0,660] | 0,894 [0,855, 0,929] | 0,853 [0,824, 0,882] |
| MSKCC | 1664 | 0,721 [0,700, 0,742] | 0,654 [0,630, 0,679] | 0,746 [0,720, 0,772] | 0,720 [0,700, 0,741] |

### PanDerm Large

| Dataset | N | Acc | BAcc | AUROC | W-F1 |
|---|---:|---|---|---|---|
| HAM10000 | 1232 | 0,888 [0,874, 0,902] | 0,575 [0,505, 0,649] | 0,954 [0,942, 0,965] | 0,880 [0,863, 0,896] |
| BCN20000 | 1242 | 0,702 [0,680, 0,724] | 0,382 [0,349, 0,420] | 0,903 [0,890, 0,916] | 0,676 [0,652, 0,698] |
| PAD-UFES-20 | 461 | 0,772 [0,735, 0,805] | 0,642 [0,572, 0,709] | 0,949 [0,935, 0,962] | 0,760 [0,721, 0,793] |
| Dermnet | 4002 | 0,550 [0,535, 0,563] | 0,495 [0,479, 0,513] | 0,931 [0,927, 0,936] | 0,540 [0,525, 0,553] |
| WSI patches | 12354 | 0,868 [0,863, 0,874] | 0,792 [0,774, 0,817] | 0,991 [0,990, 0,992] | 0,871 [0,865, 0,876] |
| DDI | 137 | 0,847 [0,788, 0,898] | 0,764 [0,667, 0,852] | 0,860 [0,786, 0,928] | 0,846 [0,784, 0,898] |
| Derm7pt clínico | 168 | 0,798 [0,744, 0,851] | 0,732 [0,661, 0,798] | 0,869 [0,812, 0,922] | 0,784 [0,721, 0,842] |
| Derm7pt dermoscópico | 225 | 0,836 [0,791, 0,880] | 0,766 [0,696, 0,827] | 0,858 [0,794, 0,915] | 0,829 [0,777, 0,877] |
| HIBA | 334 | 0,904 [0,880, 0,931] | 0,701 [0,632, 0,782] | 0,942 [0,913, 0,967] | 0,892 [0,863, 0,924] |
| MSKCC | 1664 | 0,746 [0,727, 0,765] | 0,644 [0,620, 0,669] | 0,751 [0,725, 0,776] | 0,732 [0,712, 0,752] |

---

## tab:benchmark — Líneas base convolucionales vs. PanDerm / SigLIP

Métricas publicadas en el cuerpo: HAM10000 (Acc, AUROC), PAD-UFES-20 (Acc, AUROC), DDI (AUROC binaria). Convención AUROC por fila (ver *caveat* b) y N efectivo del DDI por celda (ver *caveat* a). Formato `valor [lo, hi]`.

| Modelo | Conv. AUROC | N (DDI) | HAM Acc | HAM AUROC | PAD Acc | PAD AUROC | DDI AUROC |
|---|---|---:|---|---|---|---|---|
| ConvNeXt-Large | OvR-macro | 220 | 0,883 [0,867, 0,899] | 0,967 [0,957, 0,976] | 0,668 [0,627, 0,709] | 0,894 [0,874, 0,914] | 0,774 [0,677, 0,862] |
| EfficientNetV2-Large | OvR-macro | 220 | 0,860 [0,842, 0,877] | 0,950 [0,938, 0,961] | 0,690 [0,649, 0,729] | 0,895 [0,872, 0,914] | 0,766 [0,672, 0,852] |
| DINOv2 ViT-L/14 | — | — | *gap* (promedio LP, no específico HAM) | *gap* (promedio LP, no específico HAM) | — | — | — |
| PanDerm Large | OvO-macro¹ | 137 | 0,888 [0,874, 0,902] | 0,954 [0,942, 0,965] | 0,772 [0,735, 0,805] | 0,949 [0,933, 0,961] | 0,860 [0,786, 0,928] |
| SigLIP-Large SO400M | OvR-weighted | 137 | 0,900 [0,881, 0,912] | 0,971 [0,962, 0,980] | 0,718 [0,679, 0,757] | 0,903 [0,881, 0,923] | 0,793 [0,695, 0,885] |

> ¹ HAM AUROC de PanDerm reproduce con OvO-macro; la celda PAD-UFES-20 AUROC reproduce con OvR-macro (ver *caveat* b). Las dos celdas de DINOv2 ViT-L/14 sobre HAM10000 figuran como *gap* (sus 0,744 / 0,847 son el promedio del sondeo lineal sobre 10 datasets, no valores específicos de HAM10000).

---

## tab:equidad-fototipo — Equidad por fototipo (Fitzpatrick17k, 3 clases)

BAcc por fototipo Fitzpatrick (FST I–VI) y *gap* I–VI (= BAcc FST I − BAcc FST VI). Mecanismo: reajuste de regresión logística sobre los *embeddings* Fitzpatrick17k cacheados (3 clases, sin `StandardScaler`, igual que el *pipeline* original), predicciones por muestra del *split* de test, *bootstrap* de la BAcc dentro de cada subgrupo FST y *bootstrap* directo del *gap*. Formato `valor [lo, hi]`. N por subgrupo: I=299, II=470, III=325, IV=261, V=169, VI=73.

| Modelo | FST I | FST II | FST III | FST IV | FST V | FST VI | Gap I–VI |
|---|---|---|---|---|---|---|---|
| PanDerm Large | 0,723 [0,652, 0,788] | 0,708 [0,654, 0,764] | 0,743 [0,676, 0,806] | 0,669 [0,599, 0,745] | 0,568 [0,479, 0,662] | 0,506 [0,333, 0,684] | 0,217 [0,015, 0,387] |
| PanDerm Base | 0,653 [0,584, 0,718] | 0,693 [0,644, 0,747] | 0,669 [0,608, 0,740] | 0,630 [0,562, 0,706] | 0,585 [0,492, 0,680] | 0,384 [0,301, 0,517] | 0,269 [0,116, 0,389] |
| DermLIP v2 | 0,716 [0,649, 0,782] | 0,730 [0,677, 0,782] | 0,765 [0,695, 0,835] | 0,710 [0,642, 0,785] | 0,693 [0,578, 0,801] | 0,434 [0,301, 0,579] | 0,281 [0,109, 0,430] |
| DINOv2 ViT-L/14 | 0,673 [0,604, 0,736] | 0,671 [0,616, 0,726] | 0,679 [0,612, 0,745] | 0,654 [0,578, 0,735] | 0,553 [0,447, 0,666] | 0,368 [0,280, 0,501] | 0,306 [0,136, 0,426] |
| BiomedCLIP | 0,569 [0,506, 0,638] | 0,602 [0,554, 0,653] | 0,606 [0,544, 0,669] | 0,566 [0,496, 0,644] | 0,590 [0,479, 0,699] | 0,445 [0,307, 0,584] | 0,124 [−0,052, 0,280] |

El IC del *gap* I–VI se calcula bootstrapeando directamente la diferencia de BAcc entre los subgrupos FST I y FST VI (no como resta de los dos IC marginales). **Obsérvese que los IC del *gap* son anchos** por el reducido N de FST VI (73): el de **BiomedCLIP cruza el cero** (`0,124 [−0,052, 0,280]`), lo que respalda numéricamente el *caveat* del cuerpo sobre la fragilidad del *gap* en fototipos oscuros.

---

## tab:llm-comparativa — Comparativa de paradigmas de razonamiento clínico

Los modelos de lenguaje producen clase, no probabilidad calibrada por muestra → solo se reportan **Acc** y **BAcc** (la AUROC no es computable). Mecanismo: *bootstrap* directo sobre el CSV de predicciones por muestra. N efectivo del DDI por fila (ver *caveat* a). Formato `valor [lo, hi]`. Las BAcc de Gemini no figuran en el cuerpo: se aportan marcadas como **«inf.»** (informativo).

| Fila | N (DDI) | HAM Acc | HAM BAcc | PAD Acc | PAD BAcc | DDI Acc | DDI BAcc |
|---|---:|---|---|---|---|---|---|
| MedGemma 27B + LoRA | 137 | 0,802 [0,791, 0,813] | 0,270 [0,235, 0,305] | 0,430 [0,397, 0,462] | 0,347 [0,294, 0,409] | 0,657 [0,577, 0,730] | 0,618 [0,516, 0,714] |
| MedGemma 27B | 137 | 0,665 [0,644, 0,685] | 0,243 [0,216, 0,270] | 0,447 [0,423, 0,469] | 0,342 [0,292, 0,400] | 0,474 [0,394, 0,547] | 0,566 [0,470, 0,649] |
| GPT-4o | 137 | 0,485 [0,458, 0,511] | 0,465 [0,398, 0,526] | 0,553 [0,512, 0,592] | 0,523 [0,453, 0,584] | 0,723 [0,657, 0,796] | 0,648 [0,555, 0,746] |
| GPT-4o-mini | 137 | 0,256 [0,233, 0,278] | 0,319 [0,276, 0,362] | 0,390 [0,343, 0,436] | 0,435 [0,360, 0,506] | 0,737 [0,672, 0,796] | 0,556 [0,474, 0,643] |
| Gemini 2.5 Pro | 220 | 0,403 [0,379, 0,429] | inf. 0,433 [0,384, 0,488] | 0,477 [0,436, 0,518] | inf. 0,474 [0,435, 0,514] | 0,441 [0,377, 0,500] | inf. 0,551 [0,477, 0,621] |
| Gemini 2.5 Flash | 220 | 0,380 [0,355, 0,406] | inf. 0,402 [0,359, 0,447] | 0,486 [0,443, 0,527] | inf. 0,470 [0,399, 0,537] | 0,555 [0,482, 0,614] | inf. 0,616 [0,532, 0,687] |
| BLIP-2 (Flan-T5-XL) | 137 | 0,015 [0,014, 0,016] | 0,145 [0,143, 0,149] | 0,017 [0,017, 0,017] | 0,167 [0,167, 0,167] | 0,796 [0,766, 0,825] | 0,542 [0,494, 0,599] |
| InstructBLIP (Flan-T5-XL) | 137 | 0,024 [0,017, 0,033] | 0,123 [0,077, 0,178] | 0,171 [0,139, 0,200] | 0,224 [0,155, 0,297] | 0,788 [0,788, 0,788] | 0,500 [0,500, 0,500] |

> Las 6 celdas de BAcc de Gemini 2.5 Pro / Flash van marcadas **«inf.»** (el cuerpo las omite con `---`; el IC se aporta solo a título orientativo). Las filas de Gemini sobre el DDI usan N = 220 (test + val), frente a N = 137 del resto de filas LLM-DDI. Los IC de ancho nulo (p. ej. InstructBLIP-DDI `[0,788, 0,788]`, BLIP-2-PAD) corresponden a clasificadores degenerados que predicen una única clase, por lo que el remuestreo estratificado no altera la métrica.

---

## Procedencia y reproducción

El cómputo de la Fase 1 se ejecuta con el *script* reutilizable [`bootstrap-ci/bootstrap_ci.py`](bootstrap-ci/bootstrap_ci.py), que carga los *embeddings* congelados o las predicciones por muestra, reajusta el clasificador lineal con la configuración fija indicada y emite por celda el valor puntual publicado, la media *bootstrap* y el IC95 %, junto con el informe de *sanity check* ([`bootstrap-ci/sanity_check_report.md`](bootstrap-ci/sanity_check_report.md), 177 OK / 0 paradas / 8 *gap* sobre 185 celdas) y el volcado completo ([`bootstrap-ci/bootstrap_ci_all.csv`](bootstrap-ci/bootstrap_ci_all.csv)). Los pares `[lo, hi]` de las tablas anteriores provienen de ese volcado (columnas `ci_lo`, `ci_hi`), redondeados a 3 decimales para casar con la convención de cifras del cuerpo.
