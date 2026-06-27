# Calibración en el *endpoint* de melanoma de HAM10000 (ECE + Brier)

<a id="sec:repo-calibracion-ham"></a>

> Material complementario del TFG EPS0270. Amplía el cuerpo de la memoria (§4.9, *endpoint* 2 del test de DeLong, `tab:delong-ham`) con el detalle por *bin* de la calibración de los tres codificadores en el *endpoint* binario **melanoma frente al resto** sobre el test de HAM10000 ($N = 1\,232$, $70$ melanomas). El cuerpo reporta el error de calibración esperado (ECE) y el *Brier score* de forma compacta; aquí se aportan las curvas de fiabilidad (tablas por-*bin*) que los sustentan. El cuerpo enlaza a esta sección mediante el ancla `sec:repo-calibracion-ham`.

## Nota de método

Las probabilidades por muestra son **exactamente las del test de DeLong de HAM10000** (`tab:delong-ham`): no se reentrena ni se re-infiere. Para cada modelo se cargan los *embeddings* congelados ya extraídos (PanDerm Large y DermLIP v2 cacheados; SigLIP-Large desde su `.pt`) y se reajusta la **misma** cabeza lineal con configuración fija `LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000, random_state=42)` sobre el *split* de entrenamiento ($N = 8\,207$); las probabilidades de la clase positiva sobre el *split* de test ($N = 1\,232$) son las que se evalúan. El reajuste de la cabeza lineal es determinista y reproduce el valor de AUROC publicado (no constituye reentrenamiento del codificador).

- **ECE** (*Expected Calibration Error*): **15 *bins* de igual anchura** sobre la probabilidad de la clase positiva (cortes en $k/15$, $k = 0,\dots,15$). Por *bin* se calcula la confianza media (probabilidad media) y la exactitud observada (fracción de positivos reales); el ECE es la media ponderada por ocupación del valor absoluto de su diferencia, $\mathrm{ECE} = \sum_b \frac{n_b}{N}\,\lvert \mathrm{acc}_b - \mathrm{conf}_b\rvert$.
- **Brier score**: error cuadrático medio entre la probabilidad de la clase positiva y la etiqueta binaria, $\frac{1}{N}\sum_i (p_i - y_i)^2$.

**Verificación (*sanity check*).** Antes de admitir cualquier cifra de calibración se recomputa la **AUROC binaria** desde esas mismas probabilidades y se contrasta contra el valor publicado en `tab:delong-ham` (tolerancia a 3 decimales). Las **tres celdas reproducen** el valor publicado; el ECE y el *Brier* solo se reportan tras pasar esta verificación (criterio de parada: cualquier discrepancia aborta el cómputo sin escribir resultados).

## Resumen

| Modelo | AUROC (publicada / recomputada) | *sanity* | ECE (15 *bins*) | Brier |
|---|---|:---:|---:|---:|
| PanDerm Large | 0,9523 / 0,952336 | OK | 0,0174 | 0,0291 |
| SigLIP-Large | 0,9495 / 0,949533 | OK | 0,0174 | 0,0297 |
| DermLIP v2 | 0,9463 / 0,946349 | OK | 0,0178 | 0,0315 |

La discriminación es fuerte (AUROC $\sim 0{,}95$) **y** la calibración sobre este *endpoint* es razonable (ECE $< 0{,}02$, *Brier* $< 0{,}032$ en los tres modelos), sin necesidad de recalibración posterior. El grueso de la masa de probabilidad se concentra en el primer *bin* ($p \le 0{,}067$): más del $80\,\%$ de las muestras de test son negativos claros (la prevalencia de melanoma es $70/1232 \approx 5{,}7\,\%$), y ahí la confianza casa con la exactitud casi perfectamente (*gap* $\approx 0{,}001$). Los *bins* intermedios, escasamente poblados (de $5$ a $30$ muestras), presentan *gaps* mayores pero con peso despreciable en el ECE agregado. El **análisis sistemático de calibración entre tareas** queda como trabajo futuro (capítulo de limitaciones, elementos no abordados).

---

## Curvas de fiabilidad (tablas por-*bin*)

Formato: `n` = muestras en el *bin*; `conf` = confianza media (probabilidad media); `acc` = exactitud observada (fracción de positivos reales); `gap` = $\lvert \mathrm{acc} - \mathrm{conf}\rvert$. Los *bins* vacíos se omiten. Coma decimal.

### PanDerm Large

| *Bin* | n | conf | acc | gap |
|---|---:|---:|---:|---:|
| (0,00, 0,07] | 1017 | 0,0077 | 0,0088 | 0,0011 |
| (0,07, 0,13] | 64 | 0,0959 | 0,0625 | 0,0334 |
| (0,13, 0,20] | 31 | 0,1605 | 0,1290 | 0,0314 |
| (0,20, 0,27] | 28 | 0,2355 | 0,1071 | 0,1283 |
| (0,27, 0,33] | 14 | 0,3038 | 0,1429 | 0,1609 |
| (0,33, 0,40] | 8 | 0,3708 | 0,5000 | 0,1292 |
| (0,40, 0,47] | 7 | 0,4341 | 0,5714 | 0,1373 |
| (0,47, 0,53] | 9 | 0,5000 | 0,3333 | 0,1666 |
| (0,53, 0,60] | 6 | 0,5471 | 0,1667 | 0,3804 |
| (0,60, 0,67] | 11 | 0,6382 | 0,3636 | 0,2746 |
| (0,67, 0,73] | 6 | 0,7053 | 0,6667 | 0,0386 |
| (0,73, 0,80] | 10 | 0,7667 | 0,7000 | 0,0667 |
| (0,80, 0,87] | 5 | 0,8540 | 1,0000 | 0,1460 |
| (0,87, 0,93] | 5 | 0,8967 | 1,0000 | 0,1033 |
| (0,93, 1,00] | 11 | 0,9664 | 1,0000 | 0,0336 |

### SigLIP-Large

| *Bin* | n | conf | acc | gap |
|---|---:|---:|---:|---:|
| (0,00, 0,07] | 1004 | 0,0080 | 0,0070 | 0,0010 |
| (0,07, 0,13] | 71 | 0,0933 | 0,0704 | 0,0229 |
| (0,13, 0,20] | 36 | 0,1693 | 0,0833 | 0,0860 |
| (0,20, 0,27] | 20 | 0,2328 | 0,0500 | 0,1828 |
| (0,27, 0,33] | 12 | 0,2977 | 0,3333 | 0,0356 |
| (0,33, 0,40] | 14 | 0,3692 | 0,2857 | 0,0835 |
| (0,40, 0,47] | 9 | 0,4420 | 0,3333 | 0,1087 |
| (0,47, 0,53] | 12 | 0,4941 | 0,3333 | 0,1608 |
| (0,53, 0,60] | 2 | 0,5725 | 0,5000 | 0,0725 |
| (0,60, 0,67] | 4 | 0,6280 | 0,2500 | 0,3780 |
| (0,67, 0,73] | 9 | 0,7001 | 0,7778 | 0,0777 |
| (0,73, 0,80] | 9 | 0,7680 | 0,5556 | 0,2124 |
| (0,80, 0,87] | 3 | 0,8232 | 0,0000 | 0,8232 |
| (0,87, 0,93] | 6 | 0,9040 | 0,8333 | 0,0706 |
| (0,93, 1,00] | 21 | 0,9721 | 0,9524 | 0,0197 |

### DermLIP v2

| *Bin* | n | conf | acc | gap |
|---|---:|---:|---:|---:|
| (0,00, 0,07] | 1024 | 0,0081 | 0,0107 | 0,0026 |
| (0,07, 0,13] | 53 | 0,0990 | 0,0755 | 0,0236 |
| (0,13, 0,20] | 33 | 0,1597 | 0,0606 | 0,0991 |
| (0,20, 0,27] | 19 | 0,2333 | 0,2105 | 0,0228 |
| (0,27, 0,33] | 19 | 0,3042 | 0,2632 | 0,0410 |
| (0,33, 0,40] | 13 | 0,3727 | 0,1538 | 0,2188 |
| (0,40, 0,47] | 16 | 0,4300 | 0,3750 | 0,0550 |
| (0,47, 0,53] | 7 | 0,5070 | 0,1429 | 0,3642 |
| (0,53, 0,60] | 9 | 0,5561 | 0,4444 | 0,1116 |
| (0,60, 0,67] | 3 | 0,6363 | 1,0000 | 0,3637 |
| (0,67, 0,73] | 8 | 0,6995 | 0,2500 | 0,4495 |
| (0,73, 0,80] | 2 | 0,7716 | 1,0000 | 0,2284 |
| (0,80, 0,87] | 4 | 0,8456 | 0,7500 | 0,0956 |
| (0,87, 0,93] | 2 | 0,8981 | 1,0000 | 0,1019 |
| (0,93, 1,00] | 20 | 0,9769 | 0,9500 | 0,0269 |

> Los *gaps* grandes de los *bins* superiores (p. ej. SigLIP-Large en `(0,80, 0,87]` con `acc = 0` sobre $n = 3$) corresponden a celdas con muy pocas muestras y, por su escasa ocupación, apenas contribuyen al ECE agregado. El comportamiento agregado lo domina el primer *bin*, que concentra la inmensa mayoría de los negativos.

---

## Procedencia y reproducción

El cómputo se ejecuta con el *script* [`calibration-ham/ece_brier_ham.py`](calibration-ham/ece_brier_ham.py), que carga los *embeddings* congelados / el `.pt` de SigLIP, reajusta la cabeza lineal con la configuración fija indicada, verifica la AUROC celda a celda contra `tab:delong-ham` (*gate* de parada) y emite el ECE, el *Brier* y la curva de fiabilidad por *bin* de cada modelo. El volcado completo está en [`calibration-ham/calibration_ham_melanoma_results.json`](calibration-ham/calibration_ham_melanoma_results.json). Las predicciones de partida son idénticas a las del test de DeLong de HAM10000 (`delong_ham.py`); el ECE y el *Brier* se calculan sobre CPU sin reentrenamiento.
