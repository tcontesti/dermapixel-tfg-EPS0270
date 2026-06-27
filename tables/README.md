# tables/ — Tablas detalladas de resultados

Tablas extendidas que desagregan, por dataset y por modelo, las cifras sintetizadas en el cuerpo de la memoria (capítulo de Resultados). Todas las cifras coinciden de forma exacta con las del PDF [`MemoriaTFG.pdf`](../MemoriaTFG.pdf); no se introduce ningún resultado adicional.

## Contenido

| Documento | Contenido |
|---|---|
| [tablas-detalladas.md](tablas-detalladas.md) | Desgloses por clase, equidad por fototipo (Fitzpatrick17k), evaluación cruzada *light/dark*, segmentación fuera de distribución y prueba de DeLong en tres *endpoints* |
| [bootstrap-ci.md](bootstrap-ci.md) | Intervalos de confianza al 95 % por *bootstrap* (B = 1 000, remuestreo estratificado por clase, percentil 2,5/97,5, semilla 42), celda a celda, de las métricas *headline* |
| [calibracion-melanoma-ham.md](calibracion-melanoma-ham.md) | Calibración de los tres codificadores en el *endpoint* binario melanoma vs. resto sobre el test de HAM10000 (ECE de 15 *bins* y *Brier score*) |

## Artefactos reproducibles

- [`bootstrap-ci/`](bootstrap-ci/) — script `bootstrap_ci.py`, resultados consolidados `bootstrap_ci_all.csv` y `sanity_check_report.md`.
- [`calibration-ham/`](calibration-ham/) — script `ece_brier_ham.py` y resultados `calibration_ham_melanoma_results.json`.

## Nota sobre DeLong

La prueba de DeLong se reporta en tres *endpoints*. La potencia estadística suficiente solo se alcanza en **Fitzpatrick17k**; en los demás *endpoints* la falta de potencia impide concluir, lo que **no** equivale a ausencia de efecto. El detalle está en [tablas-detalladas.md](tablas-detalladas.md).
