# Ablación R0-AUG — ¿mejora el augmentation de train la head L2 de Dermapixel R0?

> Material complementario del TFG EPS0270. Ablación complementaria con un **resultado negativo limpio**; no altera ninguna cifra del cuerpo de la memoria y refuerza la conclusión de que la palanca de la cola larga son datos reales nuevos.

## Pregunta

¿Aplicar *data augmentation* sobre las imágenes de *train* mejora la head L2 supervisada de Dermapixel R0 (la head `mlp_512` sobre el encoder **PanDerm-Large congelado**)? **Respuesta: no (FLAT).**

## Método

Se aumentan las imágenes de *train*, se re-extraen sus *embeddings* del encoder PanDerm-Large **congelado** y se reentrena la misma head `mlp_512`. La señal principal es una **validación cruzada de 5 *folds* OOF (*out-of-fold*) *case-aware*** sobre las 1.062 imágenes de DermapixelAI 1.0 (39 clases L2), en comparación **OOF-vs-OOF** (baseline sin aumento y variantes comparten *folds* y espacio de etiquetas, *apples-to-apples*). El aumento se aplica **solo al *train* de cada *fold*** y se re-extrae por *fold*; el *held-out* se evalúa siempre con su *embedding* original (sin aumentar). Se prueban N = 3, 5 y 10 copias aumentadas por imagen. **Paso 0:** la head reentrenada sobre los *embeddings* de *train* sin aumentar reproduce el *baseline* publicado (`dermapixel_v1_abe`) **exactamente** (Δ = 0,0000 en todas las métricas).

El *augmentation* es **derm-safe** (el color es diagnóstico): *flip* horizontal/vertical, rotación ±15°, *random resized crop* (224, escala 0,8–1,0) y *jitter* suave de brillo/contraste (0,1) **sin tocar tono ni saturación**, con la misma normalización ImageNet que el extractor original.

## Resultados (señal principal — 5-fold OOF *case-aware*, 1.062 imgs, 39 clases L2)

| Variante | Acc@1 | Acc@3 | BAcc | AUROC | W-F1 |
|---|---:|---:|---:|---:|---:|
| **baseline (sin aug)** | 0,2232 | 0,4115 | 0,1459 | 0,6777 | 0,2137 |
| N = 3 | 0,2175 | 0,4077 | 0,1401 | 0,6739 | 0,2051 |
| N = 5 | 0,2232 | 0,4011 | 0,1469 | 0,6766 | 0,2112 |
| N = 10 | 0,2222 | 0,4115 | 0,1443 | 0,6748 | 0,2096 |

**Δ frente al baseline** (positivo = mejora):

| Variante | Acc@1 | Acc@3 | BAcc | AUROC | W-F1 |
|---|---:|---:|---:|---:|---:|
| N = 3 | −0,0057 | −0,0038 | −0,0058 | −0,0038 | −0,0086 |
| N = 5 | +0,0000 | −0,0104 | +0,0010 | −0,0011 | −0,0025 |
| N = 10 | −0,0010 | +0,0000 | −0,0016 | −0,0029 | −0,0041 |

El **delta absoluto máximo** en cualquier métrica y N es **±0,0104** (ruido). N = 5 es neutro; N = 3 y N = 10 son ligeramente negativos. No se ajustó ningún hiperparámetro para perseguir una cifra.

## Referencia secundaria — test-36 fijo (ruidoso, no comparable)

Sobre el *split* de test fijo (36 imágenes, 38 clases; protocolo distinto al OOF) aparecen subidas aparentes (Acc@1 0,2500 → 0,3056 en N = 5; BAcc 0,2647 → 0,3529 en N = 3) **no monótonas** (N = 10 vuelve al *baseline*) y dentro del ruido de 36 muestras. **No constituyen señal** y no se usan como conclusión.

## Comprobación de fuga de información (todas pasan)

- *Folds* *case-aware*: `case_overlap_train_held = 0` en los 5 *folds* (*assert* en código).
- El aumento se genera **solo** para imágenes de *train* de cada *fold* (`aug_cases ⊆ train_cases`, `aug ∩ held = ∅`, *assert*).
- *Held-out* y test-36 se evalúan con *embeddings* **originales** (sin aumentar).
- test-36: `train_test_case_overlap = 0`.

Tamaños de *train* por *fold* (antes → aumentado para N = 3/5/10), p. ej. *fold* 1: 851 → 3.404 / 5.106 / 9.361.

## Conclusión honesta

**FLAT / sin mejora.** El *augmentation* de imagen sobre un encoder **congelado** no añade señal a la cola larga (clases con 1–2 imágenes): re-muestrear transformaciones de las mismas imágenes no crea información nueva que el encoder no codifique ya. Es coherente con el hallazgo de que **escalar con datos genéricos externos también resulta FLAT** ([rama contrastiva de Dermapixel R0](ablacion-dermlip-en-y-r0-contrastiva.md)). **Refuerza la tesis del trabajo: la palanca de la cola larga son datos reales nuevos y validados, no reprocesar lo existente.** Recomendación: no adoptar *augmentation* de *train* para la head L2 de R0.

> Código: [`code/dermapixel_r0/dermapixel_aug_oof.py`](../code/dermapixel_r0/dermapixel_aug_oof.py).
