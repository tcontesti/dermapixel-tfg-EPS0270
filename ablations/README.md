# ablations/ — Ablaciones complementarias

Ablaciones y análisis complementarios sobre el dataset DermapixelAI 1.0, el pipeline de construcción y caracterización del dataset, la adaptación del modelo fundacional inglés con traducción ES–EN y las iteraciones de la rama contrastiva en castellano (Dermapixel R0 / SpanDerm).

## Contenido

| Documento | Contenido |
|---|---|
| [pipeline-dataset.md](pipeline-dataset.md) | Pipeline de construcción y caracterización de DermapixelAI 1.0: hash MD5, deduplicación, mapeo ontológico con revisión experta, particionado *case-aware* y análisis exploratorio |
| [ablacion-dermlip-en-y-r0-contrastiva.md](ablacion-dermlip-en-y-r0-contrastiva.md) | Ablación del modelo fundacional inglés (DermLIP) con traducción ES–EN, y las cinco iteraciones de la rama contrastiva de Dermapixel R0 (SpanDerm-CLIP) |
| [ablaciones-complementarias.md](ablaciones-complementarias.md) | Heads de clasificación, función de pérdida, TTA, codificadores generalistas, *k*-NN y *zero-shot* jerárquico |
| [ablacion-r0-augmentation.md](ablacion-r0-augmentation.md) | Ablación R0-AUG: el *augmentation* de *train* sobre el encoder congelado **no mejora** la head L2 de Dermapixel R0 (5-fold OOF *case-aware*, FLAT ±0,01) — la palanca de la cola larga son datos reales nuevos, no reprocesar |

## Precisión importante (rama contrastiva)

La rama contrastiva texto→imagen de Dermapixel R0 (SpanDerm-CLIP) demuestra la **viabilidad técnica** del espacio dual texto-imagen, pero **no** una capacidad de recuperación útil en régimen real: con consulta corta y conjunto de test no visto, la recuperación cae a un nivel cercano al azar. La partición es limpia (verificada disjunta por caso e imagen), lo que descarta la contaminación como causa. **No está desplegada.** La búsqueda visual en producción la cubre M4-bis (imagen→imagen, PanDerm Large + FAISS). Ver el [prototipo](../prototype/README.md).
