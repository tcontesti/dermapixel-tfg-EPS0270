# Anexo J — Ablaciones complementarias sobre DermapixelAI 1.0

> Material complementario del TFG EPS0270. Corresponde al antiguo Anexo J de la memoria. Recoge estudios complementarios sobre DermapixelAI 1.0 que confirman, sin desplazarla, la línea experimental principal: validación del etiquetado, tres variantes de clasificación sobre *embeddings* congelados (cabezas alternativas, función de pérdida y *Test-Time Augmentation*) y dos extensiones con codificadores generalistas y arquitecturas de inferencia alternativas. Ninguno modifica las conclusiones del cuerpo principal.

## J.1 Validación del procedimiento de etiquetado: subestudio rosa_verified

Comparación, por sondeo lineal, entre el subconjunto verificado caso a caso (`rosa_verified=True`, N = 919; train = 744, test = 32) y el dataset completo (`all_verified`, N = 1 062; train = 874, test = 36), sobre los tres niveles ontológicos. Entre corchetes, IC95 % por *bootstrap* estratificado.

| Modelo | Subconjunto | Nivel | Acc | BAcc | W-F1 |
|---|---|---|---|---|---|
| PanDerm Large | rosa_true_only | L1 | 0,719 [0,594–0,844] | 0,747 [0,646–0,865] | 0,700 [0,550–0,841] |
| PanDerm Large | all_verified | L1 | 0,750 [0,639–0,861] | 0,735 [0,637–0,835] | 0,722 [0,596–0,845] |
| PanDerm Large | rosa_true_only | L2 | 0,250 [0,188–0,344] | 0,250 [0,203–0,313] | 0,227 [0,150–0,304] |
| PanDerm Large | all_verified | L2 | 0,250 [0,167–0,333] | 0,265 [0,206–0,338] | 0,223 [0,139–0,291] |
| PanDerm Large | rosa_true_only | L3 | 0,167 [0,167–0,167] | 0,176 [0,176–0,176] | 0,125 [0,117–0,139] |
| PanDerm Large | all_verified | L3 | 0,179 [0,143–0,214] | 0,184 [0,158–0,211] | 0,143 [0,100–0,184] |
| PanDerm Base | rosa_true_only | L1 | 0,625 [0,469–0,781] | 0,523 [0,326–0,735] | 0,618 [0,451–0,773] |
| PanDerm Base | all_verified | L1 | 0,639 [0,472–0,778] | 0,570 [0,371–0,751] | 0,624 [0,457–0,760] |

Las diferencias son pequeñas, inconsistentes en dirección y compatibles con la variabilidad muestral: la imputación etiológica no degrada apreciablemente el rendimiento.

## J.2 Ranking L3 por prototipos coseno de Dermapixel R0

Ranking del nivel L3 por prototipos coseno (media de las representaciones de entrenamiento por clase), sobre las 250 clases efectivas del vocabulario de entrenamiento. Cifras: media ± desviación estándar sobre tres semillas. ×azar: factor de mejora respecto a la asignación aleatoria uniforme sobre 250 clases (1/250 = 0,004).

| Métrica | Media | Std | ×azar |
|---|---:|---:|---|
| Acc@1 | 0,167 | 0,000 | ×42 |
| Acc@3 | 0,259 | 0,013 | ×22 |
| Acc@5 | 0,296 | 0,026 | ×15 |
| BAcc | 0,148 | 0,000 | — |

## J.3 Cabezas alternativas sobre embeddings congelados

Sobre los mismos *embeddings* de PanDerm Large del split fijo (N_train = 874, N_test = 36 en L1/L2, N_test = 28 en L3) se evalúan tres cabezas adicionales: k-vecinos más cercanos con distancia coseno (k = 10), perceptrón multicapa con una capa oculta de 512 unidades y *dropout* 0,3 (MLP 512), y regresión logística con `class_weight='balanced'` (LogReg *balanced*). AUROC *one-vs-rest* por nivel. Mejor por nivel en **negrita**.

| Cabeza | L1 (4 cls) | L2 (38 cls) | L3 (224 cls) |
|---|---:|---:|---:|
| LP (referencia) | 0,796 | 0,793 | 0,813 |
| kNN k=10 | 0,752 | 0,698 | 0,694 |
| MLP 512 | **0,817** | **0,812** | **0,827** |
| LogReg *balanced* | 0,797 | 0,800 | 0,810 |

El MLP de 512 unidades alcanza la mejor AUROC sobre los tres niveles, con incrementos modestos pero consistentes sobre la cabeza lineal: +2,1 pp en L1 (0,796 → 0,817), +1,9 pp en L2 (0,793 → 0,812) y +1,4 pp en L3 (0,813 → 0,827). La cabeza k-NN queda por debajo de las paramétricas en los tres niveles, por insuficiente densidad local para N_train = 874 frente a un vocabulario de 224 clases L3. La elección de la regresión logística estándar como cabeza primaria se mantiene defendible: la mejora absoluta del MLP no compensa la pérdida de interpretabilidad lineal ni la mayor sensibilidad a la inicialización aleatoria.

## J.4 Función de pérdida: Cross-Entropy frente a Focal

La cola larga severa (77 % de las clases L3 con ≤ 5 ejemplos, 3,9 muestras por clase de media en *train*) motiva contrastar la entropía cruzada estándar con *Focal Loss* (Lin et al. 2017, factor (1−p_t)^γ). Sobre los *embeddings* congelados del split fijo se reentrena la cabeza lineal con *Focal Loss* bajo dos parámetros de focalización (γ ∈ {1,0, 2,0, 5,0}) y dos regímenes de ponderación (`class_weight = None` vs `balanced`), optimizada con LBFGS. Mejor por columna y nivel en **negrita**.

| Nivel | Pérdida | Acc@1 | BAcc | AUROC |
|---|---|---:|---:|---:|
| L1 (4 cls) | LP CE (referencia) | 0,750 | 0,735 | 0,796 |
| L1 (4 cls) | LogReg *balanced* | 0,694 | 0,703 | 0,797 |
| L1 (4 cls) | *Focal* γ = 2,0 | **0,778** | **0,768** | **0,800** |
| L2 (38 cls) | LP CE (referencia) | 0,250 | 0,265 | **0,793** |
| L2 (38 cls) | LogReg *balanced* | 0,250 | 0,265 | **0,800** |
| L2 (38 cls) | *Focal* γ = 2,0 | **0,278** | **0,324** | 0,778 |
| L2 (38 cls) | *Focal* γ = 5,0 | **0,278** | **0,324** | **0,793** |
| L3 (224 cls) | LP CE (referencia) | **0,143** | **0,132** | **0,813** |
| L3 (224 cls) | *Focal* γ = 2,0 | **0,143** | **0,132** | 0,771 |

*Focal Loss* con γ = 2,0 y sin re-ponderación supera marginalmente a la entropía cruzada estándar sobre L1 (+3,3 pp BAcc) y L2 (+5,9 pp BAcc), pero degrada la AUROC L2 en −1,5 pp (0,793 → 0,778). Añadir `class_weight = balanced` no aporta, porque el factor (1−p_t)^γ ya modula el peso por dificultad. Sobre L3 ningún ajuste de la pérdida mejora a la entropía cruzada (Acc@1 y BAcc idénticos en 0,143 y 0,132; AUROC −4,2 pp). La cola larga estructural de L3 no admite mitigación por la función de pérdida sobre el dataset actual.

## J.5 Aumento en tiempo de inferencia (TTA)

Se replica el protocolo de TTA sobre la cabeza lineal entrenada sobre los *embeddings* de PanDerm Large: para cada imagen de test se extraen los cinco *embeddings* de las cinco augmentaciones deterministas (original, simetría horizontal, simetría vertical, rotación 90°, rotación 270°, omitiendo 180°), cada uno se evalúa con la cabeza lineal y las cinco distribuciones se promedian antes del *argmax*. Mejor por par en **negrita**.

| Nivel | Modo | Acc@1 | Acc@3 | BAcc | AUROC |
|---|---|---:|---:|---:|---:|
| L1 (4 cls) | LP | **0,750** | 0,944 | **0,735** | **0,796** |
| L1 (4 cls) | LP + TTA | 0,694 | **1,000** | 0,703 | 0,779 |
| L2 (38 cls) | LP | **0,250** | 0,278 | **0,265** | **0,793** |
| L2 (38 cls) | LP + TTA | 0,222 | **0,500** | 0,250 | 0,787 |
| L3 (224 cls) | LP | **0,179** | 0,214 | **0,184** | **0,813** |
| L3 (224 cls) | LP + TTA | 0,179 | **0,250** | 0,184 | 0,807 |

El patrón es consistente: TTA degrada Acc@1, BAcc y AUROC en magnitudes pequeñas (−5,6, −2,8 y 0,0 pp Acc@1 sobre L1, L2 y L3; −3,2, −1,5 y 0,0 pp BAcc; −1,7, −0,6 y −0,6 pp AUROC) y mejora sistemáticamente Acc@3 (+5,6 pp en L1 hasta 1,000, +22,2 pp en L2 y +3,6 pp en L3). Coherente con M1 sobre HAM10000, donde TTA produce −2,6 pp BAcc y +4,3 pp *recall* de melanoma. El promediado suaviza la distribución y distribuye masa entre clases visualmente cercanas, lo que penaliza la precisión *top-1* y mejora *top-3*. PanDerm Base presenta la misma dirección de efecto. TTA es preferible en escenarios de cribado con presentación *top-k* (Acc@3 L2 = 0,500 frente a 0,278 sin TTA); el sondeo lineal sin TTA es preferible para decisiones automatizadas *top-1*.

## J.6 Codificadores generalistas no médicos: DINOv2 y CLIP-L

Dos codificadores fuera del dominio biomédico bajo sondeo lineal congelado, sobre el mismo split fijo. DINOv2 ViT-L (`vit_large_patch14_dinov2.lvd142m`, embedding 1 024-D, CLS), auto-supervisado visual generalista de Meta sobre LVD-142M; CLIP-L-336 (OpenAI, ViT-L/14 resolución 336), contrastivo texto-imagen sobre 400 millones de pares web, embedding proyectado de 768-D. Cabeza: regresión logística L-BFGS con C = 1,0 y max_iter = 5 000. Mejor por columna y nivel en **negrita**.

| Encoder | BAcc L1 | AUROC L1 | BAcc L2 | AUROC L2 | BAcc L3 | AUROC L3 |
|---|---:|---:|---:|---:|---:|---:|
| PanDerm Large | **0,735** | **0,796** | 0,265 | 0,793 | 0,184 | 0,813 |
| DINOv2 ViT-L | 0,537 | 0,650 | 0,250 | 0,718 | **0,211** | 0,794 |
| CLIP-L-336 | 0,556 | 0,658 | **0,279** | **0,826** | 0,184 | **0,827** |

CLIP-L-336 alcanza la mejor AUROC sobre L2 (0,826 frente a 0,793 de PanDerm Large, +3,3 pp) y L3 (0,827 frente a 0,813, +1,4 pp) a pesar de no incorporar material biomédico en su preentrenamiento. DINOv2 ViT-L gana la BAcc L3 (0,211 frente a 0,184, +2,7 pp) pero pierde sistemáticamente sobre L1 (−19,8 pp BAcc, −14,6 pp AUROC) y se mantiene por debajo sobre L2 (−1,5 pp BAcc, −7,5 pp AUROC). PanDerm Large conserva la primacía sobre L1. Ningún codificador domina simultáneamente todas las métricas y niveles. La clasificación *zero-shot* con CLIP-L (*prompts* en castellano e inglés) resulta catastrófica (AUROC L1 0,559 en castellano y 0,635 en inglés, AUROC L2 0,699 en castellano, AUROC L3 0,617 en castellano), confirmando que los codificadores generalistas requieren adaptación supervisada para operar sobre material clínico dermatológico.

## J.7 Retrieval k-NN con índice FAISS sobre embeddings PanDerm Large

Índice `IndexFlatIP` de FAISS sobre los 874 *embeddings* L2-normalizados de PanDerm Large del conjunto de *train*. Dos variantes (votación por mayoría y *softmax* ponderado por similitud) cruzadas con k ∈ {1, 5, 10, 20}. Mejor combinación por nivel y métrica frente a la referencia LP paramétrica. Mejor por columna y nivel en **negrita**.

| Nivel | Método | Acc@1 | BAcc | AUROC |
|---|---|---:|---:|---:|
| L1 | LP paramétrico (LR) | 0,750 | **0,735** | **0,796** |
| L1 | k-NN k=10 *majority* | 0,722 | 0,686 | 0,748 |
| L2 | LP paramétrico (LR) | 0,250 | 0,265 | **0,793** |
| L2 | k-NN k=20 *majority* | **0,278** | **0,265** | 0,700 |
| L3 | LP paramétrico (LR) | 0,179 | 0,184 | **0,813** |
| L3 | k-NN k=5 *majority* | **0,214** | **0,210** | 0,615 |
| L3 | k-NN k=20 *weighted* | 0,214 | 0,210 | 0,719 |

El *retrieval* k-NN supera al sondeo lineal paramétrico sobre el régimen de cola larga severa L3 (224 clases efectivas, 3,5 muestras/clase media en *train*): +3,5 pp Acc@1 (0,179 → 0,214) y +2,6 pp BAcc (0,184 → 0,210) con k = 5 y votación por mayoría. Sobre L2, k-NN con k = 20 empata en BAcc (0,265) y mejora Acc@1 en +2,8 pp (0,250 → 0,278). Sobre L1, k-NN pierde respecto al LP paramétrico (−4,9 pp BAcc, −4,8 pp AUROC). La AUROC global del k-NN es sistemáticamente inferior a la del LP (diferencias entre −4,8 y −9,3 pp), atribuible a que la estimación de probabilidad por frecuencia relativa de vecinos no es densa sobre el conjunto de clases. Coherente con la arquitectura del módulo M4 del prototipo (*retrieval* sobre Derm1M), que emplea el mismo principio de similitud coseno como soporte conceptual. La búsqueda de casos similares en producción la resuelve M4-bis (imagen→imagen, PanDerm Large + FAISS).

## J.8 Zero-shot jerárquico: cascada L1→L2→L3 con propagación de errores

Cascada jerárquica: predecir primero L1, restringir las candidatas L2 a las hijas del L1 predicho, y restringir las candidatas L3 a las hijas del L2 predicho. La jerarquía L1→L2 se deriva del dataset de *train* (Inflamatoria: 25 subcategorías L2; Infecciosa: 5; Tumoral: 12; Genodermatosis: 2). Encoder: DermLIP v2 (el mejor *zero-shot*), *prompts* en castellano. La fila «L2 hier | L1 correcto» restringe la evaluación a las muestras con L1 predicho correctamente. Mejor por columna en **negrita**.

| Variante | L1 Acc@1 | L2 Acc@1 | L3 Acc@1 |
|---|---:|---:|---:|
| ZS plano (referencia) | **0,361** | **0,222** | **0,107** |
| ZS jerárquico (cascada estricta) | 0,361 | 0,139 | 0,036 |
| ZS jerárquico (top-3 L1) | 0,361 | 0,194 | 0,036 |
| L2 hier \| L1 correcto (condicionada) | — | 0,385 | — |

La cascada jerárquica estricta degrada el rendimiento plano sobre L2 en −8,3 pp Acc@1 (0,222 → 0,139) y sobre L3 en −7,1 pp Acc@1 (0,107 → 0,036). La causa principal es la propagación de errores desde L1: dado que DermLIP v2 acierta L1 solo en el 36,1 % de los casos, el 63,9 % restante recibe en L2 un conjunto de candidatas desalineado con la verdad de terreno. La variante *top-3 L1* amortigua parcialmente la propagación y recupera 0,194 Acc@1 sobre L2 (−2,8 pp respecto a plano), sin efecto sobre L3. Condicionado a L1 correcto, la cascada jerárquica L2 alcanza Acc@1 = 0,385, casi el doble del plano: la restricción del espacio de candidatas a las subcategorías clínicamente coherentes con la categoría etiológica acertada incrementa la discriminación de forma sustancial. La arquitectura jerárquica funciona si la decisión inicial sobre la familia etiológica es robusta; con *zero-shot* modesto, la cascada acumula errores y degrada el rendimiento plano.

## J.N Tablas desplazadas del cuerpo (condensación paper-style v6)

Tablas reportadas aquí para no fragmentar la argumentación del capítulo de resultados; sus cifras-ancla figuran en el cuerpo.

### J.N.1 Nomenclatura de los cuatro sistemas evaluados sobre DermapixelAI 1.0

| Sistema | Qué es | Sección cuerpo |
|---|---|---|
| PanDerm LP | Codificador congelado + regresión logística | §4.10 |
| DermLIP zero-shot | Modelo visión-lenguaje externo (sin cabeza entrenada) | §4.11 |
| Dermapixel R0 (cabeza superv. L2) | Rama supervisada con adaptación LoRA r=16 | §4.12 |
| Dermapixel R0 (rama contrastiva) | Rama CLIP-style castellana (SpanDerm-CLIP), zero-shot de clase abierta; **no desplegada** | §4.13 |

### J.N.2 LoRA frente a ajuste fino denso sobre L2 (DermapixelAI 1.0, N_test=36)

| Protocolo | BAcc L2 | AUROC L2 | Params entren. |
|---|---|---|---|
| LP test fijo (referencia) | 0,265 | 0,793 | 39 K |
| FT parcial denso | 0,324 | 0,852 | 25,2 M |
| **Dermapixel R0 (LoRA)** | **0,363 ± 0,007** | **0,855 ± 0,006** | **524 K** |

### J.N.3 Brechas zero-shot cuantificadas sobre DermapixelAI 1.0

| Contraste | Métrica | Gap |
|---|---|---|
| DermLIP v2, L1, castellano vs. inglés | Acc@1 | -13,9 pp |
| SigLIP-SO400M, L1, castellano vs. inglés | Acc@1 | -11,1 pp |
| DermLIP v2, L2, castellano vs. inglés | AUROC | -3,2 pp |
| LP supervisado vs. zero-shot, L1 | BAcc | +29,4 pp |
| LP supervisado vs. zero-shot, L3 | BAcc | +10,5 pp |

### J.N.4 Recuperación cross-modal i2t R@5 de la rama contrastiva de Dermapixel R0

En su régimen más favorable (validación, descripciones ricas `llm_case_summary`, checkpoint v2) la recuperación texto→imagen alcanza i2t R@5 = **0,654** sobre los 107 pares (imagen, descripción rica): el caso correcto aparece entre los cinco primeros en el 65,4 % de las consultas, frente a una referencia de azar de ≈ 0,047 (≈ 5/107) sobre ese conjunto. Dentro de ese régimen v2 es el mejor *checkpoint* recuperando, aunque v4 sea el mejor clasificando (zero-shot L3 Acc@1 = 0,247), lo que confirma que clasificar y buscar no son el mismo objetivo. La trayectoria completa por versión (v1→v5) figura en el material reproducible.

| Régimen de medición | i2t R@5 | Azar | Lift |
|---|---:|---:|---:|
| Validación, *rich-caption*, checkpoint v2 (107 pares) | **0,654** | ≈0,047 | ≈14× |
| Uso real: consulta corta, imágenes *held-out*, archivo completo (~1 000 img) | ≈0,02 | ≈0,005 | orden de azar |

**Esta cifra no generaliza y la rama no se despliega.** El 0,654 caracteriza únicamente el escenario de validación con texto rico. Medida como se usaría en producción —consulta corta de usuario, imágenes *held-out* no vistas en entrenamiento y archivo completo de ~1 000 imágenes— la recuperación se desploma al orden del azar (i2t R@5 ≈ 0,02 frente a un azar de ≈ 0,005). El experimento demuestra la viabilidad técnica de un espacio dual texto-imagen en castellano —la arquitectura entrena, converge y es recuperable en validación con texto rico—, no un buscador útil. En consecuencia, SpanDerm-CLIP (la rama contrastiva de Dermapixel R0) **no se despliega**; en producción la búsqueda de casos similares la resuelve **M4-bis** (imagen→imagen, PanDerm Large + FAISS).

### J.N.5 Combinación de codificadores sobre DermapixelAI 1.0 (`tab:dermapixel-ensemble`)

Tabla desplazada del cuerpo (§4.10). Cabeza logística por codificador (PanDerm Base, PanDerm Large, DermLIP v2; protocolo de sondeo lineal) sobre los *embeddings* congelados, frente a cinco estrategias de combinación, en el mismo test fijo por nivel. Cifra-ancla en el cuerpo: ningún codificador lidera simultáneamente los tres niveles (L1 mejor con la combinación avg Large+DermLIP, AUROC 0,817; L2 con DermLIP v2 individual, AUROC 0,860; L3 con PanDerm Large individual, AUROC 0,813). Mejor por columna y nivel en **negrita**.

| Nivel | Estrategia | Acc@1 | Acc@3 | BAcc | AUROC |
|---|---|---:|---:|---:|---:|
| L1 (4 cls) | LP PanDerm Base | 0,639 | — | 0,570 | 0,697 |
| L1 (4 cls) | LP PanDerm Large | **0,750** | **1,000** | **0,735** | 0,796 |
| L1 (4 cls) | LP DermLIP v2 | 0,694 | **1,000** | 0,680 | 0,789 |
| L1 (4 cls) | Avg L+D | **0,750** | **1,000** | 0,702 | **0,817** |
| L1 (4 cls) | Avg 3 | 0,722 | **1,000** | 0,694 | 0,811 |
| L1 (4 cls) | Máx 3 | 0,694 | **1,000** | 0,652 | 0,805 |
| L1 (4 cls) | Stack 3 | 0,694 | **1,000** | 0,703 | 0,790 |
| L2 (38 cls) | LP PanDerm Base | 0,222 | — | 0,221 | 0,707 |
| L2 (38 cls) | LP PanDerm Large | 0,250 | 0,500 | 0,265 | 0,793 |
| L2 (38 cls) | LP DermLIP v2 | **0,333** | 0,528 | **0,279** | **0,860** |
| L2 (38 cls) | Avg L+D | 0,278 | **0,556** | **0,279** | 0,832 |
| L2 (38 cls) | Avg 3 | 0,278 | 0,528 | 0,268 | 0,821 |
| L2 (38 cls) | Máx 3 | 0,222 | 0,528 | 0,235 | 0,797 |
| L2 (38 cls) | Stack 3 | 0,250 | 0,528 | 0,258 | 0,804 |
| L3 (224 cls) | LP PanDerm Base | 0,143 | — | 0,132 | 0,735 |
| L3 (224 cls) | LP PanDerm Large | **0,179** | **0,286** | **0,184** | **0,813** |
| L3 (224 cls) | LP DermLIP v2 | 0,107 | 0,214 | 0,079 | 0,809 |
| L3 (224 cls) | Avg L+D | **0,179** | 0,250 | **0,184** | 0,802 |
| L3 (224 cls) | Avg 3 | 0,143 | 0,250 | 0,152 | 0,795 |
| L3 (224 cls) | Máx 3 | 0,143 | 0,250 | 0,132 | 0,780 |
| L3 (224 cls) | Stack 3 | 0,143 | 0,250 | 0,158 | 0,791 |

### J.N.6 Ajuste fino parcial multinivel del codificador (`tab:dermapixel-ft-multilevel`)

Tabla desplazada del cuerpo (§4.12). PanDerm Large con las dos últimas capas descongeladas (25,2 M parámetros entrenables) y cabeza *fully-connected*, sobre los tres niveles ontológicos. Test fijo N_test = 36 en L1/L2 y N_test = 28 en L3. La cabeza L2 dimensiona 1 024 → 37 (37 clases L2 efectivas en *train*; 38 nominales). Entre corchetes, IC95 % por *bootstrap* estratificado (1 000 remuestreos). Cifra-ancla en el cuerpo: el ajuste fino parcial alcanza la mejor BAcc en L1 (0,622, +18,2 pp sobre el sondeo lineal congelado) y la mejor AUROC en L2 (0,852). Mejor por columna en **negrita**.

| Nivel | Acc@1 | Acc@3 | BAcc | AUROC |
|---|---|---|---|---|
| L1 (4 cls) | **0,583** [0,444–0,722] | **0,972** [0,917–1,000] | **0,622** [0,510–0,735] | 0,715 [0,603–0,830] |
| L2 (37 cls) | 0,250 [0,167–0,333] | 0,500 [0,389–0,611] | 0,324 [0,265–0,397] | **0,852** [0,814–0,888] |
| L3 (224 cls) | 0,179 [0,143–0,214] | 0,286 [0,214–0,357] | 0,184 [0,158–0,211] | 0,804 [0,773–0,835] |
