# Anexo F — Sparse Autoencoders y diccionario de conceptos clínicos

> Material complementario del TFG EPS0270. Corresponde al antiguo Anexo F de la memoria. Documenta el módulo de interpretabilidad conceptual basado en Sparse Autoencoders (SAE) sobre las representaciones de PanDerm Large, su evaluación frente al diccionario SkinCon y la propuesta de extensión con conceptos clínicos adicionales validados por la Dra. R. Taberner.
>
> Este módulo se materializa en el prototipo [dermapixel.eu](https://dermapixel.eu) como componente **M3** (conceptos clínicos derivados del SAE Large), uno de los módulos M1–M11 de inferencia descritos en [`../prototype/README.md`](../prototype/README.md). El texto íntegro del capítulo figura en [`../MemoriaTFG.pdf`](../MemoriaTFG.pdf).

## F.1 Motivación

Los encoders auto-supervisados como PanDerm producen representaciones densas (1 024 dimensiones para la variante *Large*) cuyos componentes individuales no admiten interpretación directa. Los Sparse Autoencoders descomponen estas representaciones en un diccionario disperso de *features*, donde cada *feature* puede asociarse, en buena parte de los casos, a un concepto clínicamente identificable. DermFM-Zero adopta este mecanismo, pero sus pesos no son públicos. El presente trabajo replica el procedimiento sobre PanDerm Large con pesos liberados y lo valida sobre SkinCon como ground-truth conceptual.

## F.2 Arquitectura del SAE

Configuración comparada del SAE entrenado en el trabajo («SAE Large») y del SAE publicado por DermFM-Zero («SAE Base»).

| Parámetro | SAE Base (DermFM-Zero) | SAE Large (este trabajo) |
|---|---|---|
| Encoder visual base | PanDerm Base (ViT-B/16) | PanDerm Large (ViT-L/16) |
| Dimensión de entrada | 768 | 1 024 |
| *Features* aprendidas | 6 144 (8×) | 16 384 (16×) |
| Arquitectura | TiedBias + UnitNormDecoder | Pre/PostBias + Linear + UnitNormDecoder |
| Pérdida | L1 (3×10⁻⁵) + L2 | L1 (1×10⁻³) + MSE |
| Datos de entrenamiento | No especificado (pipeline DermFM-Zero) | 50 454 imágenes (7 datasets) |
| Épocas | 200 | 100 (convergido) |
| Sparsity media observada | — | 16,5 % (2 753 ± 112 activas) |

Los siete datasets de entrenamiento del SAE Large son HAM10000 (10 015 imágenes), BCN20000 (12 413), Dermnet (19 559), PAD-UFES-20 (2 298), Fitzpatrick17k (3 887), DDI (647) e HIBA (1 635), con un total de 50 454 imágenes (composición exacta en [`tables/tablas-detalladas.md`](../tables/tablas-detalladas.md)). La elección de L1 (1×10⁻³) en lugar del valor publicado por DermFM-Zero (3×10⁻⁵) responde a la dimensión de entrada mayor: el factor L1 relativo a la magnitud típica del *embedding* se mantiene comparable.

## F.3 Comportamiento del SAE Large frente a SAE Base

Discriminación binaria sobre cuatro tareas con *embeddings* crudos y con activaciones del SAE. Media ± desviación estándar sobre 20 particiones aleatorias. Mejor por fila en **negrita** (figura `fig_sae_comparison.png`).

| Tarea | Raw 768D (Base) | SAE Base (6 144) | Raw 1 024D (Large) | SAE Large (16 384) |
|---|---|---|---|---|
| Melanoma | 0,906 ± 0,020 | 0,897 ± 0,019 | 0,897 ± 0,020 | **0,900 ± 0,019** |
| Sesgo pelo | 0,888 ± 0,025 | 0,879 ± 0,023 | 0,884 ± 0,023 | **0,889 ± 0,024** |
| Sesgo tinta | **0,848 ± 0,024** | 0,836 ± 0,025 | 0,846 ± 0,028 | 0,851 ± 0,028 |
| Sesgo regla | 0,888 ± 0,024 | 0,889 ± 0,026 | 0,894 ± 0,022 | **0,908 ± 0,022** |

El SAE Base degrada consistentemente la AUROC respecto a los *embeddings* crudos correspondientes (promedio −0,008 pp), mientras que el SAE Large produce mejoras de +0,007 pp de promedio sobre los *embeddings* crudos de 1 024 dimensiones. El SAE Large preserva la información discriminativa relevante para clasificación binaria pese a la proyección al espacio disperso de 16 384 *features*.

## F.4 Concept Bottleneck Model sobre SkinCon

La validación clínica del SAE Large se realiza mediante un *Concept Bottleneck Model* (CBM) sobre SkinCon, que anota densamente 3 230 imágenes con 48 conceptos clínicos por dermatólogos certificados. Cada imagen se codifica con PanDerm Large; su *embedding* de 1 024 dimensiones se proyecta al espacio disperso del SAE Large; sobre las 16 384 activaciones se entrena un clasificador lineal binario por concepto (regresión logística L-BFGS con C = 1,0, `max_iter` = 5 000, `random_state` = 42). Métrica: AUROC del mejor *feature* individual como predictor del concepto.

### Conceptos con mejor predictor

n₊: número de imágenes positivas para el concepto en SkinCon. AUROC: del mejor *feature* individual del SAE como predictor binario del concepto.

| Concepto clínico | n₊ | Mejor *feature* | AUROC |
|---|---:|---|---:|
| Pedunculated | 26 | #1930 | 0,929 |
| Friable | 153 | #5973 | 0,913 |
| Exophytic/Fungating | 42 | #10114 | 0,908 |
| Comedo | 24 | #7191 | 0,903 |
| Ulcer | 154 | #13908 | 0,897 |
| Exudate | 144 | #5973 | 0,884 |
| Wheal | 21 | #9552 | 0,884 |
| Black | 90 | #3926 | 0,846 |
| Telangiectasia | 100 | #11172 | 0,828 |
| Nodule | 189 | #1930 | 0,818 |
| Umbilicated | 49 | #13783 | 0,810 |

Cada *feature* del SAE puede actuar como predictor para más de un concepto relacionado (#1930 cubre *pedunculated* y *nodule*; #5973 cubre *friable* y *exudate*), lo que sugiere que el espacio disperso codifica dimensiones clínicas de granularidad superior a la del concepto individual.

### Rendimiento agregado del CBM

Sobre los 22 conceptos de SkinCon con cobertura suficiente, el protocolo *CBM-Selected* (regresión logística entrenada sobre 134 *features* del SAE Large correlacionadas con cada concepto) alcanza AUROC media de 0,880. El protocolo de comparación *LP-Direct* (regresión logística sobre el *embedding* crudo de 1 024 dimensiones de PanDerm Large, sin SAE) alcanza AUROC media de 0,864 sobre los mismos conceptos. CBM-Selected supera a LP-Direct en 16 de los 22 conceptos evaluados (once conceptos superan AUROC 0,80, máximos en *pedunculated* 0,929 y *friable* 0,913). Este comportamiento —las *features* dispersas preservan, y en la mayoría de conceptos exceden, el poder predictivo de la representación densa subyacente— sostiene el uso del SAE Large como módulo de interpretabilidad M3 del prototipo dermapixel.eu, que expone los conceptos clínicos por imagen al clínico (véase [`../prototype/README.md`](../prototype/README.md)).

## F.5 Propuesta de extensión: quince conceptos clínicos adicionales

La revisión clínica de la Dra. R. Taberner sobre DermapixelAI 1.0 ha identificado 15 conceptos clínicos adicionales no contemplados por SkinCon, frecuentes en la práctica dermatológica hispanohablante.

| Categoría | Concepto propuesto | Relevancia clínica |
|---|---|---|
| Pigmentación | Pigmentación reticular | Diagnóstico diferencial nevus |
| Pigmentación | Pseudored | Lentigo solar vs. lentigo maligno |
| Vasculatura | Lagunas vasculares | Hemangioma, angioqueratoma |
| Vasculatura | Vasos en horquilla | Carcinoma basocelular |
| Estructura epidérmica | Hiperqueratosis | Queratosis seborreica |
| Estructura epidérmica | Comedo invertido | Queratosis seborreica adelgazada |
| Patrón inflamatorio | Eritema heliotropo | Dermatomiositis |
| Patrón inflamatorio | Patrón en *papel de seda* | Liquen escleroso |
| Distribución | Distribución dermatomérica | Herpes zóster |
| Distribución | Distribución *en alas* | Lupus subagudo |
| Superficie | Cráteres centrales | Molluscum contagioso |
| Superficie | Anillos concéntricos | Eritema multiforme |
| Color | Coloración *salmón* | Psoriasis guttata |
| Color | Halo blanco perilesional | Nevus halo |
| Otros | Bordes geográficos | Vitíligo segmentario |

La incorporación de estos quince conceptos constituye una línea de continuación inmediata, condicionada al muestreo de ~200 imágenes anotadas por concepto para entrenar el clasificador lineal sobre las activaciones del SAE Large.

Sobre esta base, la Dra. R. Taberner está definiendo en una plataforma de anotación autocontenida un vocabulario dermatoscópico propio en castellano, más amplio que las escalas clásicas: una matriz que asocia 37 estructuras dermatoscópicas (retículo pigmentado, velo blanco-azulado, telangiectasias arboriformes, lagunas rojas, estructuras en hoja de arce, lágrimas amarillas o el signo del ala delta, entre otras) con 15 diagnósticos, indicando qué buscar en cada uno. Este vocabulario ampliado aún no se ha empleado para entrenar (trabajo en curso); su finalidad es que, en el futuro, las explicaciones del modelo se expresen en los mismos términos que un dermatólogo emplearía ante el paciente.

## F.6 Limitaciones del módulo SAE

1. La asociación *feature*↔concepto se establece sobre el mejor predictor individual, sin garantía de *monosemanticidad* estricta (la misma *feature* puede correlacionarse con varios conceptos relacionados).
2. El rendimiento del SAE Large sobre discriminación binaria es ligeramente superior al de los *embeddings* crudos, pero esta mejora es del orden de la desviación estándar inter-partición, lo que limita afirmar superioridad estadísticamente significativa sin las pruebas pendientes.
3. La validación conceptual se realiza sobre SkinCon, con sesgo de selección hacia patología dermatoscópica documentada, lo que condiciona la transferencia a otros contextos clínicos.

## F.7 Tablas de detalle del experimento Derm7pt

### E2 — Cabeza intermedia del CBM (pesos por concepto)

AUROC binario sobre el conjunto de test Derm7pt (N_test = 395) por concepto del *Seven-Point Checklist*, y peso aprendido por el meta-clasificador logístico para la predicción binaria melanoma/no-melanoma. Pesos ordenados de mayor a menor importancia relativa.

| Concepto | AUROC concepto | Peso meta-LP |
|---|---:|---:|
| blue_whitish_veil | 0,895 | +2,54 |
| regression_structures | 0,818 | +2,07 |
| streaks | 0,829 | +1,78 |
| vascular_structures | 0,906 | +1,65 |
| pigmentation | 0,792 | +1,44 |
| dots_and_globules | 0,723 | +1,41 |
| pigment_network | 0,901 | +1,04 |

La AUROC binaria de la cabeza intermedia se sitúa en [0,723, 0,906]. *blue whitish veil* y *regression structures* tienen el mayor peso relativo para la decisión final (+2,54 y +2,07), jerarquía que coincide cualitativamente con el peso clínico del *Seven-Point Checklist*.

### E3 — Cruce de los 16 conceptos del Grupo A con el Seven-Point Checklist

Sobre Derm7pt (N = 1 011). Para los conceptos con *mapping* se reporta la AUROC efectiva de la mejor *feature* SAE Large sobre la binarización correspondiente. Conceptos cubiertos (AUROC ≥ 0,70) marcados con ✓.

| # | Concepto Rosa (Grupo A) | *Mapping* Derm7pt | N₊ | AUROC eff | Cobertura |
|---:|---|---|---:|---:|:---:|
| 1 | Red dermatoscópica regular | pigment_network=typical | 381 | **0,748** | ✓ |
| 2 | Red dermatoscópica atípica | pigment_network=atypical | 230 | **0,732** | ✓ |
| 3 | Glóbulos pigmentarios | dots_and_globules | 782 | 0,681 | — |
| 4 | Puntos pigmentarios | dots_and_globules | 782 | 0,681 | — |
| 5 | Estrías radiales | streaks | 358 | 0,688 | — |
| 6 | Pseudopodios | streaks=irregular | 251 | **0,729** | ✓ |
| 7 | Velo azul-blanquecino | blue_whitish_veil=present | 195 | **0,746** | ✓ |
| 8 | Estructuras de regresión | regression_structures ≠ absent | 253 | 0,687 | — |
| 9 | Patrón empedrado | — | — | — | — |
| 10 | Patrón paralelo de surcos | — | — | — | — |
| 11 | Patrón paralelo de crestas | — | — | — | — |
| 12 | Lagunas vasculares | — | — | — | — |
| 13 | Vasos polimorfos | vascular_structures (agregado) | 140 | **0,810** | ✓ |
| 14 | Vasos en horquilla | vascular_structures=hairpin | 15 | **0,926** | ✓ |
| 15 | Vasos en corona | — | — | — | — |
| 16 | Patrón homogéneo desestructurado | pigmentation (*diffuse*) | 380 | **0,704** | ✓ |

De los 16 conceptos del Grupo A, 11 admiten *mapping* operativo (directo en tres casos, por subcategoría en ocho), y los 5 restantes (patrón empedrado, patrón paralelo de surcos, patrón paralelo de crestas, lagunas vasculares, vasos en corona) corresponden a estructuras no contempladas en la escala original. Sobre los 11 mapeados, 7 quedan cubiertos por al menos una *feature* SAE con AUROC efectivo ≥ 0,70, con máximo absoluto de 0,926 sobre vasos en horquilla (solo 15 positivos sobre 1 011 imágenes).

### E4 — Comparativa concepto a concepto: LP separado vs. fine-tuning multitarea

AUROC binaria *present*/*absent* por concepto sobre el conjunto de test Derm7pt (N_test = 395). Comparativa entre sondeo lineal separado por concepto sobre las activaciones SAE Large (E2 LP separado) y las siete cabezas conceptuales del *fine-tuning* multitarea LoRA (E4). Δ en puntos porcentuales.

| Concepto | E2 LP separado | E4 multitarea | Δ |
|---|---:|---:|---:|
| pigment_network | 0,901 | 0,889 | −1,2 pp |
| vascular_structures | 0,906 | 0,880 | −2,6 pp |
| blue_whitish_veil | 0,895 | 0,879 | −1,6 pp |
| streaks | 0,829 | 0,818 | −1,1 pp |
| pigmentation | 0,792 | 0,793 | +0,1 pp |
| regression_structures | 0,818 | 0,791 | −2,7 pp |
| dots_and_globules | 0,723 | 0,725 | +0,2 pp |

Las cabezas conceptuales del régimen multitarea (E4) pierden entre 1 y 3 pp de AUROC binaria frente al sondeo lineal separado (E2) en seis de los siete conceptos, con máximo de −2,7 pp en *regression structures*: degradación menor que confirma que el aprendizaje conjunto de melanoma y conceptos no compromete la calidad conceptual.

## F.N Tablas desplazadas del cuerpo (condensación paper-style v6)

Tablas reportadas aquí para no fragmentar la argumentación del capítulo de resultados; sus cifras-ancla figuran en el cuerpo.

### F.N.1 Alineamiento single-feature SAE Large ↔ Seven-Point Checklist (Derm7pt, N=1011)

AUROC efectiva max(AUROC, 1−AUROC) de la mejor *feature* (top-1) y promedio de las diez mejores (top-10).

| Criterio | N+ | N− | AUROC top-1 | AUROC top-10 |
|---|---|---|---|---|
| pigment_network | 611 | 400 | 0,754 | 0,737 |
| streaks | 358 | 653 | 0,688 | 0,684 |
| pigmentation | 423 | 588 | 0,700 | 0,657 |
| regression_structures | 253 | 758 | 0,687 | 0,673 |
| dots_and_globules | 782 | 229 | 0,681 | 0,663 |
| blue_whitish_veil | 195 | 816 | 0,746 | 0,721 |
| vascular_structures | 188 | 823 | **0,823** | **0,782** |

La AUROC efectiva de la mejor *feature* por criterio se sitúa entre 0,68 y 0,82, con máximo en *vascular structures* (0,823). El alineamiento no descansa en una única «neurona mágica»: la diferencia entre el mejor *feature* (top-1) y el promedio de los diez mejores (top-10) no excede 0,043 de AUROC en ningún criterio, lo que indica que cada concepto se reparte sobre un grupo coherente de *features*. Una inspección cualitativa de *features* representativas de dirección positiva alcanza *precision*@5 = 1,0 (las cinco imágenes que más activan el *feature* comparten el concepto) en cuatro criterios del *Seven-Point Checklist* —estructuras vasculares (#4740, AUROC 0,785), retículo pigmentado (#2040, 0,742), velo azul-blanquecino (#14541, 0,722) y estrías (#1967, 0,680)—. Las cifras de detalle por criterio figuran en los scripts de evidencia `derm7pt_sae_e1`.

### F.N.2 Sondeo lineal directo, CBM jerárquico y *fine-tuning* multitarea (Derm7pt, N_test=395, ≈98 melanomas)

Comparativa sobre la tarea binaria melanoma/no-melanoma entre el sondeo lineal directo sobre las 16 384 *features* del SAE Large, el *Concept Bottleneck* de siete conceptos y el *fine-tuning* multitarea LoRA (r = 16 sobre los dos últimos bloques de PanDerm Large, ocho cabezas paralelas). Mismo *split* oficial de test. AUROC en negrita.

| Método | Params entr. | Acc@1 | BAcc | AUROC | W-F1 | Kappa |
|---|---:|---|---|---|---|---|
| Sondeo lineal directo (SAE → melanoma) | 16 K | 0,841 | 0,735 | **0,890** | 0,829 | 0,527 |
| CBM (SAE → 7 conceptos → melanoma) | ~120 | 0,825 | 0,681 | **0,832** | 0,801 | 0,440 |
| **Multitarea LoRA (8 salidas)** | 0,56 M | **0,841** | **0,804** | **0,891** | **0,843** | **0,590** |

La pérdida de 5,8 pp de AUROC del CBM (0,890 → 0,832) cuantifica el coste de razonar exclusivamente mediante conceptos explícitos. El *fine-tuning* multitarea LoRA recupera ese margen: con solo 0,56 M parámetros entrenables (0,18 % del modelo) alcanza AUROC 0,891 —equivalente al sondeo lineal directo (0,890) y +5,9 pp sobre el CBM— al tiempo que entrega los siete conceptos en la misma pasada, sin coste apreciable de precisión.
