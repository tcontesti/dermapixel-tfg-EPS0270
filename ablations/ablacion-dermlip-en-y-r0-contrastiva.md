# Anexo I — Ablación: modelo fundacional inglés (DermLIP) con traducción ES–EN

> Material complementario del TFG EPS0270. Corresponde al antiguo Anexo I de la memoria. Documenta una ablación independiente que cuantifica el *trade-off* entre la rama contrastiva de Dermapixel R0 y un *wrapper* alternativo basado en un modelo fundacional dermatológico inglés (DermLIP v2) con traducción de etiquetas ES–EN, sobre el mismo conjunto de test *holdout* de DermapixelAI 1.0. Incluye los detalles de reproducibilidad y el material complementario de la rama contrastiva.

## I.1 Motivación

¿Se justifica el esfuerzo experimental de la rama contrastiva de Dermapixel R0 (cinco iteraciones de entrenamiento cross-modal castellano) frente a una alternativa pragmática: emplear un modelo fundacional dermatológico inglés ya alineado imagen-texto, sin entrenar ningún modelo adicional, sobre las etiquetas castellanas traducidas al inglés? Esta ablación cuantifica el *trade-off* sobre el mismo conjunto de test *holdout* de DermapixelAI 1.0.

## I.2 Protocolo experimental

El modelo fundacional inglés evaluado es DermLIP v2 (`PanDerm-base-w-PubMed-256`), un sistema CLIP-style con codificador visual PanDerm-Base y codificador textual PubMedBERT, preentrenado contrastivamente sobre Derm1M (≈403 000 pares imagen-texto). Ambas torres permanecen congeladas; no se entrena ningún parámetro. Las 54 clases L3 representadas en el test se tradujeron al inglés clínico mediante un glosario estándar (Bolognia/Rook) con verificación cruzada de unicidad, y por cada clase se generó un *rich prompt* clínico en inglés (~37 palabras). La inferencia replica el protocolo híbrido (prototipos visuales sobre *train* más texto rico, régimen *H6* w_v = 0,3, w_t = 0,7) sobre las mismas 101 imágenes de test y 54 clases.

Adicionalmente se midió el *eje idioma puro* sobre el *checkpoint* v4 de Dermapixel R0: el mismo modelo evaluado con *prompts* castellanos originales frente a los mismos *prompts* traducidos al inglés.

## I.3 Resultados

Test *holdout* DermapixelAI 1.0, N = 101, 54 clases L3 representadas; régimen *H6 hybrid* v30_r70 salvo donde se indica. Mejor en **negrita**.

| Sistema | Régimen | acc@1 L3 | BAcc L3 | acc@5 L3 |
|---|---|---:|---:|---:|
| Dermapixel R0 v4 | H6 hybrid (castellano) | 0,2475 | 0,2685 | 0,4455 |
| Dermapixel R0 v5 | H6 hybrid (castellano) | 0,2079 | 0,2185 | 0,4158 |
| Dermapixel R0 v4 | H6 hybrid (inglés) | 0,2277 | 0,2556 | 0,4554 |
| DermLIP v2 | H6 hybrid (inglés) | 0,2871 | 0,3667 | 0,6436 |
| **DermLIP v2** | **templates+rich (inglés)** | **0,3861** | 0,4099 | 0,6931 |
| DermLIP v2 | visual-only | 0,2178 | 0,2278 | 0,4851 |

## I.4 Interpretación

DermLIP v2 inglés supera a Dermapixel R0 v4 en el régimen comparable *H6* (0,2871 frente a 0,2475, +3,96 pp) y con mayor margen en su mejor régimen de plantillas más texto rico (0,3861, +13,86 pp), confirmando que el cuello de botella de Dermapixel R0 residía en la *escala* de la alineación cross-modal y no en la formulación textual. Un modelo fundacional preentrenado sobre ≈403 000 pares supera a un CLIP castellano dedicado entrenado sobre 854 pares, incluso pagando el ruido de la traducción.

El eje idioma matiza: sobre el mismo *checkpoint* v4, la consulta en castellano supera a la consulta en inglés (0,2475 frente a 0,2277 en *H6*, +1,98 pp; y 0,1287 frente a 0,0891 en *rich-only*, +3,96 pp), mientras que el régimen visual-only permanece idéntico (0,1683). Dermapixel R0 sí captura un componente específico del registro clínico castellano, si bien de segundo orden frente a la ventaja de escala del modelo fundacional inglés.

*Trade-off* explícito:

- **DermLIP inglés más traducción**: rendimiento bruto superior, cero entrenamiento, pero dependencia de traducción de etiquetas y de un dataset de preentrenamiento externo (Derm1M), con pérdida potencial de matices clínicos castellanos no traducibles uno a uno.
- **Dermapixel R0 (rama contrastiva)**: entrenamiento dedicado sobre 854 pares reales castellanos, preservación de la *fingerprint* clínica del archivo de la Dra. R. Taberner, inferencia sin traducción en *runtime* y operación *on-prem* compatible con la ruta de extensión al banco hospitalario HUSLL.

La rama contrastiva de Dermapixel R0 no se justifica como sistema de mayor rendimiento bruto en *zero-shot* L3, sino por sus propiedades operativas de privacidad, especificidad lingüística y autonomía de despliegue. La palanca pendiente para Dermapixel R0 es la anotación experta dirigida *in-domain* sobre las clases L3 con peor cobertura, no la escala genérica.

## I.5 Limitaciones de la ablación

La comparación no está equiparada en datos: DermLIP v2 se preentrena sobre ≈403 000 pares frente a los 854 pares castellanos de Dermapixel R0, y los *backbones* visuales difieren (PanDerm-Base contrastivo frente a PanDerm-Large congelado con LoRA). Debe leerse como «modelo fundacional inglés grande más traducción» frente a «CLIP castellano dedicado pequeño», no como contraste de igual presupuesto de datos.

La carga de DermLIP v2 requiere el *fork* de `open_clip` del proyecto Derm1M: la implementación estándar instancia un Vision Transformer genérico e ignora silenciosamente la torre visual PanDerm-Base (152 claves ausentes, codificador visual aleatorio), un modo de fallo verificado y evitado en esta ablación empleando el mismo *loader* que el módulo de producción.

La verificación de solapamiento con Derm1M se realizó a nivel de fuente: las fuentes constituyentes de Derm1M (vídeo, foros, literatura biomédica y libros de texto en inglés y chino) no incluyen ningún blog dermatológico español, y no se hallaron coincidencias para *dermapixel* ni *taberner* en las 417 257 filas del *manifest*. El archivo Dermapixel es privado y no figura como fuente; no se ejecutó una comparación perceptual exhaustiva imagen a imagen. La traducción ES–EN introduce un ruido no cuantificado en las etiquetas.

## I.6 Detalles de reproducibilidad de la rama contrastiva

**Generación de *captions* sintéticas.** Las ocho perspectivas clínicas por imagen se generan mediante un LLM a partir del texto original de cada caso y de sus metadatos. Total: 7 680 *captions* sobre 960 imágenes (fase v1–v4); 102 imágenes fallan la generación automática y conservan solo el texto base. Coste real 32,54 USD (0,030 USD por imagen completa), longitud media 87,5 palabras por *caption*. Combinadas con las nueve variantes textuales base (plantillas L1/L2/L3, jerárquicas `L1L2L3`, título de blog, tres *chunks* de `case_text` y `diagnosis_raw`), el dataset de entrenamiento alcanza del orden de 22 000 pares (imagen, texto), muestreados por hash determinista `(seed, epoch, image_id)`.

**Tiempos de entrenamiento.** Sobre NVIDIA DGX Spark GB10: ~18 min (v1), ~3 min (v2), ~5 min (v3), ~9 min (v4) y ~54 min (v5).

**Lección metodológica: longitud de las *captions*.** La generación de *captions* sintéticas del dataset externo de v5 (coste 42,33 USD) reveló la importancia de calibrar la longitud objetivo respecto al umbral del verificador automático. Un primer intento con rango 30–40 palabras produjo una tasa de rechazo del 54 %; recalibrar el objetivo a 40–55 palabras redujo el rechazo a cero. Regla operativa: dimensionar la longitud objetivo con margen explícito sobre el umbral mínimo aceptable, no sobre la media deseada.

## I.7 Material complementario de la rama contrastiva

### Cabeza supervisada L2: best-checkpoint frente a últimas épocas

Sobre el test L2 (N_test = 36, 38 clases con queratinización consolidada). Cifras: media ± desviación estándar sobre tres semillas ({42, 43, 44}). *Best val* aplica selección por máxima BAcc de validación; *últimas 5 ép.* promedia las cinco épocas finales sin selección. Mejor por fila en **negrita**.

| Métrica | Best val checkpoint | Media últimas 5 ép. |
|---|---|---|
| Acc@1 | **0,250 ± 0,000** | 0,209 ± 0,005 |
| Acc@3 | **0,500 ± 0,023** | 0,457 ± 0,011 |
| BAcc | **0,363 ± 0,007** | 0,309 ± 0,006 |
| AUROC | **0,855 ± 0,006** | 0,837 ± 0,008 |
| W-F1 | **0,218 ± 0,013** | — |
| Kappa | **0,220 ± 0,001** | — |

La estabilidad entre semillas es alta (desviación ≤ 0,013, ≤ 0,008 sobre AUROC y BAcc). La diferencia entre el *best val checkpoint* y la media de las últimas cinco épocas (+5,4 pp BAcc, +1,8 pp AUROC, +4,1 pp Acc@1) refleja un sobreajuste suave atenuado por la restricción al 0,17 % de parámetros entrenables de LoRA.

### Barrido de épocas de la iteración v3

Figura `fig_spanderm_epoch_sweep.png`: *accuracy*@1 L1, L2 y L3 y `val i2t_R@5` frente al paso de entrenamiento. L3 oscila en [0,04, 0,12] a lo largo de las diez épocas sin aproximarse al objetivo ≥ 0,25, mientras que `val i2t_R@5` alcanza su máximo en la época 2 y desciende con el entrenamiento conjunto adicional del codificador visual.

### Techo de Linear Probing y diagnóstico mecanístico

Techo *Linear Probing* supervisado sobre *features* 1 024-D del image encoder. Comparativa *frozen* (v2 e7) frente a image-LoRA tras entrenamiento adicional (v3 época 6). Métrica: *accuracy*@1. Las cifras de v3 corresponden a la evaluación intermedia conservada del Sprint97; la reejecución posterior arroja desviaciones < 0,01 absolutas.

| Nivel | LP v2-e7 frozen | LP v3 best.pt (e2) | LP v3 e6 (+image train) | Δ frozen→trained |
|---|---:|---:|---:|---|
| L1 (4 vías) | 0,584 | 0,584 | 0,594 | +0,01 |
| L2 (38 vías) | 0,214 | 0,214 | 0,225 | +0,01 |
| L3 (225 vías) | 0,079 | 0,079 | 0,089 | +0,01 |

Dos observaciones. Primero, el *zero-shot* L3 de v1/v2/v3 (~0,13) supera ya el techo LP supervisado (0,079–0,089) sobre el mismo encoder frozen: las *features* 1 024-D del PanDerm-Large congelado no separan linealmente los diagnósticos castellanos finos, y ningún ajuste textual puede extraer información que las *features* visuales no contienen. Segundo, la LoRA visual de v3 (últimos cuatro bloques, 1,05 M parámetros) eleva el techo LP solo +0,01 por nivel, dentro del ruido: capacidad insuficiente para reesculpir las *features* en el régimen *fine-grained* castellano.

El mecanismo por el que el *zero-shot* de v3 cae sin que el LP suba es que el entrenamiento conjunto desplaza los embeddings visuales hacia el espacio de captions ricas y los aleja del de plantillas de nombre de clase: L1 *zero-shot accuracy*@1 cae 0,426 (e2) → 0,287 (e7) mientras el *train loss* sigue bajando a 0,29, señal de sobreajuste *rich-caption*. La capacidad arquitectónica (v4) es la palanca efectiva; la escala externa genérica (v5) no.

### Calibración por temperature scaling

Los logits *zero-shot* de la rama contrastiva están severamente sobreconfiados. La calibración con *temperature scaling* de un parámetro escalar sobre los 107 pares de validación reduce el ECE sobre los 101 pares de test. Ajuste del escalar τ por optimización LBFGS. La calibración sobre v5 no converge (la optimización LBFGS diverge sobre el espacio embebido *image-LoRA* expandido y τ → 0), por lo que queda registrada como limitación metodológica abierta.

| Nivel | ECE antes | ECE después | τ |
|---|---:|---:|---:|
| L3 (v4) | 0,7262 | 0,0213 | 0,4165 |

La reducción del ECE es del −97 % sobre L3 con un único parámetro escalar (figura `fig_spanderm_calibration.png`). Sin calibración post-hoc, las probabilidades de un sistema CLIP-style *zero-shot* castellano no son interpretables como confianza clínica, por lo que el despliegue debe incorporar este escalado como capa final ajustado sobre el archivo de destino.

### Diagnóstico cuantitativo del techo sobre L2

Rama contrastiva de Dermapixel R0 *zero-shot* L2 frente a las dos referencias internas del capítulo sobre el mismo nivel ontológico. La cifra de la iteración v2 corresponde al test de 101 imágenes del *split* per-case del experimento contrastivo; las dos restantes al test fijo N_test = 36. La comparación es indicativa por la diferencia muestral. Mejor en **negrita**.

| Sistema | Régimen | Acc@1 L2 | Observaciones |
|---|---|---:|---|
| LP PanDerm Large (referencia) | Supervisado L2 cerrado | 0,250 | 38 clases, N_test = 36 |
| Dermapixel R0 cabeza L2 | Supervisado L2 cerrado | 0,250 | 38 clases, N_test = 36 |
| **Dermapixel R0 contrastivo v2** | *Zero-shot* open | **0,133** | 23 clases efectivas, N_test = 101 |

La cabeza supervisada cerrada con LoRA alcanza Acc@1 0,250 sobre L2, frente a 0,133 de la rama contrastiva *zero-shot open-class* (v2) sobre las 23 clases efectivas del test per-case. La diferencia operativa es la capacidad de la rama contrastiva de aceptar consultas textuales libres en castellano sin fijar el vocabulario.

### Síntesis de las cinco iteraciones

Régimen evaluativo *zero-shot* L3 sobre 101 imágenes de test (54 clases L3 representadas), *H6 hybrid* v30_r70 (30 % prototipos visuales + 70 % texto rico) aplicado de forma homogénea. v4 queda a 0,0025 del objetivo fijado a priori de 0,25 acc@1 L3 (dentro del ruido). v5 confirma negativamente que la escala externa genérica con *captions* sintéticas no supera v4. Las cifras Linear Probing BAcc L2 absolutas para v4 (0,1860) y v5 (0,2871) quedan registradas en el material de experimentos del proyecto; el delta relativo +54,4 % queda confirmado.

| Versión | LoRA texto | LoRA visual | acc@1 L3 | Lift i2t R@5 | LP BAcc L2 | Notas |
|---|---|---|---:|---|---:|---|
| v1 | — | — | 0,1782 | — | — | Baseline e5 *full FT* |
| v2 | r=16, α=32 | — | 0,1980 | **14,0×** | — | Pico retrieval rich |
| v3 | r=16, α=32 | r=16, últ. 4/24 | 0,1980 | — | — | Image-LoRA infradotado |
| v4 | r=32, α=64 | r=32, últ. 12/24 | **0,2475** | 12,4× | 0,1860 | A 0,0025 del objetivo |
| v5 | r=32, α=64 | r=32, últ. 12/24 | 0,2079 | 10,8× | 0,2871 | FLAT: escala externa (+54,4 % relativo sobre v4) |

*Nota histórica.* Los valores *rich-only baseline* (~30–80 palabras/clase) fueron 0,1287 (v1), 0,1386 (v2) y 0,1089 (v3); la transición al régimen *H6 hybrid* homogéneo permite la comparativa cross-*checkpoint*.

El experimento articula tres ejes. La capacidad arquitectónica (v3→v4) es la palanca efectiva: el cuello de botella reside en el *image encoder*, no en la formulación textual. La escala externa (v5) queda *flat-to-negative*. El eje de supervisión conceptual —anotaciones del *Seven-Point Checklist* o del Grupo A (ver [`sae/`](../sae/README.md))— queda abierto como línea de aprendizaje activo dirigido. El experimento valida la utilidad de PanDerm Large para tareas castellanas más allá del LP/FT supervisado (v4 a 0,0025 del umbral objetivo L3 0,25) y establece una metodología reproducible para evaluaciones CLIP-style sobre archivos privados de pequeña escala.
