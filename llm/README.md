# Anexo G — Modelos de lenguaje multimodales: comparación extendida

> Material complementario del TFG EPS0270. Corresponde al antiguo Anexo G de la memoria. Amplía la comparación entre modelos de lenguaje multimodales generalistas: *prompt* clínico empleado, análisis de patrones de error sobre HAM10000, sensibilidad al *prompt* en DermLIP v2 y comparación de costes operativos agregados.

## G.1 Modelos evaluados

Los modelos accesibles vía API se han evaluado con la versión disponible en marzo de 2026; los modelos locales se han ejecutado sobre la DGX Spark con chip Grace Hopper GB10 en precisión BF16 sin cuantización adicional.

| Modelo | Proveedor | Parámetros | Acceso |
|---|---|---|---|
| GPT-4o | OpenAI | No público | API REST |
| GPT-4o-mini | OpenAI | No público | API REST |
| Gemini 2.5 Pro | Google | No público | API REST |
| Gemini 2.5 Flash | Google | No público | API REST |
| MedGemma 4B Vision | Google | ~4 B | Local (Ollama) |
| MedGemma 27B Text | Google | ~27 B | Local (DGX Spark, BF16) |
| MedGemma 27B + LoRA | Google | ~27 B + 41 M LoRA | Local (DGX Spark, BF16) |
| BLIP-2 Flan-T5-XL | Salesforce | — | Local (HuggingFace) |
| InstructBLIP Flan-T5-XL | Salesforce | — | Local (HuggingFace) |

## G.2 Prompt clínico estandarizado

El *prompt* tiene aproximadamente 300 palabras y se mantiene constante entre los nueve proveedores evaluados, lo que elimina la formulación del *prompt* como variable confusoria:

> *You are assisting in a dermatology research study. This is a dermoscopic image of a skin lesion taken during clinical practice. Your task is to classify the lesion as exactly one of the following seven categories used in the HAM10000 dataset: melanoma (mel), melanocytic nevus (nv), basal cell carcinoma (bcc), actinic keratosis (akiec), seborrheic keratosis (bkl), dermatofibroma (df) or vascular lesion (vasc). Provide your answer as a single diagnostic label followed by a brief clinical justification (1–2 sentences). The justification should mention the morphological features that support the diagnosis (e.g., asymmetry, border regularity, color pattern, vascular structures). Avoid hedging language; commit to the most likely diagnosis based on the visible features. Do not refuse to answer. Respond only with the category name and the justification.*

Para Gemini 2.5 Pro y Gemini 2.5 Flash, el *prompt* se ajusta levemente para evitar la activación de filtros de seguridad (*refusal rate* documentado: ~22 % en Gemini 2.5 Pro sobre HAM10000 frente a <1 % en GPT-4o). La etiqueta predicha se extrae mediante *parsing* determinista que mapea sinónimos admitidos al código canónico de HAM10000.

## G.3 Comparativa completa de paradigmas sobre HAM10000, PAD-UFES-20 y DDI

Sobre HAM10000 (N = 1 232), PAD-UFES-20 (N = 461) y DDI (N = 137). FT: ajuste fino; LP: sondeo lineal; ZS: zero-shot. Las tres primeras filas son los especialistas de referencia (secciones 4.2 y 4.4 del cuerpo); esta tabla corresponde a `tab:llm-comparativa` desplazada del cuerpo de la memoria (cifra-ancla en §4.8: accuracy 0,920 del mejor especialista frente a 0,485 del mejor generalista zero-shot, brecha 43,5 pp). Mejor por columna en **negrita**.

| Modelo | Modo | HAM Acc | HAM BAcc | PAD Acc | PAD BAcc | DDI Acc | DDI BAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| PanDerm Base (TTA HAM) | FT | **0,920** | **0,852** | 0,755 | **0,704** | 0,774 | 0,693 |
| PanDerm Large | LP | 0,888 | 0,575 | **0,772** | 0,642 | **0,847** | **0,764** |
| SigLIP-Large | LP | 0,900 | — | 0,718 | — | 0,789 | — |
| MedGemma 27B + LoRA | FT | 0,802 | 0,270 | 0,430 | 0,347 | 0,657 | 0,618 |
| MedGemma 27B | ZS | 0,665 | 0,243 | 0,447 | 0,342 | 0,474 | 0,566 |
| GPT-4o | ZS | 0,485 | 0,465 | 0,553 | 0,523 | 0,723 | 0,648 |
| Gemini 2.5 Pro | ZS | 0,403 | — | 0,477 | — | 0,441 | — |
| Gemini 2.5 Flash | ZS | 0,380 | — | 0,486 | — | 0,555 | — |
| GPT-4o-mini | ZS | 0,256 | 0,319 | 0,390 | 0,435 | 0,737 | 0,556 |
| DermLIP | ZS | 0,239 | 0,581 | 0,601 | — | 0,329 | — |
| BLIP-2 (Flan-T5-XL) | ZS | 0,015 | 0,145 | 0,017 | 0,167 | 0,796 | 0,542 |
| InstructBLIP (Flan-T5-XL) | ZS | 0,024 | 0,123 | 0,171 | 0,224 | 0,788 | 0,500 |

## G.4 Latencias, refusal rates y costes operativos

*Refusal* sobre Gemini 2.5 Pro corresponde a respuestas bloqueadas por filtros de seguridad. «Coste» agrega HAM10000 + PAD-UFES-20 + DDI.

| Modelo | Latencia/imagen | Refusal HAM | Refusal PAD | Refusal DDI | Coste total |
|---|---:|---:|---:|---:|---|
| GPT-4o | ~4 s | <1 % | <1 % | <1 % | ~$2,71 |
| GPT-4o-mini | ~3 s | <1 % | <1 % | <1 % | ~$1,36 |
| Gemini 2.5 Pro | ~8 s | 21,5 % | 15,0 % | 18,2 % | — |
| Gemini 2.5 Flash | ~2 s | 0,3 % | 2,4 % | 2,7 % | — |
| MedGemma 27B | 1,76 s | 0 % | 0 % | 0 % | 0 EUR |
| MedGemma 27B + LoRA | 1,82 s | 0 % | 0 % | 0 % | 0 EUR (entrenam.) |

Tres observaciones: (i) Gemini 2.5 Pro presenta *refusal rate* elevado sobre HAM10000 (21,5 %), atribuible a filtros de seguridad sobre imagen clínica. (ii) MedGemma 27B y su variante con LoRA operan localmente sin coste recurrente, con latencias de ~1,8 s por imagen, inferiores a las de los proveedores comerciales. (iii) El coste agregado de GPT-4o-mini (~$1,36 sobre 1 830 imágenes) es de ~$0,001 por imagen; el de GPT-4o (~$2,71) duplica esta cifra.

## G.5 Patrones de error sobre HAM10000

La fila «Predicciones GPT-4o» indica el número de imágenes asignadas a cada clase por el modelo; la comparación con N_test revela los sesgos sistemáticos.

| Clase | N_test | Predicciones GPT-4o | Factor |
|---|---:|---:|---|
| actinic keratosis | 35 | 2 | 0,06× |
| basal cell carcinoma | 44 | — | — |
| seborrheic keratosis | 107 | — | — |
| dermatofibroma | 8 | 284 | 35,5× |
| melanoma | 70 | 299 | 4,3× |
| melanocytic nevus | 951 | — | — |
| vascular lesion | 17 | — | — |

El sesgo más acusado es la sobreasignación de *dermatofibroma* (284 frente a 8 muestras reales, factor 35,5×). El modelo responde al espacio léxico del *prompt*: la enumeración explícita de *dermatofibroma* activa preferentemente esta categoría cuando la evidencia visual no es saliente. Patrón análogo de menor magnitud con melanoma (4,3×), clase clínicamente prioritaria que el modelo podría sobreasignar por un sesgo precautorio aprendido.

GPT-4o-mini presenta un patrón distinto: sesgo masivo hacia *seborrheic keratosis* (528 predicciones frente a 107 reales) y melanoma (359 frente a 70). Solo 196 de los 951 casos de *melanocytic nevus* se clasifican correctamente (*recall* 0,196), mientras GPT-4o alcanza *recall* 0,505 sobre la misma clase.

## G.6 MedGemma 27B con LoRA: análisis por dataset

Adaptación con LoRA sobre MedGemma 27B (*rank* r = 16, α = 32, dropout 0,05, 1 época, η = 2×10⁻⁵, *effective batch* 8, ~41 M parámetros entrenables sobre el *split* de entrenamiento de HAM10000).

| Clase | N_test | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| actinic keratosis | 35 | 0,32 | 0,26 | 0,29 |
| basal cell carcinoma | 44 | 0,34 | 0,25 | 0,29 |
| dermatofibroma | 8 | 0,00 | 0,00 | 0,00 |
| melanocytic nevus | 951 | 0,85 | 0,99 | 0,92 |
| melanoma | 70 | 0,38 | 0,36 | 0,37 |
| seborrheic keratosis | 107 | 0,67 | 0,04 | 0,07 |
| vascular lesion | 17 | 0,00 | 0,00 | 0,00 |

La concentración del rendimiento sobre la clase mayoritaria (nevus melanocítico: *recall* 0,99) a costa de las minoritarias (dermatofibroma y vascular lesion: F1 0,00) contrasta con PanDerm Base con ajuste fino y TTA sobre el mismo dataset, que mantiene *recall* superior a 0,65 sobre todas las clases minoritarias gracias al *weighted random sampler* de la receta original. El *fine-tuning* de LLMs multimodales sobre datasets desbalanceados requiere estrategias específicas de balanceo no contempladas en la configuración estándar de LoRA.

## G.7 Sensibilidad al prompt en DermLIP v2

El módulo de zero-shot del prototipo opera sobre DermLIP v2 con envoltorio automático del *prompt*. Curva de AUROC observada en función de la longitud del *prompt* sobre HAM10000:

| Longitud del *prompt* | Estrategia | AUROC HAM10000 |
|---|---|---|
| 1 palabra | Etiqueta desnuda | ≈ aleatorio |
| ~5 palabras | Envoltorio clínico (`dermoscopy image of {X}`) | ~0,72 |
| 8–15 palabras | Envoltorio + 1 rasgo clínico | **0,854** |
| ~20 palabras | Envoltorio + 3 conceptos Derm1M | 0,842 |
| > 25 palabras | Envoltorio + 5 conceptos Derm1M | 0,826 |

La curva presenta un punto óptimo en 8–15 palabras con un único rasgo clínico añadido (AUROC 0,854). El rendimiento decae tanto en el régimen muy corto (etiqueta desnuda: AUROC aleatorio) como en el muy largo (>25 palabras: AUROC 0,826).

Causa propuesta para el régimen muy corto: PubMedBERT (encoder textual de DermLIP v2) está preentrenado sobre *captions* de Derm1M con estructura `dermoscopy image of [clase], [descripción]`. Una palabra aislada se proyecta sobre regiones del espacio de *embedding* muy poco pobladas, produciendo similitudes coseno arbitrarias. Para el régimen muy largo: la acumulación de conceptos clínicos introduce ruido semántico que diluye la señal de la clase principal.

El *wrapping* automático del prototipo implementa esta curva en tres capas: presets predefinidos con *prompts* ya en el punto óptimo, un composable `useZeroShot.js` que aplica el envoltorio `dermoscopy image of {term}` cuando la entrada del usuario contiene cinco o menos palabras, y un conmutador en la interfaz que permite al dermatólogo desactivar el envoltorio. La extensión del barrido sistemático de longitud de *prompt* a los demás datasets constituye una línea de continuación inmediata.

## G.8 Síntesis

La evaluación extendida confirma los patrones cualitativos: brecha de 43,5 pp accuracy entre el mejor especialista ajustado (PanDerm Base FT con TTA) y el mejor LLM multimodal generalista *zero-shot* (GPT-4o); sesgo léxico sistemático en los modelos cerrados; *refusal rates* elevados en Gemini 2.5 Pro que limitan su utilidad clínica; y asimetría en la adaptación con LoRA sobre MedGemma 27B. Los LLMs multimodales generalistas no constituyen sustitutos defendibles de los encoders fundacionales específicos del dominio en el régimen clínico evaluado, pero pueden complementarlos como generadores de razonamiento textual estructurado.
