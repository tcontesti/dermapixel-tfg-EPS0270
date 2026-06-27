# Anexo H — Prototipo DermApIxel

> Material complementario del TFG EPS0270. Corresponde al antiguo Anexo H de la memoria. Describe el prototipo *DermApIxel*, sistema integrado de apoyo al diagnóstico dermatológico construido como entorno operativo de los módulos derivados del trabajo experimental. El prototipo **no constituye objeto experimental** del estudio principal; materializa la integración técnica entre los modelos evaluados y un sistema operativo accesible desde navegador. La versión inicial opera con seis módulos de IA (M1–M6, despliegue *end-to-end* de 2026-04-05). La evolución posterior incorpora módulos castellanos adicionales (M7, M9, M10, M4-bis) más un ensemble ponderado (M11).

## H.1 Motivación arquitectónica

Cuatro requisitos vertebran el diseño: (i) separación física entre el equipo cliente (PC del clínico, sin GPU) y el servidor de inferencia (DGX Spark, con GPU); (ii) comunicación asíncrona entre ambos para evitar bloqueos en la interfaz durante inferencias de varios segundos; (iii) persistencia estructurada de imagen original, máscara, conceptos, casos similares, razonamiento textual y validación posterior; y (iv) interfaz web multilingüe (catalán, español, inglés) compatible con flujos de teledermatología.

## H.2 Arquitectura

### Topología cliente-servidor

El sistema se distribuye en dos nodos físicos sobre LAN. El primer nodo (equipo Windows del usuario clínico) aloja la API REST, la base de datos, el proxy inverso y la interfaz web. El segundo nodo (servidor GPU on-prem NVIDIA DGX Spark con chip Grace Hopper GB10) aloja el servidor de inferencia con los seis modelos cargados, el servidor del modelo de lenguaje multimodal de gran tamaño y los modelos de razonamiento auxiliares servidos vía Ollama. La comunicación se realiza exclusivamente vía AMQP sobre RabbitMQ, sin tráfico HTTP directo.

### Componentes del backend

Backend FastAPI 0.115 asíncrono, desplegado como contenedores Docker Compose con cinco servicios.

| Servicio | Imagen | Puerto host | Función |
|---|---|---|---|
| `derm-svc` | FastAPI 0,115 + Uvicorn | 9030 | API REST principal |
| `derm-mysql` | MySQL 8.0 | (interno) | Persistencia estructurada |
| `derm-rabbitmq` | RabbitMQ 3.13 | 5673 / 15675 | Bus AMQP y panel de gestión |
| `derm-nginx` | Nginx 1.25 | 8091 | Proxy inverso y archivos estáticos |
| `result-consumer` | Python asyncio | — | Worker asíncrono que recibe resultados |

### Persistencia

Base de datos con doce tablas relacionales. La tabla raíz `derm_studies` contiene el identificador único, el estado del estudio (`pending`, `processing`, `done`, `error`), los *timings* por etapa y la imagen original como `LONGBLOB`. Las diez tablas restantes contienen los resultados de cada módulo, enlazadas mediante clave foránea con borrado en cascada.

### Bus de mensajería AMQP

Dos colas RabbitMQ: `derm.inference` (*work queue*, una tarea por estudio con *manual ack*) y `derm.results` (*publish-subscribe*, *topic exchange*). El módulo M6 usa una tercera vía RPC sobre AMQP para invocaciones síncronas de baja latencia (~20 ms). Ventajas frente a HTTP directo: tolerancia a fallos, desacoplamiento temporal y trazabilidad auditable.

## H.3 Módulos de inteligencia artificial

Seis módulos cargados simultáneamente sobre la VRAM unificada de la DGX Spark.

| Módulo | Modelo subyacente | Latencia | Función |
|---|---|---|---|
| M1 | PanDerm Large FT (HAM10000, TTA) | ~315 ms | Clasificación cerrada 7 clases |
| M2 | SAM2.1-Large (decoder FT, ISIC2018) | — | Segmentación binaria de lesión |
| M3 | SAE 1 024 → 16 384 + SkinCon | — | 32 conceptos clínicos por imagen |
| M4 | FAISS sobre Derm1M (421 327 vec.) | ~150 ms | Recuperación visual densa *top-5* |
| M5 | MedGemma 27B / GPT-4o / GPT-5 (9 prov.) | 6–300 s | Razonamiento clínico estructurado |
| M6 | DermLIP v2 (PubMedBERT) | ~20 ms | Clasificación abierta zero-shot |

Los módulos M7 (clasificador unificado L1/L2/L3) y M8 (SigLIP-Large SO400M con sondeo lineal sobre HAM10000) están integrados en el pipeline de evaluación pero no se exponen como pestañas independientes.

**M1 — Clasificador cerrado.** PanDerm Large *fine-tuned* sobre HAM10000. La inferencia emplea *test-time augmentation* con cinco augmentaciones deterministas (*recall* de melanoma +4,3 pp con TTA). Salida: vector de siete probabilidades softmax.

**M2 — Segmentación binaria.** SAM2.1-Large con *fine-tuning* del decodificador. Salida: máscara binaria (`LONGBLOB`) con métricas derivadas (asimetría, área, perímetro), superpuesta como overlay semitransparente.

**M3 — Conceptos clínicos.** Interpretabilidad conceptual basada en SAE: la imagen se codifica con PanDerm Large, el *embedding* de 1 024 dimensiones se proyecta al espacio disperso de 16 384 *features* (SAE entrenado sobre 50 454 imágenes de siete datasets), y las activaciones se mapean a los 32 conceptos clínicos de SkinCon mediante regresión logística por concepto.

**M4 — Recuperación visual densa.** Índice FAISS HNSW sobre 421 327 *embeddings* DermLIP de Derm1M. Devuelve las cinco imágenes más similares con su diagnóstico y distancia coseno (~150 ms). M4 devuelve casos análogos para inspección humana, sin alimentar un modelo generativo posterior.

**M5 — Razonamiento clínico estructurado.** Genera explicación textual en español integrando los hallazgos de M1–M4 (y opcionalmente M6). Soporta nueve proveedores intercambiables:

| `provider` | Modelo | Latencia | Coste/llamada | Infraestructura |
|---|---|---:|---|---|
| `gpt-4o-mini` | GPT-4o-mini | ~6 s | ~$0,001 | API OpenAI |
| `gpt-4o` | GPT-4o | ~10 s | ~$0,01 | API OpenAI |
| `gpt-5-mini` | GPT-5 mini | ~15 s | ~$0,001 | API OpenAI |
| `gpt-5-nano` | GPT-5 nano | ~25 s | ~$0,0003 | API OpenAI |
| `gpt-5` | GPT-5 | ~30 s | ~$0,01 | API OpenAI |
| `medgemma-4b` | MedGemma 4B Vision | ~40 s | 0 EUR (local) | Ollama, ~4 GB VRAM |
| `medgemma-vision` | MedGemma 27B Vision (BF16) | 3–5 min | 0 EUR (local) | DGX Spark, 54,9 GB VRAM |

El `provider` por defecto es `gpt-4o-mini`. Los proveedores locales se reservan para escenarios de cumplimiento estricto sobre datos sanitarios. Los nueve comparten el mismo *system prompt* (≥300 palabras, cinco secciones: morfología, dermatoscopia, criterios ABCDE, integración con M1–M4, juicio clínico). Salida JSON con `reasoning`, `differential_diagnosis` (3–5 objetos) y `recommendation`.

**M6 — Clasificación zero-shot abierta.** DermLIP v2 (ViT-B/16 + PubMedBERT) con *prompts* A3 de Derm1M. Acepta lista arbitraria de descripciones y devuelve similitud coseno softmax (~20 ms *warm*). Cuatro *presets* clínicos (31 clases únicas):

| `preset` | Clases | Uso clínico |
|---|---:|---|
| `all_dermatology` | 31 | Default: el clínico no precategoriza |
| `common_dermatoses` | 8 | Sospecha de patología inflamatoria |
| `infections` | 9 | Sospecha de infección |
| `tumors_extended` | 14 | Sospecha de tumor o lesión pigmentada |

Hallazgo operativo: sensibilidad extrema de DermLIP v2 a la formulación del *prompt*. Etiquetas desnudas producen predicciones aleatorias; el envoltorio clínico estandarizado restaura el rendimiento. Punto óptimo en 8–15 palabras con un rasgo clínico (AUROC 0,854). El prototipo implementa *wrapping* automático en tres capas (presets, composable `useZeroShot.js`, conmutador UI).

## H.4 Frontend e interfaz clínica

Vue 3.5 + Tailwind CSS 4.2 + Vite 7, en tres idiomas (catalán, español, inglés), tema claro/oscuro. Vista principal `AnalyzeView` con seis pestañas: `CLASS` (M1), `SEG` (M2 con overlay), `CONC` (M3), `RAG` (M4), `REASON` (M5) y `Z-SHOT` (M6). Vistas complementarias: `HistoryView`, `ValidationView`, `StatsView`. El `LesionViewer` permite zoom real, overlay de máscara con opacidad ajustable y deslizador de umbral dinámico. El `ValidationPanel` expone tres opciones (correcto/parcial/incorrecto) y dibujo manual de *bounding boxes* (persistido en `derm_manual_annotations`).

## H.5 Rendimiento operativo (medidas del 2026-04-05)

| Métrica | Valor | Notas |
|---|---|---|
| Pipeline M1–M4 *end-to-end* | ~315 ms (warm) | Tras carga inicial |
| Pipeline M1–M4 *cold* | ~450 ms | Primera inferencia tras reinicio |
| M6 Zero-shot (3 clases) *warm* | ~20 ms server | ~25 ms incluyendo red |
| M6 Zero-shot (3 clases) *cold* | ~25,5 ms | ~46,9 ms incluyendo red |
| M6 *warmup* kernel PubMedBERT (7 clases) | 614 ms | Una vez tras reinicio |
| VRAM total app M1–M4 + M6 | 4,40 GB | Sin cuantización ni *offloading* |
| VRAM MedGemma 27B Vision | 54,9 GB | Coexiste con app principal |
| VRAM total ocupada | ~59 GB | Sobre 128 GB disponibles |
| Speedup M6 frente a M1–M4 | ~30× | Por arquitectura ligera y *embeddings* cacheados |

La latencia agregada de un análisis completo (M1–M4 + M5 con `gpt-4o-mini`) es de 6–8 segundos, dominada por el razonamiento clínico generativo. La latencia percibida es inferior gracias al patrón asíncrono.

## H.6 Estado de despliegue

A fecha de cierre, el sistema está en producción *end-to-end* sobre el entorno de desarrollo (Windows local + DGX Spark sobre LAN), con seis módulos operativos (M1–M6) y validación funcional sobre casos de prueba controlados. **No** se ha desplegado sobre la infraestructura sanitaria del HUSLL ni validado sobre cohorte prospectiva real. Condiciones operativas para el despliegue hospitalario: integración PACS (DICOM C-STORE), autenticación Keycloak OIDC, HTTPS con certificado institucional, servicio `systemd` para arranque automático, y trazabilidad auditable conforme al Esquema Nacional de Seguridad (ENS) con logs encadenados SHA-256. Ninguno forma parte del alcance del TFG.

## H.7 Discusión: rol del prototipo

El prototipo no constituye objeto experimental. Las cifras de rendimiento se obtienen directamente de los modelos mediante protocolos estándar. Tres aportaciones técnicas: (1) verificación de que seis modelos heterogéneos coexisten en la VRAM unificada (~59 GB sobre 128 GB, sin cuantización ni *offloading*); (2) cuantificación del coste operativo por proveedor LLM en M5; (3) documentación operativa del hallazgo sobre sensibilidad al *prompt* en DermLIP v2 y su mitigación mediante *wrapping* automático en tres capas. El despliegue hospitalario completo, la validación prospectiva y la certificación CE quedan fuera del alcance.

## H.8 Evolución a M1–M11: módulos castellanos

El prototipo incorporó tres módulos castellanos adicionales (M9, M10, M4-bis) más un ensemble ponderado (M11) sobre la ontología jerárquica de la Dra. R. Taberner (4 categorías L1, 43 subcategorías L2, 367 subcategorías L3), mayoritariamente disjunta de HAM10000.

**M7 — Clasificador unificado jerárquico merged43+TTA.** PanDerm Large *fine-tuned* sobre las 43 clases L3 de la ontología unificada multi-dataset, con *weighted sampling* sobre siete datasets. Inferencia con TTA (cinco augmentaciones, promediando *softmax*). Sobre el test fijo de DermapixelAI 1.0: L1 accuracy 0,947, L2 accuracy 0,819, L3 accuracy 0,797 y BAcc L3 0,818 (+4,72 pp BAcc con TTA). Latencia *warm* ~80 ms. Tab `UNIF`.

**M9 — Dermapixel R0 (cabeza L2), clasificador L2 castellano supervisado.** Cabeza FC 1 024 → 38 sobre PanDerm Large con LoRA r = 16 en los dos últimos bloques transformer, entrenada sobre las 874 imágenes *train* del manifiesto filtrado de DermapixelAI 1.0. Las 38 clases L2 efectivas resultan de consolidar las dos variantes ortográficas L2 *queratinización*. *Checkpoint slim* ~2,3 MB (solo pesos LoRA y cabeza FC). Inferencia con TTA, ~91 ms. Tab `SPAN`.

**M10 — Seven-Point Checklist multitarea.** `MultitaskDerm7` con siete cabezas conceptuales (una por criterio del *Seven-Point Checklist*) más una cabeza binaria melanoma/no-melanoma, sobre el mismo encoder PanDerm Large + LoRA r = 16, entrenado sobre Derm7pt. Inferencia con TTA promediando *softmax* de las ocho cabezas, ~90 ms. Tab `7-POI`. Explota las anotaciones conceptuales por imagen de Derm7pt.

**M4-bis — RAG castellano sobre DermapixelAI 1.0.** Índice FAISS `IndexFlatIP` sobre 874 *embeddings* PanDerm Large L2-normalizados (*split* train+val de DermapixelAI 1.0). Devuelve los k vecinos más similares con metadatos clínicos castellanos (`ontology_l1/l2/l3`, `case_id`, `case_title`, `case_text_preview`, `rosa_verified`, `year`) y voto mayoritario por nivel. Latencia *warm* ~0,5 ms. Tab `RAG-I`.

**M11 — Ensemble ponderado por AUROC y coherencia jerárquica.** Combinación ponderada de M1, M7, M9 y M4-bis por niveles L1/L2/L3. Pesos derivados de las AUROC reportadas (M7 merged43+TTA es el más fiable: AUROC L1 = 0,873, L2 = 0,860, L3 = 0,813) más ajuste empírico por coherencia jerárquica top-down. Iteración estable `v1.7`:

| Nivel | M1 (HAM FT) | M7 (merged43+TTA) | M9 (Dermapixel R0) | M4-bis (RAG ES) |
|---|---:|---:|---:|---:|
| L1 | 1,0 | 1,0 | 0,5 | 0,3 |
| L2 | (no aplica) | 1,5 | 1,0 | 0,5 |
| L3 | (no aplica) | 1,5 | (no aplica) | 0,5 |

El *mapping* castellano-inglés exhaustivo (4 L1 + 26 L2 + 33 L3) colapsa las duplicaciones entre la ontología inglesa de M7 (merged43) y la castellana de M9/M4-bis. M11 se expone en el tab `M11` como vista de consenso, posicionada en primer lugar.

**Sistema de consenso de cinco módulos y warning de derivación urgente.** *Banner* superior con cinco *chips* (M1, M7, M9, SigLIP-LP y M10) más un sexto M4-bis (ontología del vecino top-1 castellano). Cada *chip* muestra BEN/MAL/? y un *badge* de consenso. Cuando ≥2 módulos votan melanoma con probabilidad alta, el sistema emite un *warning* explícito de derivación urgente. *Caveat* metodológico: el voto actual es por mayoría sin ponderación clínica; una iteración futura debería ponderar por confianza per-módulo.

## H.9 Estrategias de seguridad clínica: ensemble para detección de melanoma

Se evalúa un *ensemble* orientado a maximizar el *recall* de melanoma sobre HAM10000, donde el coste de un falso negativo es muy superior al de un falso positivo. Se combinan tres clasificadores independientes sobre el *split* de test de HAM10000 (N = 1 232 imágenes, 70 melanomas): M1 (PanDerm Large FT sobre HAM10000), SigLIP-Large SO400M con sondeo lineal sobre los *embeddings* de 1 152 dimensiones, y M7 (clasificador unificado sobre merged43).

*Recall top-3*: la clase melanoma aparece entre las tres primeras del ranking. FP mel: falsos positivos melanoma.

| Configuración | Acc | BAcc | AUROC | Mel rec. top-1 | Mel rec. top-3 | FP mel |
|---|---:|---:|---:|---|---|---:|
| M1 (PanDerm FT) | 0,919 | 0,813 | 0,986 | 0,714 (50/70) | 0,971 (68/70) | 36 |
| SigLIP-L LP | 0,900 | 0,776 | 0,971 | 0,614 (43/70) | 0,986 (69/70) | 31 |
| Avg M1 + SigLIP | 0,917 | 0,836 | 0,986 | 0,686 | 0,986 | 32 |
| MaxMel M1 + SigLIP | 0,909 | 0,832 | 0,985 | **0,771** | **1,000** | 50 |
| OR top-3 M1 + SigLIP | — | — | — | — | **1,000** (70/70) | 885 |

La combinación OR *top-3* de M1 y SigLIP alcanza *recall* de melanoma del 100 % (70 de 70), pero a costa de 885 falsos positivos sobre las 1 232 imágenes: la sensibilidad máxima se obtiene marcando como sospechosa cerca del 72 % del conjunto, lo que acota su utilidad a un escenario de *cribado*.

El hallazgo principal no es ese 100 %, sino la evidencia de complementariedad entre una arquitectura especializada (M1) y una generalista (SigLIP): ambas recuperan melanomas distintos. Los dos melanomas que M1 omite en *top-3* (ISIC_0024886.jpg e ISIC_0030360.jpg) sí los recupera SigLIP, y el único melanoma que SigLIP no sitúa en *top-3* sí lo recupera M1. Los errores no están completamente correlacionados —condición necesaria para que el *ensemble* aporte beneficio clínico real—. M7, evaluado sobre el subconjunto de 198 imágenes con predicción simultánea de los tres modelos (12 melanomas), no recupera melanomas adicionales sobre el par, pero aporta cobertura sobre las 43 clases ontológicas.

Tres caveats: (i) el *recall* del 100 % es *top-3*, no *top-1*; (ii) el tamaño muestral de melanomas es N = 70, con IC95 % amplios; (iii) no existe validación prospectiva sobre cohorte hospitalaria. Prueba de concepto para cribado oportunista, no sustituto del juicio clínico.

## H.10 Migración a AMQP (2026-04-13)

Desde el 13 de abril de 2026 toda comunicación entre el backend Docker (PC Windows) y el servidor de inferencia (DGX Spark) circula por RabbitMQ (`derm.inference` para *requests*, `derm.results` para *replies*). HTTP queda reservado a `/health` y depuración. Puerto 5673 abierto en dirección backend → Spark. Beneficios: resiliencia, trazabilidad auditable y desacoplamiento temporal. Lección operacional: tras cualquier cambio del backend Docker es obligatorio reconstruir sin caché (`docker compose build --no-cache && docker compose up -d`).

## H.11 Evaluación end-to-end del prototipo extendido

*Batch test* sobre 454 imágenes del conjunto de test de HAM10000 no vistas previamente. M7 supera a M1 sobre HAM10000 (BAcc 0,886 frente a BAcc 0,743), lo que justifica el peso M7 = 1,5 en L2/L3 pese a que M1 está *fine-tuned* sobre HAM. El *recall* de melanoma por módulo es 40 % para M1, 60 % para M7 y 80 % para el OR *top-3* entre M1, M7 y SigLIP-LP, lo que justifica el patrón de consenso de cinco módulos. El primer nivel jerárquico (Tumoral / Inflamatoria / Infecciosa / Genodermatosis) alcanza accuracy en [0,80, 1,00] sobre las cuatro categorías; los errores *fine-grained* se concentran en L2 y L3. TTA añade +74 ms de latencia por imagen y +4,72 pp de BAcc L3, compromiso replicado en M9 y M10. Los módulos castellanos M7–M11 mejoran las métricas críticas (*recall* de melanoma, BAcc sobre L2/L3) sin sustituir a M1–M6.
