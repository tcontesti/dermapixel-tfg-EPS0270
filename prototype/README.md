# Prototipo DermApIxel — del banco de pruebas al asistente clínico

> Material complementario del TFG EPS0270. Describe el prototipo **DermApIxel**,
> sistema integrado de apoyo al diagnóstico dermatológico que **materializa la
> integración técnica** de los modelos evaluados en el trabajo. **No constituye
> objeto experimental** del estudio: las cifras de la defensa empírica se
> obtienen directamente de los modelos mediante protocolos estándar; el prototipo
> aporta la *traslación* de esos modelos a una interfaz clínica multicanal.
>
> A fecha de cierre el sistema está en **producción *end-to-end* en
> [dermapixel.eu](https://dermapixel.eu)**, con la Dra. R. Taberner como
> evaluadora clínica de referencia. **No** está validado sobre cohorte
> prospectiva real ni integrado en la infraestructura de un centro hospitalario.

El prototipo integra once módulos de inferencia (M1–M11 más M4-bis), un asistente
conversacional multicanal (web, WhatsApp y voz) y las decisiones de seguridad
clínica que un *benchmark* no obliga a tomar pero un sistema en producción sí.

---

## 1. De la evaluación al producto: qué se traslada

El prototipo no es la demostración de un único modelo, sino la integración de
**varias líneas del trabajo**, cada una en el papel para el que el banco de
pruebas la encontró útil.

| Estudiado en el trabajo | Materialización en el prototipo | Estado |
|---|---|---|
| Clasificación supervisada PanDerm | M1 (HAM10000) / M7 (jerárquico merged43+TTA) | Desplegado, visible |
| Dermapixel R0, head L2 castellana | M9 (PanDerm + LoRA r=16) | Desplegado, visible |
| Segmentación promptable | Segmentación + corrección experta | Desplegado, visible |
| Interpretabilidad por SAE + SkinCon | M3 (32 conceptos por imagen) | Desplegado, visible |
| *Seven-Point Checklist* multitarea | M10 + puntuación de Argenziano (capa *backend*) | Desplegado, visible |
| Zero-shot multimodal DermLIP | M6 (clase abierta) + glosario ES→EN | Desplegado, visible |
| Recuperación visual densa | M4 (Derm1M) / M4-bis (DermapixelAI ES) | Desplegado, visible |
| Comparación con LLM multimodales | Razonamiento clínico estructurado (M5) | Desplegado, visible |
| Ensemble por complementariedad | M11 + *banner* de consenso de 5 módulos | Desplegado, visible |
| Ontología jerárquica L1/L2/L3 | Coherencia de niveles + traducción EN→ES | Desplegado, interno |
| Texto clínico como contexto | RAG conversacional sobre el archivo Dermapixel | Desplegado, visible |
| Dermapixel R0, **rama contrastiva** (SpanDerm-CLIP) | Búsqueda textual ES | **No desplegado** |

Ninguno sustituye al juicio clínico: el prototipo se diseña como **apoyo**, no
como decisor.

---

## 2. Arquitectura del sistema desplegado

El *backend* se organiza como **microservicios en contenedores** (Docker
Compose) repartidos en dos redes: una expuesta al exterior (proxy inverso y
salida a APIs externas) y otra interna sin acceso directo a Internet, por mínima
exposición. Las bases de datos están separadas por servicio (identidad y datos
clínicos), sin enlace directo entre ellas; la relación usuario–estudio se
comprueba en la aplicación, para acotar el alcance de un posible incidente sobre
datos de salud.

**Separación cliente / inferencia GPU.** La inferencia pesada (modelos M1–M11)
corre en un servidor GPU propio (NVIDIA DGX Spark, chip GB10), separado del
*backend* y accesible solo por una red privada. El *backend* no llama a la GPU
directamente, sino a través de un servicio proxy que la convierte en una
dependencia tolerante a fallos (reintentos, *circuit breaker* y caché): una caída
momentánea de la GPU degrada el servicio con elegancia en lugar de propagar el
error al clínico. La comunicación entre componentes es **asíncrona por
mensajería (RabbitMQ)**, sin tráfico HTTP directo para la inferencia.

**Persistencia y trazabilidad.** Cada estudio diagnóstico se persiste con su
imagen (cifrada en reposo), un *thumbnail* pre-renderizado, el canal de origen
(web / WhatsApp / voz), la salida cruda de la inferencia, el razonamiento
generado y el tipo de imagen detectado. Toda acción sensible queda auditada con
la IP *hasheada* (SHA-256), y `auth-svc` mantiene una bitácora encadenada
criptográficamente orientada al Esquema Nacional de Seguridad (ENS).

---

## 3. El pipeline diagnóstico integrado

La vista de análisis expone el resultado de los módulos en pestañas y, sobre
ellas, un **banner de consenso** que resume la lectura de malignidad de cinco
clasificadores independientes (M1, M7, M9, sondeo lineal SigLIP y M10) más un
sexto indicador con la ontología del vecino *top-1* castellano (M4-bis). Cuando
**dos o más módulos votan melanoma** con probabilidad alta, el sistema emite un
aviso explícito de derivación urgente (patrón de segunda opinión clínica).

### Tres modos de interpretación, una misma red de seguridad

El clínico elige *cuánto* quiere que la IA interprete el caso, mediante un
selector con tres modos que comparten el mismo análisis de base:

- **Determinista** — el veredicto procede únicamente de los clasificadores (el
  ensemble M11); el módulo de razonamiento no interviene. Reproducible y siempre
  disponible (es el respaldo si el razonamiento falla).
- **Híbrido (IA-asistido)** *(por defecto)* — parte del veredicto determinista y
  lo *refina* con las señales del razonamiento clínico estructurado (M5): solo
  cambia el diagnóstico principal si el razonamiento propone otro **con alta
  confianza**.
- **Interpretación IA máxima** — el razonamiento de M5 lleva la voz cantante: el
  diagnóstico principal se sustituye por el que propone y el pipeline de modelos
  queda como contexto; si M5 no ofrece un diagnóstico claro, vuelve al
  determinista.

Sobre los tres rige una **red de seguridad innegociable: la urgencia nunca baja
por acción de la IA** —el razonamiento puede *subir* el nivel de alarma ante una
lesión sospechosa, nunca rebajarlo— y el veredicto determinista se conserva
siempre como base auditable.

### La Lista de los 7 puntos (M10)

M10 detecta por imagen los siete criterios dermatoscópicos del *Seven-Point
Checklist* más una *head* binaria de melanoma. La **puntuación ponderada de
Argenziano** (criterios mayores valen 2, menores 1, melanoma a partir de 3) la
calcula la capa clínica del *backend*, no la red, de modo que el esquema de
puntuación se puede auditar y ajustar sin reentrenar el modelo. El panel **solo
se activa sobre imagen dermatoscópica** (la Lista no aplica a fotografía
clínica), con un *gate* *fail-closed* y confirmación experta de la modalidad.

### Segmentación con corrección experta

La máscara automática de la lesión se muestra superpuesta y el clínico la corrige
si procede. La capa de anotación registra el *origen* de cada máscara aprobada
(aceptada, corregida o dibujada desde cero), lo que permite distinguir, de cara
al reentrenamiento, qué casos requirieron intervención humana.

### Zero-shot abierto y el idioma del espacio latente

M6 (DermLIP) compara una imagen contra una lista arbitraria de descripciones,
con apertura a clases no previstas. Su integración documenta un hallazgo
operativo: **la sensibilidad de un modelo contrastivo al idioma de la consulta**.
DermLIP se entrenó con texto en inglés; al recibir nombres de diagnóstico en
castellano sin traducir, el *ranking* se degradaba. La corrección —un glosario
clínico ES→EN marcado para revisión de la Dra. Taberner— restauró el
comportamiento esperado.

---

## 4. Los módulos de inteligencia artificial

| Módulo | Modelo subyacente | Latencia *warm* | Función |
|---|---|---:|---|
| M1 | PanDerm Large FT (HAM10000, TTA) | ~315 ms | Clasificación cerrada 7 clases |
| M2 | SAM2.1-Large (decoder FT, ISIC2018) + corrección experta | — | Segmentación binaria de lesión |
| M3 | SAE 1 024 → 16 384 + SkinCon | — | 32 conceptos clínicos por imagen |
| M4 | FAISS sobre Derm1M (≈421 k vec.) | ~150 ms | Recuperación visual densa *top-5* |
| M5 | LLM multimodal (varios proveedores) | 6–300 s | Razonamiento clínico estructurado |
| M6 | DermLIP v2 (PubMedBERT) | ~20 ms | Clasificación abierta zero-shot |
| M7 | PanDerm Large FT merged43 + TTA | ~80 ms | Clasificador unificado jerárquico L1/L2/L3 |
| M9 | PanDerm Large + LoRA r=16 (Dermapixel R0) | ~91 ms | Clasificador L2 castellano supervisado |
| M10 | MultitaskDerm7 (PanDerm + LoRA) | ~90 ms | *Seven-Point Checklist* + melanoma |
| M4-bis | FAISS `IndexFlatIP` sobre 874 emb. PanDerm Large (DermapixelAI 1.0) | ~0,5 ms | Recuperación visual en castellano |
| M11 | Ensemble ponderado por AUROC + coherencia jerárquica | — | Consenso L1/L2/L3 |

Notas destacadas:

- **M7 (merged43+TTA)** sobre el test fijo de DermapixelAI 1.0: L1 accuracy
  0,947, L2 0,819, **L3 0,797**, BAcc L3 0,818 (+4,72 pp BAcc con TTA).
- **M9 (Dermapixel R0)** entrena solo los pesos LoRA y la *head* FC (≈524 k
  parámetros); *checkpoint slim* ~2,3 MB.
- **M4-bis** indexa los *embeddings* **de imagen** de PanDerm Large y recupera
  por similitud **imagen→imagen** con metadatos clínicos en castellano.
- **M11** combina M1, M7, M9 y M4-bis por niveles (pesos derivados de las AUROC;
  M7 es el más fiable). El *banner* de consenso de cinco módulos dispara el aviso
  de derivación urgente cuando ≥2 votan melanoma. *Caveat*: el voto actual es por
  mayoría, sin ponderación clínica calibrada con datos prospectivos.

---

## 5. El asistente conversacional multicanal

La aportación que más distingue al prototipo de un clasificador de imágenes es el
**asistente conversacional anclado al archivo Dermapixel** (la segunda vía del
trabajo: el texto clínico como contexto).

- **RAG sobre el archivo del blog** (indexado vectorialmente): busca los textos
  más relevantes —combinando coincidencia de palabras y similitud semántica— y,
  antes de responder, comprueba si lo recuperado basta para hacerlo, para
  responder con cautela o para declinar. Cada respuesta incluye **citas** a las
  entradas del blog que la sustentan.
- **Tres canales con paridad de veredicto** (la misma entrada produce el mismo
  veredicto determinista en los tres): **web** (con citas al archivo),
  **WhatsApp** (análisis de imagen sobre cuenta vinculada; el número se almacena
  *hasheado*) y **voz** (conversación en tiempo real; durante ella el asistente
  puede pedir una foto, que el usuario aporta por arrastre o, desde otro
  dispositivo, escaneando un código QR que abre una página de captura).
- **Herramientas externas (MCP)**: el asistente puede consultar PubMed, la
  información de medicamentos de la AEMPS y la ontología clínica, bajo lista
  blanca (solo herramientas explícitamente autorizadas).
- **Guardarraíles clínicos**: validación de *prompt* y respuesta, reglas clínicas
  acordadas con la Dra. Taberner, y un disparador de urgencia («llame al 112»)
  que nunca se suprime. Una inyección en el contexto **no puede** elevar la
  urgencia del veredicto determinista: el texto libre del usuario no altera la
  salida de los clasificadores.

---

## 6. Ramas estudiadas no desplegadas

La **rama contrastiva texto→imagen de Dermapixel R0 (SpanDerm-CLIP)**, concebida
como buscador textual en castellano, **no se promovió a producción**. Su métrica
de cabecera más favorable corresponde a un régimen concreto (partición de
validación, descripciones clínicas largas, un *checkpoint* específico) que **no
generaliza**: en uso real —consulta corta del usuario, conjunto de test no visto,
archivo completo— la recuperación cae a un nivel cercano al azar. La partición es
limpia (disjunta por caso e imagen), lo que descarta la contaminación como causa.
Queda demostrada la **viabilidad técnica** del espacio dual texto-imagen, pero
**no** una capacidad de recuperación útil en régimen real. La búsqueda visual en
producción la cubre **M4-bis** (imagen→imagen, PanDerm Large + FAISS).

---

## 7. Estado de despliegue y reconocimiento

A fecha de cierre, el sistema está en **producción *end-to-end* en
[dermapixel.eu](https://dermapixel.eu)**, con los módulos M1–M11 + M4-bis
operativos, el asistente conversacional en sus tres canales y la Dra. R. Taberner
como evaluadora clínica de referencia. **No** está validado sobre cohorte
prospectiva real ni integrado en la infraestructura de un centro hospitalario;
quedan fuera del alcance del TFG la integración PACS, la validación prospectiva
con aprobación de un comité ético y la certificación como producto sanitario.

El proyecto fue distinguido con la **Beca de Innovación e Inteligencia Artificial
de la Academia Española de Dermatología y Venereología (AEDV)** —dotada con
10 000 € para su puesta en producción— en el 53.º Congreso Nacional de
Dermatología y Venereología (Maspalomas, mayo de 2026).

---

## 8. Estrategias de seguridad clínica: ensemble para detección de melanoma

Exploración aplicada orientada a maximizar el *recall* de melanoma sobre HAM10000
(donde un falso negativo cuesta mucho más que un falso positivo). Se combinan
tres clasificadores independientes sobre el *split* de test de HAM10000
(N = 1 232 imágenes, 70 melanomas): M1 (PanDerm Large FT), SigLIP-Large SO400M con
sondeo lineal, y M7 (clasificador unificado merged43).

*Recall top-3*: la clase melanoma aparece entre las tres primeras del ranking.

| Configuración | Acc | BAcc | AUROC | Mel rec. top-1 | Mel rec. top-3 | FP mel |
|---|---:|---:|---:|---|---|---:|
| M1 (PanDerm FT) | 0,919 | 0,813 | 0,986 | 0,714 (50/70) | 0,971 (68/70) | 36 |
| SigLIP-L LP | 0,900 | 0,776 | 0,971 | 0,614 (43/70) | 0,986 (69/70) | 31 |
| Avg M1 + SigLIP | 0,917 | 0,836 | 0,986 | 0,686 | 0,986 | 32 |
| MaxMel M1 + SigLIP | 0,909 | 0,832 | 0,985 | **0,771** | **1,000** | 50 |
| OR top-3 M1 + SigLIP | — | — | — | — | **1,000** (70/70) | 885 |

La combinación OR *top-3* alcanza *recall* de melanoma del 100 % (70/70), pero a
costa de 885 falsos positivos sobre 1 232 imágenes: la sensibilidad máxima marca
como sospechosa cerca del 72 % del conjunto, lo que acota su utilidad a un
escenario de **cribado**. El hallazgo principal no es ese 100 %, sino la
**complementariedad** entre una arquitectura especializada (M1) y una generalista
(SigLIP): recuperan melanomas distintos (errores no completamente correlacionados,
condición necesaria para que el ensemble aporte beneficio real). *Caveats*: el
*recall* del 100 % es *top-3*, no *top-1*; N = 70 melanomas con IC95 % amplios; y
no hay validación prospectiva. Prueba de concepto para cribado, no sustituto del
juicio clínico.

---

## 9. Evaluación *end-to-end* del prototipo extendido

*Batch test* sobre 454 imágenes del test de HAM10000 no vistas previamente. **M7
supera a M1** sobre HAM10000 (BAcc 0,886 frente a 0,743), lo que justifica el peso
M7 = 1,5 en L2/L3 pese a que M1 está *fine-tuned* sobre HAM. El *recall* de
melanoma por módulo es 40 % (M1), 60 % (M7) y 80 % (OR *top-3* entre M1, M7 y
SigLIP-LP), lo que justifica el patrón de consenso de cinco módulos. El primer
nivel jerárquico (Tumoral / Inflamatoria / Infecciosa / Genodermatosis) alcanza
accuracy en [0,80, 1,00]; los errores *fine-grained* se concentran en L2 y L3. TTA
añade +74 ms de latencia por imagen y +4,72 pp de BAcc L3.

---

> El detalle completo del prototipo (capítulo «El prototipo DermApIxel» de la
> memoria) figura en `MemoriaTFG.pdf`. Las cifras de este documento coinciden con
> las de la memoria; no se introduce ningún resultado adicional.
