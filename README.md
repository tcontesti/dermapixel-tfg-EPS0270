# Dermapixel — TFG EPS0270

Repositorio reproducible y material complementario del Trabajo de Fin de Grado **«Dermapixel: Evaluación de modelos fundacionales para Dermatología Clínica»** (código oficial EPS-UIB: **EPS0270**), de Antonio Contestí Coll, Grau d'Enginyeria Informàtica, Escola Politècnica Superior de la Universitat de les Illes Balears, curso 2025-2026, bajo la dirección del Dr. Javier Varona Gómez (UIB) y con colaboración clínica de la Dra. Rosa Taberner (Hospital Universitari Son Llàtzer).

- 🌐 **Portada navegable (GitHub Pages):** https://tcontesti.github.io/dermapixel-tfg-EPS0270/
- 📖 **Memoria (PDF, 97 págs):** [`MemoriaTFG.pdf`](MemoriaTFG.pdf)
- 📚 **Wiki del proyecto:** https://github.com/tcontesti/dermapixel-tfg-EPS0270/wiki
- 🩺 **Prototipo en producción:** https://dermapixel.eu

## Qué es

El trabajo aporta tres contribuciones cerradas y reproducibles:

1. **Evaluación comparativa homogénea e independiente** de hasta catorce modelos fundacionales con pesos públicos —con PanDerm (codificador visual) y DermLIP (vision-language) como referencia— sobre un conjunto común de tareas (clasificación, segmentación, recuperación de casos y clasificación *zero-shot* guiada por texto) y de datasets públicos armonizados mediante una [ontología clínica de tres niveles (L1/L2/L3)](ontology/README.md). Incluye la auditoría del solapamiento (*leakage*) entre el corpus de preentrenamiento y los conjuntos de evaluación.
2. **DermapixelAI 1.0**, un dataset clínico en castellano con texto narrativo asociado a cada caso (1 089 imágenes, 672 casos con imagen), entregado con documentación formal tipo *datasheet* y licencia CC BY-NC-SA 4.0.
3. **Dermapixel R0**, una adaptación propia que demuestra que un modelo fundacional puede especializarse al castellano con recursos modestos (rama supervisada LoRA L2 desplegada como módulo M9; rama contrastiva SpanDerm-CLIP, estudiada y documentada pero **no** desplegada por no generalizar en uso real).

Sobre estas piezas se construye el prototipo clínico **DermApIxel**, en producción *end-to-end* sobre [dermapixel.eu](https://dermapixel.eu): once módulos (M1–M11) más el buscador M4-bis, asistente conversacional multicanal (web, WhatsApp y voz) y selector de tres modos de diagnóstico. El proyecto fue distinguido con la **Beca de Innovación e Inteligencia Artificial de la AEDV** (10 000 €, 53.º Congreso Nacional de Dermatología y Venereología, Maspalomas, mayo de 2026).

## Estructura del repositorio

```
dermapixel-tfg-EPS0270/
├── MemoriaTFG.pdf      Memoria del TFG (v7, 97 páginas con apéndices)
├── docs/               Portada GitHub Pages (landing) + capturas del prototipo
├── repro/              Mapa de tareas, recetas de entrenamiento, semillas, normalizaciones y checkpoints
├── tables/             Tablas detalladas de resultados, IC bootstrap y calibración de melanoma (HAM10000)
├── datasheet/          Datasheet y documento de entrega del dataset DermapixelAI 1.0
├── ontology/           Ontología clínica jerárquica L1/L2/L3 (Dra. Taberner) + mapeo de datasets
├── ablations/          Pipeline y caracterización del dataset, y ablaciones complementarias (incl. rama contrastiva)
├── sae/                Sparse Autoencoders y diccionario de conceptos clínicos
├── llm/                Comparación extendida de modelos de lenguaje multimodales
├── prototype/          Descripción del prototipo DermApIxel (módulos M1–M11)
├── papers/             Preprints de los modelos de referencia (PanDerm, Derm1M, DermFM-Zero)
├── code/               Pipelines experimentales (en organización)
├── ai-statement/       Declaración de uso de herramientas de IA (Apéndice B de la memoria)
├── LICENSE             CC BY-NC 4.0 (material original del autor)
└── CITATION.cff        Cómo citar el TFG
```

Esta estructura reproduce la definida en el **Apéndice A** de la memoria («Material complementario y repositorio reproducible»). El repositorio está en organización y ampliación continua tras el depósito.

## Cómo navegar este material

- **Leer la memoria:** abre [`MemoriaTFG.pdf`](MemoriaTFG.pdf).
- **Visión general visual:** la [portada navegable](https://tcontesti.github.io/dermapixel-tfg-EPS0270/).
- **Consulta temática:** la [wiki](https://github.com/tcontesti/dermapixel-tfg-EPS0270/wiki), con páginas por tema (modelos, datasets y ontología, resultados, Dermapixel R0, prototipo, reproducibilidad…).
- **Verificar una cifra:** cada cifra del cuerpo de la memoria se desagrega en [`tables/`](tables/) (con intervalos de confianza *bootstrap* y calibración).
- **Reproducir un experimento:** consulta [`repro/`](repro/) (mapa de tareas, recetas, semillas y *checkpoints*).

## Reproducibilidad y material no redistribuido

Los experimentos se ejecutaron sobre un servidor GPU on-prem (NVIDIA DGX Spark, chip GB10, arquitectura aarch64). El repositorio publica el material original del autor (documentación, tablas, ontología, *datasheet*, descripción del prototipo). **No se redistribuyen**, por motivos de licencia y espacio:

- Pesos preentrenados de los modelos base (PanDerm, DermLIP, SAM2/MedSAM2, SigLIP, BiomedCLIP, DINOv2, MedGemma): disponibles a través de sus publicaciones originales.
- Datasets de terceros (HAM10000, BCN20000, PAD-UFES-20, DDI, Derm7pt, Dermnet, MSKCC, HIBA, ISIC2017/2018, PH2, Fitzpatrick17k, SkinCon, Derm1M): disponibles a través de sus repositorios oficiales según sus licencias.
- DermapixelAI 1.0: disponible bajo solicitud académica / depósito formal (Zenodo, pendiente de DOI) bajo CC BY-NC-SA 4.0.

## Licencia

El documento académico (memoria) y el material original del autor se distribuyen bajo **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** — ver [`LICENSE`](LICENSE). El dataset **DermapixelAI 1.0** se distribuye bajo **CC BY-NC-SA 4.0** (ver [`datasheet/`](datasheet/README.md)). El material de terceros (modelos base, datasets) conserva su licencia original; este trabajo solo lo referencia y evalúa, no lo redistribuye.

## Cómo citar

```bibtex
@thesis{contesti2026dermapixel,
  author       = {Contestí Coll, Antonio},
  title        = {Dermapixel: Evaluación de modelos fundacionales para Dermatología Clínica},
  type         = {Trabajo de Fin de Grado},
  institution  = {Escola Politècnica Superior, Universitat de les Illes Balears},
  address      = {Palma de Mallorca},
  year         = {2026},
  note         = {Código TFG EPS0270. Tutor: Dr. Javier Varona Gómez. Colaboración clínica: Dra. Rosa Taberner (Hospital Universitari Son Llàtzer)}
}
```

Ver también [`CITATION.cff`](CITATION.cff).

## Contacto

- **Estudiante:** Antonio Contestí Coll — antonio.contesti1@estudiant.uib.cat
- **Tutor:** Dr. Javier Varona Gómez — xavi.varona@uib.es
