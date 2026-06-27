# code/ — Pipelines experimentales

> **En organización.** El repositorio está en ampliación continua tras el depósito de la memoria. Esta carpeta alojará, una vez revisado y depurado para publicación, el código de los *pipelines* experimentales:
>
> - *Linear probing* sobre los codificadores fundacionales evaluados.
> - *Fine-tuning* supervisado (cabezas L1/L2/L3, configuración *merged43* + TTA).
> - Segmentación promptable con LoRA.
> - Evaluación *zero-shot* guiada por texto.

Mientras tanto, el **mapa de tareas, las recetas de entrenamiento, las semillas, las normalizaciones y los *checkpoints*** necesarios para reproducir las cifras del capítulo de Resultados están documentados en [`repro/`](../repro/README.md).

Los pesos preentrenados de los modelos base (PanDerm, DermLIP, SAM2/MedSAM2, SigLIP, BiomedCLIP, DINOv2, MedGemma) y los datasets de terceros **no se redistribuyen**: se obtienen a través de sus publicaciones y repositorios oficiales según su licencia.
