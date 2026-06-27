# Anexo E — Documento de entrega del dataset DermapixelAI 1.0

> Material complementario del TFG EPS0270. Corresponde al antiguo Anexo E de la memoria. Documento de entrega formal del dataset DermapixelAI 1.0, complementario al pipeline interno de construcción ([`ablations/pipeline-dataset.md`](../ablations/pipeline-dataset.md)). Sigue el formato *Datasheet for Datasets*.

## E.1 Identidad del dataset

- **Nombre:** DermapixelAI.
- **Versión:** 1.0 (primera versión estable y publicable).
- **Fecha de cierre de la versión 1.0:** mayo de 2026.
- **Identificador persistente (DOI):** pendiente de asignación en el momento de la publicación efectiva del dataset. El depósito formal se realiza en **Zenodo** (https://zenodo.org) bajo licencia CC BY-NC-SA 4.0 con DOI permanente y citable, complementado con el repositorio de la UIB cuando éste habilite el alojamiento de datasets binarios de tamaño equivalente.
- **Cita corta:** *Taberner, R. y Contestí Coll, A. (2026). DermapixelAI 1.0: dataset dermatológico clínico en castellano. Versión 1.0.*

## E.2 Procedencia y autoría

El dataset se construye a partir del archivo del blog Dermapixel (https://dermapixel.com), mantenido desde hace más de quince años por la Dra. R. Taberner, dermatóloga clínica del Hospital Universitario Son Llàtzer. El archivo original recoge casos clínicos comentados con fines docentes.

La construcción del dataset, el pipeline de extracción y estructuración, la definición de la ontología jerárquica L1/L2/L3 y el particionado han sido desarrollados por Antonio Contestí Coll en el marco del presente TFG. El uso del archivo Dermapixel como materia prima se ha realizado bajo acuerdo previo con la Dra. R. Taberner, formalizado por escrito. La revisión experta de las etiquetas diagnósticas, del mapeo a la ontología jerárquica y de los campos de validación (`label_source`, `diagnosis_source`, `rosa_verified`) ha sido aportada por la Dra. Taberner.

El dataset se publica con autoría compartida Taberner & Contestí Coll, en este orden, dado que el archivo Dermapixel del que deriva es propiedad intelectual de la Dra. Taberner y constituye el material original sin el cual el dataset no existiría.

## E.3 Licencia de uso

DermapixelAI 1.0 se publica bajo **Creative Commons Reconocimiento–NoComercial–CompartirIgual 4.0 Internacional (CC BY-NC-SA 4.0)**. Texto canónico: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es

- **Reconocimiento (BY).** Cualquier uso exige la cita explícita del dataset original.
- **No comercial (NC).** No puede emplearse con fines primordialmente comerciales. La investigación académica, el uso docente y el desarrollo de prototipos en fase no comercial quedan dentro del alcance.
- **Compartir Igual (SA).** Cualquier obra derivada deberá distribuirse bajo la misma licencia o una compatible.

## E.4 Cita recomendada

> Taberner, R. y Contestí Coll, A. (2026). *DermapixelAI 1.0: dataset dermatológico clínico en castellano construido sobre el archivo Dermapixel*. Versión 1.0. Licencia CC BY-NC-SA 4.0.

Entrada BibTeX equivalente (pendiente de completar con el DOI tras la publicación):

```bibtex
@dataset{dermapixelai2026,
  author    = {Taberner, Rosa and Contestí Coll, Antonio},
  title     = {DermapixelAI 1.0: dataset dermatológico
               clínico en castellano construido sobre
               el archivo Dermapixel},
  year      = {2026},
  version   = {1.0},
  license   = {CC BY-NC-SA 4.0},
  doi       = {pendiente},
  url       = {pendiente},
}
```

## E.5 Disponibilidad y distribución

La publicación efectiva en un repositorio externo está pendiente de decisión en la fecha de cierre del trabajo. Estructura del paquete previsto:

- `dataset.csv`: fichero maestro con una fila por imagen (1 089 filas) y campos de modalidad, ruta, hash MD5, diagnóstico L1/L2/L3, campos de validación experta, identificador de caso, texto narrativo y partición del split.
- `metadata/cases.csv`: una fila por caso clínico, incluidos los casos sin imagen (`num_images_in_dataset = 0`) por trazabilidad.
- `metadata/ontology.csv`: vocabulario jerárquico L1/L2/L3 con códigos SNOMED CT y CIE-10 asociados a cada entrada L3.
- `metadata/excluded_images.csv`: registro de imágenes descartadas, con razón documentada.
- `metadata/audit_log.txt`: traza de auditoría de las modificaciones.
- `splits/train.csv`, `splits/val.csv`, `splits/test.csv`: particiones canónicas case-aware.
- `images/`: imágenes organizadas por modalidad (`clinical/`, `dermoscopy/`, `histology/`, `ultrasound/`, `wood_lamp/`).
- `README.md`: documento plano que reproduce el contenido de este anexo.
- `LICENSE.txt`: texto canónico de CC BY-NC-SA 4.0 con cláusulas de privacidad del paciente y provenance clínico.
- `CITATION.cff`: ficha de citación en Citation File Format 1.2.0.
- `MANIFEST.sha256`: manifiesto global del paquete con hash SHA-256 de cada archivo, verificable con `sha256sum -c MANIFEST.sha256`.
- `enriched_captions/`: dataset enriquecido para la rama contrastiva de Dermapixel R0 con captions sintéticas multi-perspectiva generadas por LLM, y `spanderm_clip_trilogy/` con los resultados experimentales completos del experimento dual-encoder.

La integridad reproducible se verifica en dos niveles: (i) el manifiesto interno de cada partición o subconjunto experimental, fijado en el *checkpoint* correspondiente; y (ii) el manifiesto global del paquete (`MANIFEST.sha256`), que cubre los 1 174 archivos del *bundle*. Ambos niveles son independientes y deterministas.

## E.6 Datasheet for Datasets

### Motivación

**¿Para qué se creó el dataset?** Para servir de material de investigación académica sobre dermatología clínica en castellano, con foco en el aprendizaje multimodal (imagen + texto clínico) y como base estable para trabajos posteriores. No se creó como benchmark de evaluación del TFG sino como contribución a la comunidad investigadora.

**¿Quién lo financió?** Sin financiación específica, en el marco del TFG (EPS, UIB) y de la colaboración con la Dra. R. Taberner sobre la modernización del archivo Dermapixel.

### Composición

**¿Qué representa cada instancia?** Una imagen dermatológica asociada a un caso clínico publicado en el archivo Dermapixel.

**¿Cuántas instancias hay?** 1 089 imágenes vinculadas a 672 casos con imagen —de los cuales 669 se retienen en el *split* de producción— sobre 698 casos catalogados en total.

**¿Es el dataset un subconjunto de uno mayor?** Se construye a partir del archivo público del blog Dermapixel, de composición más amplia (publicaciones y elementos no clínicos: banners, ilustraciones, fotografías de eventos) que se filtran durante la construcción.

**¿Qué información acompaña a cada imagen?** Modalidad, diagnóstico L1/L2/L3, identificador de caso, texto narrativo asociado en castellano, campos de validación experta y particionado en train/val/test.

**¿Hay etiquetas o ground-truth?** Sí. El diagnóstico L3 está mapeado a la ontología jerárquica con revisión experta. `rosa_verified` marca las imágenes adicionalmente validadas visualmente por la Dra. Taberner.

**¿Hay información que se haya eliminado?** Se han eliminado las imágenes asociadas a la pista de solución de los casos del blog (susceptibles de inducir *leakage*) y las imágenes no dermatológicas detectadas durante la auditoría de modalidad.

### Proceso de recogida / Preprocesamiento

Detalle del pipeline en [`ablations/pipeline-dataset.md`](../ablations/pipeline-dataset.md): hash MD5 por imagen para integridad; deduplicación exacta; mapeo ontológico con revisión experta; particionado case-aware.

### Usos previstos

**¿Para qué tareas se ha diseñado?** Investigación académica sobre dermatología clínica en castellano, en particular escenarios que requieren material multimodal imagen-texto en español, la armonización ontológica entre datasets heterogéneos y la docencia universitaria sobre análisis de imagen médica.

**¿Para qué tareas *no* debe usarse?** No constituye un dispositivo médico ni puede emplearse como base de un sistema de diagnóstico autónomo de uso clínico directo. Cualquier aplicación clínica derivada debe seguir los procesos regulatorios correspondientes y no puede sustituir al juicio del especialista.

### Distribución

CC BY-NC-SA 4.0. Plataforma pendiente de decisión en la fecha de cierre del trabajo.

### Mantenimiento

Versionado semántico. Política de errata reportable mediante el contacto de la sección E.9.

## E.7 Uso responsable y limitaciones clínicas

- **No es un dispositivo médico.** Material de investigación académica. Cualquier prototipo derivado que aspire a uso clínico real debe seguir los procedimientos regulatorios aplicables (marcado CE, evaluación clínica, etc.).
- **No para uso diagnóstico autónomo.** Las predicciones de modelos entrenados o evaluados sobre el dataset no pueden sustituir al juicio del dermatólogo.
- **Limitaciones de cobertura.** Desbalance de modalidad, cola larga diagnóstica, cobertura ontológica incompleta y cobertura parcial por subcategoría en los splits (documentadas en [`ablations/pipeline-dataset.md`](../ablations/pipeline-dataset.md)). Cualquier publicación derivada debe declararlas al reportar cifras de rendimiento.
- **Generalización geográfica.** Proviene del archivo Dermapixel, alimentado por casos atendidos en el HUSLL (Mallorca, España). La validez sobre poblaciones de fototipo, geografía o circuito asistencial distintos requiere verificación independiente.

## E.8 Privacidad y consentimiento

El archivo Dermapixel es un blog público dirigido a la formación de profesionales sanitarios. Las imágenes proceden de casos atendidos por la Dra. R. Taberner y se publicaron originalmente con consentimiento implícito en el marco del compromiso docente del blog, sobre material clínico anonimizado para uso formativo. El dataset hereda esta naturaleza de material docente público.

El dataset no contiene metadatos identificativos directos de pacientes (nombres, fechas de nacimiento, números de historia clínica, ubicación geográfica concreta). Las imágenes están recortadas a la lesión cuando procede; las fotografías macroscópicas preservan únicamente la región anatómica relevante.

Cualquier reusuario se compromete, al aceptar la licencia, a no intentar reidentificar a los pacientes ni a usar el material con finalidad distinta a la investigación académica o docente.

## E.9 Mantenimiento y contacto

**Política de versionado.** Numeración semántica MAJOR.MINOR. Cambios en el conjunto de imágenes o en la ontología canónica incrementan MAJOR. Correcciones de etiquetado, normalización de campos o ampliaciones de validación experta incrementan MINOR.

**Reporte de errata.** Cualquier discrepancia entre etiquetas y diagnóstico correcto, o cualquier problema de integridad, puede reportarse al contacto. Las erratas confirmadas se incorporan en la siguiente versión MINOR.

**Contacto.**

- Propietaria del archivo original y revisión experta clínica: Dra. Rosa Taberner
- Autor técnico del dataset: Antonio Contestí Coll
- Afiliación principal: Hospital Universitario Son Llàtzer y Universitat de les Illes Balears (Escola Politècnica Superior)
- Canal de contacto preferente: mediante el repositorio oficial del dataset una vez publicado; mientras tanto, a través del autor técnico.
