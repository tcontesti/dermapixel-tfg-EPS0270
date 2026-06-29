# Ground-truth dermatoscópico de la Dra. Taberner (verdad SAE)

> Material complementario del TFG EPS0270. **Definición de la verdad de terreno conceptual** para los *Sparse Autoencoders* (módulo M3, ver [README del anexo F](README.md)). Fuente: `DERMATOSCOPIA-ESTRUCTURAS.xlsx`, definido por la **Dra. R. Taberner (HUSLL)**. Sustituye/precisa la propuesta de 15 conceptos del apartado F.5 con una matriz dermatoscópica operativa.

## Alcance

Esta verdad de terreno aplica **solo a enfermedades dermatoscópicas** (las que se diagnostican por estructuras dermatoscópicas). Quedan fuera los diagnósticos clínicos no dermatoscópicos. Las **15 enfermedades** cubiertas son:

NEVUS, NEVUS ACRAL, MELANOMA, LÉNTIGO MALIGNO, CBC, BOWEN, Q. ACTÍNICA, CEC, QS, DERMATOFIBROMA, ANGIOMA, ANGIOQUERATOMA, ESCABIOSIS, LEISHMANIA, HIPERPLASIA SEBÁCEA.

## Tres capas de anotación

Cada lesión se describe con tres tipos de atributo, con semántica distinta:

1. **Asimetría** — binaria `SÍ/NO` (un único atributo global de la lesión).

2. **Color** (11) — cada color `SÍ/NO`, pero **detectado automáticamente por el algoritmo** (computable del píxel, no anotación experta): negro, marrón claro, marrón oscuro, blanco, gris, rojo, amarillo, naranja, violáceo, azul, policromo.

3. **Estructuras** (37) — estructuras dermatoscópicas, con dos regímenes de anotación:

   - **Típico / atípico** (las discriminantes): la mera presencia no es patológica; lo que discrimina es si la estructura es *típica* (benigna) o *atípica* (signo de alarma). La anotación **debe** permitir marcar el matiz, no colapsar a presente/ausente.

   - **Presencia** (el resto): la `X` de la matriz indica qué estructuras **puede presentar** cada enfermedad (conjunto de conceptos candidatos por diagnóstico).


## Estructuras con anotación TÍPICO/ATÍPICO (discriminantes)

| Estructura | Enfermedades que la pueden presentar |
|---|---|
| RETÍCULO PIGMENTADO | NEVUS, NEVUS ACRAL, MELANOMA, LÉNTIGO MALIGNO, DERMATOFIBROMA |
| GLÓBULOS MARRONES | NEVUS, NEVUS ACRAL, MELANOMA |
| PROYECCIONES LINEALES/ PSEUDÓPODOS | NEVUS, MELANOMA |
| MANCHA DE PIGMENTO | NEVUS, MELANOMA |

## Estructuras con anotación SÍ/NO

| Estructura | Enfermedades |
|---|---|
| PATRÓN HOMOGÉNEO AZUL-GRIS | NEVUS, MELANOMA |

## Estructuras de presencia (resto)

| Estructura | Enfermedades que la pueden presentar |
|---|---|
| VELO BLANCO-AZULADO | MELANOMA |
| PSEUDORRETÍCULO | LÉNTIGO MALIGNO, Q. ACTÍNICA, QS |
| ESTRUCTURAS ROMBOIDALES | LÉNTIGO MALIGNO, Q. ACTÍNICA |
| PATRÓN PARALELO AL SURCO | NEVUS ACRAL |
| PATRÓN PARALELO A LA CRESTA | MELANOMA |
| PATRÓN PARALELO EN CELOSÍA | NEVUS ACRAL |
| PATRÓN FIBRILAR | NEVUS ACRAL |
| QUISTES DE MILLIUM | QS |
| TAPONES CÓRNEOS | QS |
| ESTRUCTURAS EN HUELLA DACTILAR | QS |
| BORDE APOLILLADO | QS |
| GRÁNULOS AZUL-GRIS | LÉNTIGO MALIGNO, Q. ACTÍNICA, QS |
| VASOS EN HORQUILLA | QS |
| LAGUNAS ROJAS | ANGIOMA |
| LAGUNAS NEGRO-AZULADAS | ANGIOQUERATOMA |
| PARCHE BLANCO CENTRAL | DERMATOFIBROMA |
| TELANGIECTASIAS ARBORIFORMES | CBC |
| ULCERACIÓN | MELANOMA, CBC, CEC |
| CRISÁLIDAS | MELANOMA, CBC |
| GRANDES NIDOS OVOIDES AZUL-GRIS | CBC |
| GLÓBULOS AZUL-GRIS | CBC |
| ESTRUCTURAS EN HOJA DE ARCE | CBC |
| ESTRUCTURAS EN RUEDA DE CARRO | CBC |
| HIPERQUERATOSIS | BOWEN, Q. ACTÍNICA, CEC |
| VASOS EN CORONA | BOWEN, CEC, HIPERPLASIA SEBÁCEA |
| VASOS GLOMERULARES | Q. ACTÍNICA |
| PATRÓN EN FRESA | Q. ACTÍNICA |
| VASOS LINEALES IRREGULARES | MELANOMA, CEC |
| VASOS POLIMORFOS (3 o más estruct. Vasc) | MELANOMA, CEC |
| LÁGRIMAS AMARILLAS | LEISHMANIA |
| PATRÓN EN ESTALLIDO DE ESTRELLAS BLANCO | LEISHMANIA |
| SIGNO DEL ALA DELTA | ESCABIOSIS |

## Conceptos esperados por enfermedad

Para cada enfermedad, las estructuras que puede presentar según la matriz (entre paréntesis, el régimen de anotación cuando no es de simple presencia).


**NEVUS** (5): RETÍCULO PIGMENTADO *(típico/atípico)*; GLÓBULOS MARRONES *(típico/atípico)*; PATRÓN HOMOGÉNEO AZUL-GRIS *(sí/no)*; PROYECCIONES LINEALES/ PSEUDÓPODOS *(típico/atípico)*; MANCHA DE PIGMENTO *(típico/atípico)*.

**NEVUS ACRAL** (5): RETÍCULO PIGMENTADO *(típico/atípico)*; GLÓBULOS MARRONES *(típico/atípico)*; PATRÓN PARALELO AL SURCO; PATRÓN PARALELO EN CELOSÍA; PATRÓN FIBRILAR.

**MELANOMA** (11): RETÍCULO PIGMENTADO *(típico/atípico)*; GLÓBULOS MARRONES *(típico/atípico)*; PATRÓN HOMOGÉNEO AZUL-GRIS *(sí/no)*; PROYECCIONES LINEALES/ PSEUDÓPODOS *(típico/atípico)*; MANCHA DE PIGMENTO *(típico/atípico)*; VELO BLANCO-AZULADO; PATRÓN PARALELO A LA CRESTA; ULCERACIÓN; CRISÁLIDAS; VASOS LINEALES IRREGULARES; VASOS POLIMORFOS (3 o más estruct. Vasc).

**LÉNTIGO MALIGNO** (4): RETÍCULO PIGMENTADO *(típico/atípico)*; PSEUDORRETÍCULO; ESTRUCTURAS ROMBOIDALES; GRÁNULOS AZUL-GRIS.

**CBC** (7): TELANGIECTASIAS ARBORIFORMES; ULCERACIÓN; CRISÁLIDAS; GRANDES NIDOS OVOIDES AZUL-GRIS; GLÓBULOS AZUL-GRIS; ESTRUCTURAS EN HOJA DE ARCE; ESTRUCTURAS EN RUEDA DE CARRO.

**BOWEN** (2): HIPERQUERATOSIS; VASOS EN CORONA.

**Q. ACTÍNICA** (6): PSEUDORRETÍCULO; ESTRUCTURAS ROMBOIDALES; GRÁNULOS AZUL-GRIS; HIPERQUERATOSIS; VASOS GLOMERULARES; PATRÓN EN FRESA.

**CEC** (5): ULCERACIÓN; HIPERQUERATOSIS; VASOS EN CORONA; VASOS LINEALES IRREGULARES; VASOS POLIMORFOS (3 o más estruct. Vasc).

**QS** (7): PSEUDORRETÍCULO; QUISTES DE MILLIUM; TAPONES CÓRNEOS; ESTRUCTURAS EN HUELLA DACTILAR; BORDE APOLILLADO; GRÁNULOS AZUL-GRIS; VASOS EN HORQUILLA.

**DERMATOFIBROMA** (2): RETÍCULO PIGMENTADO *(típico/atípico)*; PARCHE BLANCO CENTRAL.

**ANGIOMA** (1): LAGUNAS ROJAS.

**ANGIOQUERATOMA** (1): LAGUNAS NEGRO-AZULADAS.

**ESCABIOSIS** (1): SIGNO DEL ALA DELTA.

**LEISHMANIA** (2): LÁGRIMAS AMARILLAS; PATRÓN EN ESTALLIDO DE ESTRELLAS BLANCO.

**HIPERPLASIA SEBÁCEA** (1): VASOS EN CORONA.


## Uso como verdad SAE

La matriz `enfermedad × estructura` es el *ground-truth* contra el que se evalúan las *features* del SAE Large: para cada concepto dermatoscópico se mide si existe una *feature* que lo prediga (AUROC binaria, ver F.3–F.7). La novedad frente a SkinCon/Derm7pt es doble: (i) vocabulario **dermatoscópico en castellano** definido por la dermatóloga de referencia, y (ii) granularidad **típico/atípico** en las estructuras pigmentadas clave, que una sonda binaria presente/ausente no capta.


## Artefacto

- [`dermatoscopia_estructuras_rosa.json`](dermatoscopia_estructuras_rosa.json) — versión legible por máquina (enfermedades, colores, estructuras con su régimen de anotación y mapeo a enfermedades).

