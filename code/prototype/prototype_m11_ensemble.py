# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
m11_ensemble.py · Módulo M11 — Ensemble ponderado de clasificadores

Combina probabilidades de M1 (HAM10000 FT) + M7 (Unified merged43 + TTA) +
M9 (SpanDerm v0 LoRA L2 castellano) + SigLIP safety screen para producir un
único veredicto consenso por nivel L1/L2/L3.

Pesos derivados de las AUROC reportadas en §4.11 del TFG sobre DermapixelAI:
- L1: PanDerm Large LP gana (M1 FT en HAM ≈ M7 unified) → M1=1.0 M7=1.0 M9=0.5
- L2: SpanDerm v0 LoRA mejor en BAcc (0.363) + DermLIP solo mejor en AUROC
       (0.860) → M9=1.5 M7=1.0 (M1 no es L2)
- L3: PanDerm Large LP mejor AUROC (0.813) → M7=1.5 (M1/M9 no son L3)

Pega este archivo en:
  $DERMAPIXEL_ROOT/scripts/dermapixel_server/m11_ensemble.py
"""
from __future__ import annotations
import time

# Mapeo de clases L1 castellano → categoría
L1_CLASSES = [
    "Patología inflamatoria",
    "Patología infecciosa",
    "Patología tumoral",
    "Genodermatosis",
]

# Mapeo HAM10000 → L1 (case-insensitive match en runtime)
# Las clases HAM en pipeline.py son: actinic keratosis, basal cell carcinoma,
# seborrheic keratosis, dermatofibroma, melanoma, melanocytic nevus, vascular lesion
# Todas mapean a "Patología tumoral" (HAM cubre solo lesiones cutáneas tumorales)
_HAM_TUMORAL = {
    "actinic keratosis", "basal cell carcinoma", "seborrheic keratosis",
    "benign keratosis", "dermatofibroma", "melanoma", "melanocytic nevus",
    "vascular lesion",
}


def _ham_class_to_l1(class_name):
    """Match case-insensitive contra clases HAM conocidas."""
    if not class_name: return None
    cn = class_name.strip().lower()
    if cn in _HAM_TUMORAL:
        return "Patología tumoral"
    return None


# Fix C — Mapping EN→ES para armonizar las salidas de M7 (inglés, merged43) con
# las de M9 (castellano, SpanDerm v0) y M4-bis (castellano, FAISS DermapixelAI).
# Extraído de el frontend del prototipo (i18n/unifiedLabels.js) y armonizado al
# castellano clínico (cláusulas "Patología X" en L1).
_EN_TO_ES = {
    # ---------- L1 (4 clases) ----------
    "Tumoral":            "Patología tumoral",
    "Inflammatory":       "Patología inflamatoria",
    "Infectious":         "Patología infecciosa",
    "Genodermatosis":     "Genodermatosis",

    # ---------- L2 (26 clases) ----------
    # Tumoral
    "Malignant tumors":               "Tumores malignos",
    "Benign melanocytic neoplasms":   "Neoplasias melanocíticas benignas",
    "Benign epithelial tumors":       "Tumores epiteliales benignos",
    "Benign tumors":                  "Tumores benignos",
    "Precancerous tumors":            "Tumores precancerosos",
    "Vascular tumors":                "Tumores vasculares",
    "Hematologic tumors":             "Tumores hematológicos",
    # Inflammatory
    "Eczema and dermatitis":          "Eczema y dermatitis",
    "Psoriasis":                      "Psoriasis",
    "Other inflammatory dermatoses":  "Otras dermatosis inflamatorias",
    "Acantholytic disorders":         "Trastornos acantolíticos",
    "Neutrophilic dermatoses":        "Dermatosis neutrofílicas",
    "Blistering diseases":            "Enfermedades ampollares",
    "Humoral reactivity":             "Reactividad humoral",
    "Drug eruptions":                 "Erupciones medicamentosas",
    "Sebaceous glands":               "Glándulas sebáceas",
    "Hair follicle disorders":        "Trastornos foliculares",
    "Pigmentation disorders":         "Trastornos de la pigmentación",
    "Exogenous agents":               "Agentes exógenos",
    "Systemic and metabolic":         "Sistémicas y metabólicas",
    # Infectious
    "Bacterial infections":           "Infecciones bacterianas",
    "Viral infections":               "Infecciones víricas",
    "Fungal infections":              "Infecciones fúngicas",
    "Parasitic infections":           "Infecciones parasitarias",
    "Infestations and bites":         "Infestaciones y picaduras",
    # Genodermatosis (L2 == L3)
    "Neurofibromatosis T1":           "Neurofibromatosis tipo 1",

    # ---------- L3 (43 clases) ----------
    # Tumoral · malignos
    "Acquired melanocytic nevus":     "Nevus melanocítico adquirido",
    "Melanoma":                       "Melanoma",
    "Basal cell carcinoma":           "Carcinoma basocelular",
    "Invasive squamous cell carcinoma":"Carcinoma escamoso invasivo",
    "Lentigo maligna":                "Lentigo maligno",
    "Bowen disease":                  "Enfermedad de Bowen",
    "Mycosis fungoides":              "Micosis fungoide",
    # Tumoral · precancerosos
    "Actinic keratosis":              "Queratosis actínica",
    # Tumoral · benignos epiteliales
    "Seborrheic keratosis":           "Queratosis seborreica",
    "Porokeratosis":                  "Poroqueratosis",
    # Tumoral · benignos
    "Dermatofibroma":                 "Dermatofibroma",
    # Tumoral · vasculares
    "Capillary angioma":              "Angioma capilar",
    # Inflammatory
    "Atopic dermatitis":              "Dermatitis atópica",
    "Plaque psoriasis":               "Psoriasis en placas",
    "Lichen planus":                  "Liquen plano",
    "Pityriasis rubra pilaris":       "Pitiriasis rubra pilaris",
    "Photosensitivity":               "Fotosensibilidad",
    "Dermatomyositis":                "Dermatomiositis",
    "Lupus erythematosus":            "Lupus eritematoso",
    "Hailey-Hailey disease":          "Enfermedad de Hailey-Hailey",
    "Cutaneous mucinosis":            "Mucinosis cutánea",
    "Prurigo nodularis":              "Prúrigo nodular",
    "Other drug eruptions":           "Otras erupciones medicamentosas",
    "Acne":                           "Acné",
    "Post-inflammatory hyperpigmentation": "Hiperpigmentación postinflamatoria",
    "Confluent and reticulated papillomatosis": "Papilomatosis confluente y reticulada",
    # Infectious · bacterial
    "Acute cellulitis":               "Celulitis aguda",
    "Bacterial folliculitis":         "Foliculitis bacteriana",
    # Infectious · viral
    "Common wart":                    "Verruga común",
    "Herpes simplex":                 "Herpes simple",
    # Infectious · fungal
    "Tinea corporis":                 "Tinea corporis",
    "Tinea unguium":                  "Onicomicosis",
    # Infectious · parasitic
    "Tungiasis":                      "Tungiasis",
}


def _normalize_class(name):
    """Aplica EN→ES si la clave existe en _EN_TO_ES; si no, devuelve el original.

    Esto colapsa entradas EN/ES duplicadas (caso típico: M7 castellano:'Tumoral'
    + M1 'Patología tumoral' agregándose en categorías separadas).
    """
    if not name: return name
    if name in _EN_TO_ES:
        return _EN_TO_ES[name]
    # Si ya viene en castellano (M9/M4-bis lo dan en castellano) lo dejamos tal cual
    return name


def _normalize_probs(probs):
    """Aplica _normalize_class a las CLAVES de un dict de probabilidades.

    Si dos claves colapsan en la misma castellano (caso esperado: M7 da 'Tumoral'
    + M4-bis da 'Patología tumoral'), SUMAMOS las probabilidades.
    """
    if not probs: return probs
    out = {}
    for k, v in probs.items():
        kn = _normalize_class(k)
        out[kn] = out.get(kn, 0.0) + v
    return out


def _l1_from_m1(m1_classifications):
    """M1 devuelve clases HAM; agregamos por L1 sumando probabilidades."""
    if not m1_classifications:
        return {}
    l1_probs = {c: 0.0 for c in L1_CLASSES}
    for cls in m1_classifications:
        l1 = _ham_class_to_l1(cls.get("class_name", ""))
        if l1:
            l1_probs[l1] += cls.get("probability", 0.0)
    return l1_probs


def _normalize_dict(d):
    s = sum(d.values())
    if s <= 0: return d
    return {k: v / s for k, v in d.items()}


def _weighted_average(probability_dicts_with_weights):
    """Combina varios diccionarios prob{clase: p} con sus pesos.

    Sólo agrega clases presentes en al menos uno. Pesos NO aplicados a
    clases inexistentes en ese diccionario.
    """
    out = {}
    total_w = {}
    for probs, w in probability_dicts_with_weights:
        if not probs: continue
        for k, v in probs.items():
            out[k] = out.get(k, 0.0) + v * w
            total_w[k] = total_w.get(k, 0.0) + w
    # Normalizar por peso efectivo aplicado a cada clase
    norm = {}
    for k, v in out.items():
        norm[k] = v / max(total_w[k], 1e-8)
    # Re-normalizar a suma 1 para mantener distribución de probabilidad
    s = sum(norm.values())
    if s > 0:
        norm = {k: v / s for k, v in norm.items()}
    return norm


def ensemble(result, unified_result=None):
    """Combina M1/M7/M9/M4-bis en un único veredicto por nivel.

    Args:
        result: dict completo de pipeline.analyze() (con todas las claves)
        unified_result: dict M7 unified (level_1, level_2, level_3) - opcional,
                        si no se pasa se intenta leer de result["unified"]

    Returns:
        dict con:
          - L1, L2, L3: cada uno {top_class, top_prob, all_probs (top-10),
                                  contributors (lista módulos que aportaron)}
          - latency_ms
    """
    t0 = time.time()

    # Extraer datos por módulo
    m1_cls = result.get("classifications", []) if result else []
    m7 = (unified_result or {}).get("level_1", {}) if unified_result else (
         (result.get("unified") or {}).get("level_1", {}) if result else {})
    m7_l1_probs_raw = m7.get("all_probs", {}) if m7 else {}

    u = unified_result or (result.get("unified") if result else None) or {}
    m7_l2_probs_raw = (u.get("level_2") or {}).get("all_probs", {}) if u else {}
    m7_l3_probs_raw = (u.get("level_3") or {}).get("all_probs", {}) if u else {}

    # Fix C — normalizar EN→ES para que NO se dupliquen las entradas
    m7_l1_probs = _normalize_probs(m7_l1_probs_raw)
    m7_l2_probs = _normalize_probs(m7_l2_probs_raw)
    m7_l3_probs = _normalize_probs(m7_l3_probs_raw)

    m9 = result.get("m9_spanderm") if result else None
    m9_l2_probs = _normalize_probs(m9.get("probabilities", {})) if m9 else {}

    m4 = result.get("m4bis_rag_es") if result else None
    m4_neighbors = m4.get("neighbors", []) if m4 else []

    # Convertir M4-bis vecinos a distribución de probabilidad ponderada por
    # similitud cosénica (softmax con T=10).
    def m4bis_dist(level_key):
        if not m4_neighbors: return {}
        import math
        sims = [n["similarity"] for n in m4_neighbors]
        maxs = max(sims) if sims else 0
        weights = [math.exp(10 * (s - maxs)) for s in sims]
        ws = sum(weights) or 1.0
        weights = [w / ws for w in weights]
        d = {}
        for n, w in zip(m4_neighbors, weights):
            cls = n.get(level_key)
            if not cls: continue
            d[cls] = d.get(cls, 0.0) + w
        return _normalize_probs(d)

    m4_l1 = m4bis_dist("ontology_l1")
    m4_l2 = m4bis_dist("ontology_l2")
    m4_l3 = m4bis_dist("ontology_l3")

    # M1 → L1 (agregado por HAM→L1; ya está en castellano "Patología tumoral")
    m1_l1 = _l1_from_m1(m1_cls)
    m1_l1_normalized = _normalize_dict(m1_l1) if sum(m1_l1.values()) > 0 else {}

    # Ensemble por nivel
    L1 = _weighted_average([
        (m1_l1_normalized, WEIGHTS["L1"]["M1"]),
        (m7_l1_probs,     WEIGHTS["L1"]["M7"]),
        (m4_l1,           WEIGHTS["L1"]["M4bis"]),
    ])
    L2 = _weighted_average([
        (m7_l2_probs, WEIGHTS["L2"]["M7"]),
        (m9_l2_probs, WEIGHTS["L2"]["M9"]),
        (m4_l2,       WEIGHTS["L2"]["M4bis"]),
    ])
    L3 = _weighted_average([
        (m7_l3_probs, WEIGHTS["L3"]["M7"]),
        (m4_l3,       WEIGHTS["L3"]["M4bis"]),
    ])

    def top_k(d, k=10):
        sorted_items = sorted(d.items(), key=lambda x: -x[1])
        return [{"class": c, "prob": float(p)} for c, p in sorted_items[:k]]

    def contributors(level):
        out = []
        if level == "L1":
            if m1_l1_normalized: out.append("M1")
            if m7_l1_probs: out.append("M7")
            if m4_l1: out.append("M4-bis")
        elif level == "L2":
            if m7_l2_probs: out.append("M7")
            if m9_l2_probs: out.append("M9")
            if m4_l2: out.append("M4-bis")
        elif level == "L3":
            if m7_l3_probs: out.append("M7")
            if m4_l3: out.append("M4-bis")
        return out

    def pack(level_dict, level):
        if not level_dict:
            return {"top_class": None, "top_prob": 0.0, "top_k": [],
                    "contributors": contributors(level)}
        top = max(level_dict.items(), key=lambda x: x[1])
        return {
            "top_class": top[0],
            "top_prob": float(top[1]),
            "top_k": top_k(level_dict, k=10),
            "contributors": contributors(level),
        }

    return {
        "model": "M11_ensemble_weighted",
        "L1": pack(L1, "L1"),
        "L2": pack(L2, "L2"),
        "L3": pack(L3, "L3"),
        "weights": WEIGHTS,
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }


# Pesos por nivel (basados en AUROC §4.11 + ajuste empírico por coherencia jerárquica)
# Iteración v1.7: M7 unified merged43+TTA es el clasificador más fiable global
# (AUROC L1 0.873, L2 0.860, L3 0.813) y SpanDerm v0 L2 es BAcc-fuerte pero
# pierde en lesiones malignas (vota mucho hacia neoplasias melanocíticas
# benignas). Subimos peso M7 en L2 a 1.5 y bajamos M9 a 1.0 para mantener
# coherencia con L1 (M7=1.0) y L3 (M7=1.5).
WEIGHTS = {
    "L1": {"M1": 1.0, "M7": 1.0, "M9": 0.5, "M4bis": 0.3},
    "L2": {"M7": 1.5, "M9": 1.0, "M4bis": 0.5},
    "L3": {"M7": 1.5, "M4bis": 0.5},
}


if __name__ == "__main__":
    # Test rápido con dummy que reproduce el bug EN/ES duplicado del v1.5
    dummy = {
        "classifications": [
            {"class_name": "Melanoma", "probability": 0.6, "rank": 1},
            {"class_name": "Melanocytic Nevus", "probability": 0.3, "rank": 2},
        ],
        "unified": {
            # M7 da en inglés: 'Tumoral' (no 'Patología tumoral')
            "level_1": {"all_probs": {"Tumoral": 0.8, "Inflammatory": 0.2}},
            "level_2": {"all_probs": {"Malignant tumors": 0.5,
                                       "Benign melanocytic neoplasms": 0.3}},
            "level_3": {"all_probs": {"Melanoma": 0.4,
                                       "Basal cell carcinoma": 0.2}},
        },
        # M9 ya da en castellano
        "m9_spanderm": {"probabilities": {"Tumores malignos": 0.4,
                                          "Tumores epiteliales benignos": 0.3}},
        # M4-bis también en castellano
        "m4bis_rag_es": {"neighbors": [
            {"similarity": 0.95, "ontology_l1": "Patología tumoral",
             "ontology_l2": "Tumores epiteliales benignos",
             "ontology_l3": "Tricoepitelioma"},
            {"similarity": 0.93, "ontology_l1": "Patología tumoral",
             "ontology_l2": "Tumores anexiales", "ontology_l3": "Hidrocistoma"},
        ]},
    }
    r = ensemble(dummy)
    import json
    print(json.dumps(r, indent=2, ensure_ascii=False))
    # Esperado: L1.top_class == "Patología tumoral" UNA SOLA VEZ
    #           L2.top_class == "Tumores malignos" (con peso M7+M9)
