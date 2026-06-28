# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_zs_eval.py · Zero-shot clínico sobre DermapixelAI 1.0

Modelos:
  1. DermLIP v2 (PanDerm-base + PubMedBERT, fork custom open_clip Derm1M)
  2. SigLIP-SO400M (Google, multilingüe)
  3. BiomedCLIP (Microsoft, PubMedBERT + ViT-B/16, open_clip standard)

Para cada modelo:
  - Castellano (palabras-clase tal cual del dataset)
  - Inglés (traducción manual L1 + L2; L3 demasiado largo → solo L1/L2)
  - Métricas: Acc@1, Acc@3, Acc@5, BAcc, AUROC OvR macro
  - Bootstrap IC95% estratificado, 1000 resamples
  - Test set DermapixelAI 1.0 (N=36 en L1/L2, N=28 en L3 tras filtrar)

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_zs/{results.json, summary.csv, report.md}

Uso (desde la Spark, cwd=$DERMAPIXEL_ROOT):
    python3 dermapixel_zs_eval.py
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, cohen_kappa_score, f1_score,
)

warnings.filterwarnings("ignore")

ROOT       = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET    = ROOT / "datasets" / "dermapixel_v1"
OUT_DIR    = ROOT / "output" / "dermapixel_v1_zs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# -----------------------------------------------------------------------------
# 1 · Cargar dataset + consolidar L2
# -----------------------------------------------------------------------------

L2_CANON = {
    "Trastornos queratinización": "Trastornos de la queratinización",
}


def load_rows():
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)
    return rows


# -----------------------------------------------------------------------------
# 2 · Prompts (castellano + inglés)
# -----------------------------------------------------------------------------

TEMPLATES = {
    "es": [
        "una fotografía clínica de {c}",
        "una imagen dermatológica de {c}",
        "una lesión cutánea de tipo {c}",
    ],
    "en": [
        "a clinical photograph of {c}",
        "a dermatological image of {c}",
        "a skin lesion of {c}",
    ],
}

L1_EN = {
    "Patología inflamatoria": "inflammatory disease",
    "Patología infecciosa":   "infectious disease",
    "Patología tumoral":      "tumoral skin disease",
    "Genodermatosis":         "genodermatosis",
}

# L2 inglés: traducción manual aproximada de los 38 efectivos (mantenemos
# castellano como nombre clínico cuando no hay traducción estándar).
# Para los casos sin entrada explícita, se usa el nombre castellano.
L2_EN = {
    "Eccemas y dermatitis":       "eczema and dermatitis",
    "Psoriasis":                  "psoriasis",
    "Acné y rosácea":             "acne and rosacea",
    "Urticaria":                  "urticaria",
    "Liquen":                     "lichen planus",
    "Infecciones bacterianas":    "bacterial skin infection",
    "Infecciones virales":        "viral skin infection",
    "Infecciones fúngicas":       "fungal skin infection",
    "Tiñas":                      "tinea",
    "Tumores melanocíticos benignos": "benign melanocytic tumor",
    "Tumores epiteliales benignos":   "benign epithelial tumor",
    "Tumores anexiales":              "adnexal tumor",
    "Tumores vasculares":             "vascular tumor",
    "Cáncer cutáneo no melanoma":     "non-melanoma skin cancer",
    "Melanoma":                       "melanoma",
    "Carcinoma basocelular":          "basal cell carcinoma",
    "Carcinoma escamoso":             "squamous cell carcinoma",
    "Queratosis actínica":            "actinic keratosis",
}


def build_prompts(classes, lang, templates):
    """Devuelve dict {class: [list of prompts]}."""
    return {
        c: [t.format(c=c) for t in templates]
        for c in classes
    }


def class_label(c, lang, level):
    """Devuelve la cadena de la clase en el idioma elegido."""
    if lang == "es":
        return c
    if level == "ontology_l1":
        return L1_EN.get(c, c)
    if level == "ontology_l2":
        return L2_EN.get(c, c)
    # L3: no traducimos (demasiados); usar castellano
    return c


# -----------------------------------------------------------------------------
# 3 · Métricas
# -----------------------------------------------------------------------------

def safe_auroc_ovr(yt, prob, n_classes):
    aurocs = []
    for c in range(n_classes):
        if (yt == c).sum() == 0 or (yt != c).sum() == 0:
            continue
        try:
            aurocs.append(roc_auc_score((yt == c).astype(int), prob[:, c]))
        except ValueError:
            continue
    return float(np.mean(aurocs)) if aurocs else float("nan")


def compute_metrics(y_true, y_pred, prob, n_classes):
    out = {
        "Acc@1":  accuracy_score(y_true, y_pred),
        "BAcc":   balanced_accuracy_score(y_true, y_pred),
        "W-F1":   f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Kappa":  cohen_kappa_score(y_true, y_pred),
        "AUROC":  safe_auroc_ovr(y_true, prob, n_classes),
    }
    # Top-k acc
    for k in (3, 5):
        if n_classes < k:
            out[f"Acc@{k}"] = out["Acc@1"]
            continue
        topk = np.argsort(-prob, axis=1)[:, :k]
        hits = np.array([y_true[i] in topk[i] for i in range(len(y_true))])
        out[f"Acc@{k}"] = float(hits.mean())
    return out


def bootstrap_ci(y_true, y_pred, prob, n_classes, metric_key, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    vals = []
    for _ in range(n_iter):
        idx = []
        for c in classes:
            ic = np.where(y_true == c)[0]
            if len(ic) == 0:
                continue
            idx.extend(rng.choice(ic, size=len(ic), replace=True))
        idx = np.array(idx)
        try:
            m = compute_metrics(y_true[idx], y_pred[idx], prob[idx], n_classes)
            vals.append(m[metric_key])
        except (ValueError, IndexError):
            continue
    if not vals:
        return None, None, None
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# -----------------------------------------------------------------------------
# 4 · DermLIP v2 (fork open_clip Derm1M)
# -----------------------------------------------------------------------------

def load_dermlip_v2():
    derm1m_src = os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "dermfm_zero/src")
    if derm1m_src not in sys.path:
        sys.path.insert(0, derm1m_src)
    # Purgar cualquier open_clip cacheado (el fork debe primar)
    for m in [k for k in list(sys.modules) if k.startswith("open_clip")]:
        del sys.modules[m]
    import open_clip
    from transformers import AutoTokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256",
    )
    model.eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
    print(f"  DermLIP v2: visual={type(model.visual).__name__}")
    return {
        "model": model, "tokenizer": tokenizer,
        "preprocess": preprocess, "kind": "dermlip",
    }


def encode_dermlip(bundle, texts, images):
    model, tokenizer = bundle["model"], bundle["tokenizer"]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=256,
                       return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # Fork Derm1M: encode_text sólo acepta input_ids
        text_feats = model.encode_text(tokens["input_ids"])
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = model.encode_image(images.to(DEVICE))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    return img_feats.cpu(), text_feats.cpu()


# -----------------------------------------------------------------------------
# 5 · SigLIP-SO400M (transformers)
# -----------------------------------------------------------------------------

def load_siglip():
    from transformers import SiglipModel, AutoProcessor
    model = SiglipModel.from_pretrained(
        "google/siglip-so400m-patch14-384").to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    print(f"  SigLIP-SO400M cargado, logit_scale={float(model.logit_scale.exp())}")
    return {"model": model, "processor": processor, "kind": "siglip"}


def encode_siglip(bundle, texts, images_pil):
    """SigLIP: usar text_model y vision_model directos. get_text_features() en
    algunas versiones devuelve el output object completo en lugar del tensor."""
    model, proc = bundle["model"], bundle["processor"]
    text_inputs = proc(text=texts, return_tensors="pt", padding="max_length",
                       truncation=True).to(DEVICE)
    img_inputs = proc(images=images_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_out = model.text_model(input_ids=text_inputs["input_ids"])
        text_feats = text_out.pooler_output if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None else text_out.last_hidden_state[:, 0]
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_out = model.vision_model(pixel_values=img_inputs["pixel_values"])
        img_feats = img_out.pooler_output if hasattr(img_out, "pooler_output") and img_out.pooler_output is not None else img_out.last_hidden_state[:, 0]
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    return img_feats.cpu(), text_feats.cpu()


# -----------------------------------------------------------------------------
# 6 · BiomedCLIP (open_clip standard)
# -----------------------------------------------------------------------------

def load_biomedclip():
    # Purgar el fork de DermLIP si está cargado
    for m in [k for k in list(sys.modules) if k.startswith("open_clip")]:
        del sys.modules[m]
    derm1m_src = os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "dermfm_zero/src")
    if derm1m_src in sys.path:
        sys.path.remove(derm1m_src)
    import open_clip  # standard
    from transformers import AutoTokenizer
    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    # Usar el BertTokenizer HF directamente (evita bug open_clip)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    model.eval().to(DEVICE)
    print(f"  BiomedCLIP cargado (tokenizer HF directo)")
    return {"model": model, "tokenizer": hf_tokenizer,
            "preprocess": preprocess, "kind": "biomedclip"}


def encode_biomedclip(bundle, texts, images):
    model, tokenizer = bundle["model"], bundle["tokenizer"]
    tokens = tokenizer(texts, padding="max_length", max_length=256,
                       truncation=True, return_tensors="pt")
    input_ids = tokens["input_ids"].to(DEVICE)
    with torch.no_grad():
        text_feats = model.encode_text(input_ids)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = model.encode_image(images.to(DEVICE))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    return img_feats.cpu(), text_feats.cpu()


# -----------------------------------------------------------------------------
# 7 · Inferencia ZS sobre un nivel ontológico
# -----------------------------------------------------------------------------

def run_zs(bundle, rows, level, lang, batch_size=32):
    """Devuelve dict con métricas (puntual + IC95%)."""
    train_classes = sorted({r[level] for r in rows if r["split"] == "train"})
    test_rows = [r for r in rows if r["split"] == "test"]
    test_rows = [r for r in test_rows if r[level] in train_classes]
    n_classes = len(train_classes)
    n_test = len(test_rows)
    print(f"  ZS {level} ({lang}): {n_classes} clases, test={n_test}")

    # Construir prompts (ensemble de 3 plantillas → promediar embeddings texto)
    templates = TEMPLATES[lang]
    class_to_id = {c: i for i, c in enumerate(train_classes)}
    all_texts = []
    for c in train_classes:
        label = class_label(c, lang, level)
        for t in templates:
            all_texts.append(t.format(c=label))

    # Encode texts (en bloque)
    kind = bundle["kind"]
    if kind == "dermlip":
        # text only call
        img_dummy = bundle["preprocess"](Image.new("RGB", (224, 224))).unsqueeze(0)
        _, text_feats = encode_dermlip(bundle, all_texts, img_dummy)
    elif kind == "siglip":
        img_dummy = [Image.new("RGB", (384, 384))]
        _, text_feats = encode_siglip(bundle, all_texts, img_dummy)
    elif kind == "biomedclip":
        img_dummy = bundle["preprocess"](Image.new("RGB", (224, 224))).unsqueeze(0)
        _, text_feats = encode_biomedclip(bundle, all_texts, img_dummy)

    # Reshape a (n_classes, n_templates, dim) y promediar
    dim = text_feats.shape[-1]
    text_feats = text_feats.view(n_classes, len(templates), dim).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # Encode images (test) en batches
    img_feats_list = []
    valid_idx = []
    for i in range(0, n_test, batch_size):
        batch = test_rows[i:i+batch_size]
        if kind == "siglip":
            imgs = [Image.open(DATASET / r["image_path"]).convert("RGB") for r in batch]
            ifeats, _ = encode_siglip(bundle, [all_texts[0]], imgs)
        else:
            preprocess = bundle["preprocess"]
            imgs = torch.stack([preprocess(Image.open(DATASET / r["image_path"]).convert("RGB"))
                               for r in batch])
            if kind == "dermlip":
                ifeats, _ = encode_dermlip(bundle, [all_texts[0]], imgs)
            else:
                ifeats, _ = encode_biomedclip(bundle, [all_texts[0]], imgs)
        img_feats_list.append(ifeats)

    img_feats = torch.cat(img_feats_list, dim=0)

    # Cosine sim → softmax → predicción
    logits = (img_feats @ text_feats.T).numpy()
    # softmax con temperatura 100 (consistente con literatura CLIP)
    e = np.exp(100 * (logits - logits.max(axis=1, keepdims=True)))
    prob = e / e.sum(axis=1, keepdims=True)
    y_pred = prob.argmax(axis=1)
    y_true = np.array([class_to_id[r[level]] for r in test_rows])

    base = compute_metrics(y_true, y_pred, prob, n_classes)
    # Bootstrap IC95
    ci = {}
    for k in ("Acc@1", "Acc@3", "Acc@5", "BAcc", "W-F1", "Kappa", "AUROC"):
        m, lo, hi = bootstrap_ci(y_true, y_pred, prob, n_classes, k)
        ci[k] = {"value": round(base[k], 4) if not np.isnan(base[k]) else None,
                 "ci95_low": round(lo, 4) if lo is not None else None,
                 "ci95_high": round(hi, 4) if hi is not None else None}
    return {
        "n_classes": n_classes, "n_test": int(len(y_true)),
        "lang": lang, "level": level, "metrics": ci,
    }


# -----------------------------------------------------------------------------
# 8 · Main
# -----------------------------------------------------------------------------

def main():
    rows = load_rows()
    print(f"Total filas: {len(rows)}")

    results = {"meta": {"dataset": "DermapixelAI 1.0", "n_total": len(rows)},
               "models": {}}
    # Importante: cargar BiomedCLIP PRIMERO, luego SigLIP, luego DermLIP (que
    # purga la cache de open_clip y deja el fork — ya no podríamos volver al
    # standard sin reiniciar).
    loaders = [
        ("BiomedCLIP", load_biomedclip),
        ("SigLIP-SO400M", load_siglip),
        ("DermLIP_v2", load_dermlip_v2),
    ]
    for name, loader in loaders:
        print(f"\n========== {name} ==========")
        t0 = time.time()
        try:
            bundle = loader()
        except Exception as e:
            print(f"  ! no se pudo cargar {name}: {e}")
            results["models"][name] = {"error": str(e)}
            continue

        model_results = {}
        for lang in ("es", "en"):
            for lvl in ("ontology_l1", "ontology_l2", "ontology_l3"):
                # ZS L3 inglés se omite (sin traducciones, == castellano)
                if lang == "en" and lvl == "ontology_l3":
                    continue
                key = f"{lvl}_{lang}"
                try:
                    res = run_zs(bundle, rows, lvl, lang)
                    model_results[key] = res
                    m = res["metrics"]
                    print(f"  {key}: Acc@1={m['Acc@1']['value']}  "
                          f"BAcc={m['BAcc']['value']}  "
                          f"AUROC={m['AUROC']['value']}  "
                          f"Acc@3={m['Acc@3']['value']}")
                except Exception as e:
                    print(f"  ! error {key}: {e}")
                    import traceback; traceback.print_exc()
                    model_results[key] = {"error": str(e)}
        results["models"][name] = model_results
        print(f"  ({time.time()-t0:.1f}s)")

        # Liberar GPU
        del bundle
        torch.cuda.empty_cache()

    # Guardar resultados
    out_json = OUT_DIR / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados → {out_json}")

    # CSV resumen
    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "lang", "n_classes", "n_test", "metric",
                    "value", "ci_low", "ci_high"])
        for model_name, model_res in results["models"].items():
            if "error" in model_res:
                continue
            for key, res in model_res.items():
                if "error" in res:
                    continue
                for met, vals in res["metrics"].items():
                    w.writerow([
                        model_name, res["level"], res["lang"],
                        res["n_classes"], res["n_test"], met,
                        vals["value"], vals["ci95_low"], vals["ci95_high"],
                    ])
    print(f"Resumen CSV → {csv_path}")


if __name__ == "__main__":
    main()
