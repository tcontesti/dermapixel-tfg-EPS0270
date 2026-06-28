# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_dinov2_clipl.py · N1+N2 · DINOv2 ViT-L y CLIP-ViT-L sobre DermapixelAI

Para cada encoder:
  1. Extraer embeddings (CLS token) de las 1062 imgs filtered
  2. LP estándar (LogReg L-BFGS C=1) sobre L1/L2/L3 + bootstrap IC95%
  3. (CLIP-L solo) ZS con prompts en castellano + inglés × 3 plantillas

DINOv2: timm `vit_large_patch14_dinov2.lvd142m` (1024d, SSL puro, generalista visual)
CLIP-L: OpenAI `clip-vit-large-patch14-336` (768d, contrastivo, generalista)

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_extra/{results.json, summary.csv}
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, roc_auc_score,
)
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
OUT_DIR  = ROOT / "output" / "dermapixel_v1_extra"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------

def safe_auroc(yt, prob, n_cls):
    aurocs = []
    for c in range(n_cls):
        if (yt == c).sum() == 0 or (yt != c).sum() == 0: continue
        try: aurocs.append(roc_auc_score((yt == c).astype(int), prob[:, c]))
        except ValueError: continue
    return float(np.mean(aurocs)) if aurocs else float("nan")


def topk_acc(yt, prob, k):
    if prob.shape[1] < k: return 1.0
    topk = np.argsort(-prob, axis=1)[:, :k]
    return float(np.mean([yt[i] in topk[i] for i in range(len(yt))]))


def metrics_all(yt, yp, prob, n_cls):
    return {
        "Acc@1": accuracy_score(yt, yp),
        "Acc@3": topk_acc(yt, prob, 3),
        "BAcc":  balanced_accuracy_score(yt, yp),
        "AUROC": safe_auroc(yt, prob, n_cls),
        "W-F1":  f1_score(yt, yp, average="weighted", zero_division=0),
        "Kappa": cohen_kappa_score(yt, yp),
    }


def bootstrap_metric(yt, yp, prob, n_cls, key, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(yt)
    vals = []
    for _ in range(n_iter):
        idx = []
        for c in classes:
            ic = np.where(yt == c)[0]
            if len(ic) == 0: continue
            idx.extend(rng.choice(ic, size=len(ic), replace=True))
        idx = np.array(idx)
        try:
            m = metrics_all(yt[idx], yp[idx], prob[idx], n_cls)
            vals.append(m[key])
        except (ValueError, IndexError): continue
    if not vals: return None, None, None
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def pack_metrics(yt, yp, prob, n_cls):
    base = metrics_all(yt, yp, prob, n_cls)
    out = {"value": {}, "ci": {}}
    for k in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
        m, lo, hi = bootstrap_metric(yt, yp, prob, n_cls, k)
        out["value"][k] = round(base[k], 4)
        out["ci"][k] = [round(lo, 4) if lo else None, round(hi, 4) if hi else None]
    return out


# -----------------------------------------------------------------------------
# Carga rows + función helper LP por nivel
# -----------------------------------------------------------------------------

def load_rows():
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)
    return rows


def lp_by_level(X, fn2idx, rows, level):
    """LP estándar + bootstrap por nivel."""
    tr_rows = [r for r in rows if r["split"] == "train" and r["image_filename"] in fn2idx]
    te_rows = [r for r in rows if r["split"] == "test"  and r["image_filename"] in fn2idx]
    X_tr = np.array([X[fn2idx[r["image_filename"]]] for r in tr_rows])
    X_te = np.array([X[fn2idx[r["image_filename"]]] for r in te_rows])
    classes = sorted({r[level] for r in tr_rows})
    lab2id = {l: i for i, l in enumerate(classes)}
    y_tr = np.array([lab2id[r[level]] for r in tr_rows])
    y_te_all = np.array([lab2id.get(r[level], -1) for r in te_rows])
    mask = y_te_all >= 0
    X_te = X_te[mask]; y_te = y_te_all[mask]
    n_cls = len(lab2id)

    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)
    if prob.shape[1] < n_cls:
        full = np.zeros((prob.shape[0], n_cls))
        for j, ci in enumerate(clf.classes_): full[:, ci] = prob[:, j]
        prob = full
    res = pack_metrics(y_te, y_pred, prob, n_cls)
    res["n_train"] = int(len(y_tr)); res["n_test"] = int(len(y_te)); res["n_classes"] = int(n_cls)
    return res


# -----------------------------------------------------------------------------
# Extractores
# -----------------------------------------------------------------------------

def extract_dinov2(rows):
    print("\n=== DINOv2 ViT-L ===")
    import timm
    model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0, dynamic_img_size=True)
    model = model.to(DEVICE).eval()
    # Transform DINOv2 oficial
    T = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    embeddings, filenames = [], []
    t0 = time.time()
    batch_imgs, batch_fns = [], []
    bs = 32
    for i, r in enumerate(rows):
        try:
            img = Image.open(DATASET / r["image_path"]).convert("RGB")
            batch_imgs.append(T(img))
            batch_fns.append(r["image_filename"])
        except Exception:
            continue
        if len(batch_imgs) >= bs or i == len(rows) - 1:
            if not batch_imgs: continue
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    feats = model(torch.stack(batch_imgs).to(DEVICE))
            embeddings.append(feats.float().cpu().numpy())
            filenames.extend(batch_fns)
            batch_imgs, batch_fns = [], []
    X = np.vstack(embeddings)
    print(f"  shape: {X.shape} en {time.time()-t0:.1f}s")
    del model; torch.cuda.empty_cache()
    return X, filenames


def extract_clip_l(rows):
    print("\n=== CLIP-ViT-L-336 (OpenAI) ===")
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    embeddings, filenames = [], []
    t0 = time.time()
    batch_imgs, batch_fns = [], []
    bs = 16
    for i, r in enumerate(rows):
        try:
            img = Image.open(DATASET / r["image_path"]).convert("RGB")
            batch_imgs.append(img)
            batch_fns.append(r["image_filename"])
        except Exception:
            continue
        if len(batch_imgs) >= bs or i == len(rows) - 1:
            if not batch_imgs: continue
            inputs = proc(images=batch_imgs, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
                feats = vision_out.pooler_output  # (B, 1024)
                feats = model.visual_projection(feats)  # (B, 768) projected space
            embeddings.append(feats.float().cpu().numpy())
            filenames.extend(batch_fns)
            batch_imgs, batch_fns = [], []
    X = np.vstack(embeddings)
    print(f"  shape: {X.shape} en {time.time()-t0:.1f}s")
    return model, proc, X, filenames


# -----------------------------------------------------------------------------
# Zero-Shot CLIP-L
# -----------------------------------------------------------------------------

L1_EN = {
    "Patología inflamatoria": "inflammatory disease",
    "Patología infecciosa":   "infectious disease",
    "Patología tumoral":      "tumoral skin disease",
    "Genodermatosis":         "genodermatosis",
}

TEMPLATES = {
    "es": ["una fotografía clínica de {c}",
           "una imagen dermatológica de {c}",
           "una lesión cutánea de tipo {c}"],
    "en": ["a clinical photograph of {c}",
           "a dermatological image of {c}",
           "a skin lesion of {c}"],
}


def run_clip_zs(model, proc, rows, level, lang="es"):
    test_rows = [r for r in rows if r["split"] == "test"]
    train_classes = sorted({r[level] for r in rows if r["split"] == "train"})
    test_rows = [r for r in test_rows if r[level] in train_classes]
    n_cls = len(train_classes)
    class_to_id = {c: i for i, c in enumerate(train_classes)}
    templates = TEMPLATES[lang]
    all_texts = []
    for c in train_classes:
        label = L1_EN.get(c, c) if (lang == "en" and level == "ontology_l1") else c
        for t in templates:
            all_texts.append(t.format(c=label))

    inputs_text = proc(text=all_texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=77).to(DEVICE)
    with torch.no_grad():
        text_out = model.text_model(**inputs_text)
        text_feats = text_out.pooler_output
        text_feats = model.text_projection(text_feats)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    dim = text_feats.shape[-1]
    text_feats = text_feats.view(n_cls, len(templates), dim).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    img_feats_all = []
    for i in range(0, len(test_rows), 16):
        batch = test_rows[i:i+16]
        imgs = [Image.open(DATASET / r["image_path"]).convert("RGB") for r in batch]
        inp = proc(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            v_out = model.vision_model(pixel_values=inp["pixel_values"])
            ifeats = model.visual_projection(v_out.pooler_output)
            ifeats = ifeats / ifeats.norm(dim=-1, keepdim=True)
        img_feats_all.append(ifeats.cpu().numpy())
    img_feats = np.vstack(img_feats_all)
    logits = img_feats @ text_feats.cpu().numpy().T
    e = np.exp(100 * (logits - logits.max(axis=1, keepdims=True)))
    prob = e / e.sum(axis=1, keepdims=True)
    y_pred = prob.argmax(axis=1)
    y_true = np.array([class_to_id[r[level]] for r in test_rows])
    return pack_metrics(y_true, y_pred, prob, n_cls)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    rows = load_rows()
    print(f"Filas: {len(rows)}")
    results = {"models": {}}

    # === N1: DINOv2 LP ===
    X_d, fn_d = extract_dinov2(rows)
    np.save(OUT_DIR / "dinov2_embeddings.npy", X_d)
    with (OUT_DIR / "dinov2_filenames.json").open("w") as f: json.dump(fn_d, f)
    fn2idx_d = {fn: i for i, fn in enumerate(fn_d)}
    dino_res = {}
    print("\n--- DINOv2 LP ---")
    for lvl in ("ontology_l1", "ontology_l2", "ontology_l3"):
        r = lp_by_level(X_d, fn2idx_d, rows, lvl)
        dino_res[lvl] = r
        v = r["value"]
        print(f"  {lvl}: Acc@1={v['Acc@1']} BAcc={v['BAcc']} AUROC={v['AUROC']}")
    results["models"]["DINOv2_ViT-L"] = {"lp": dino_res}

    # === N2: CLIP-L LP + ZS ===
    model_c, proc_c, X_c, fn_c = extract_clip_l(rows)
    np.save(OUT_DIR / "clip_l_embeddings.npy", X_c)
    with (OUT_DIR / "clip_l_filenames.json").open("w") as f: json.dump(fn_c, f)
    fn2idx_c = {fn: i for i, fn in enumerate(fn_c)}

    clip_lp = {}
    print("\n--- CLIP-L LP ---")
    for lvl in ("ontology_l1", "ontology_l2", "ontology_l3"):
        r = lp_by_level(X_c, fn2idx_c, rows, lvl)
        clip_lp[lvl] = r
        v = r["value"]
        print(f"  {lvl}: Acc@1={v['Acc@1']} BAcc={v['BAcc']} AUROC={v['AUROC']}")

    clip_zs = {}
    print("\n--- CLIP-L ZS ---")
    for lvl in ("ontology_l1", "ontology_l2", "ontology_l3"):
        for lang in ("es", "en"):
            if lvl == "ontology_l3" and lang == "en": continue
            try:
                r = run_clip_zs(model_c, proc_c, rows, lvl, lang)
                key = f"{lvl}_{lang}"
                clip_zs[key] = r
                v = r["value"]
                print(f"  {key}: Acc@1={v['Acc@1']} BAcc={v['BAcc']} AUROC={v['AUROC']}")
            except Exception as e:
                print(f"  ! error {lvl}/{lang}: {e}")

    results["models"]["CLIP-L-336"] = {"lp": clip_lp, "zs": clip_zs}

    # Save
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "level", "metric", "value", "ci_low", "ci_high"])
        for name, mdata in results["models"].items():
            for mode, lvls in mdata.items():
                for k, r in lvls.items():
                    for met in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                        v = r["value"][met]
                        lo, hi = r["ci"][met]
                        w.writerow([name, mode, k, met, v, lo, hi])
    print(f"\n✓ {OUT_DIR / 'results.json'}")
    print(f"✓ {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
