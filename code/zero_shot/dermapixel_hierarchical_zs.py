# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_hierarchical_zs.py · N4 · Hierarchical ZS L1→L2→L3 condicionado

Pipeline clínico ZS:
  1. ZS L1 (4 clases) con DermLIP v2 — predice etiología
  2. Dado L1 predicho, ZS L2 RESTRINGIDO a las clases L2 hijas de ese L1
  3. Dado L2 predicho, ZS L3 RESTRINGIDO a las clases L3 hijas de ese L2

Compara con ZS plano (§4.12) sin condicionamiento jerárquico.

Mapping L1→L2→L3 derivado del corpus (no del vocabulario ontológico completo).
Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_hierzs/{results.json, report.md}
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
OUT_DIR  = ROOT / "output" / "dermapixel_v1_hierzs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


TEMPLATES_ES = [
    "una fotografía clínica de {c}",
    "una imagen dermatológica de {c}",
    "una lesión cutánea de tipo {c}",
]


def load_rows():
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)
    return rows


def load_dermlip_v2():
    derm1m_src = os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "dermfm_zero/src")
    if derm1m_src not in sys.path:
        sys.path.insert(0, derm1m_src)
    for m in [k for k in list(sys.modules) if k.startswith("open_clip")]:
        del sys.modules[m]
    import open_clip
    from transformers import AutoTokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256")
    model.eval().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
    return model, tokenizer, preprocess


@torch.no_grad()
def encode_text(model, tokenizer, texts):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=256,
                       return_tensors="pt").to(DEVICE)
    feats = model.encode_text(tokens["input_ids"])
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


@torch.no_grad()
def encode_images(model, preprocess, rows, key="image_path"):
    feats_all = []
    bs = 32
    batch = []
    for i, r in enumerate(rows):
        img = Image.open(DATASET / r[key]).convert("RGB")
        batch.append(preprocess(img))
        if len(batch) >= bs or i == len(rows) - 1:
            x = torch.stack(batch).to(DEVICE)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats_all.append(f.cpu().numpy())
            batch = []
    return np.vstack(feats_all)


def zs_predict(img_feats, class_texts_emb, classes_order):
    """Para cada img, similitud coseno con cada clase, argmax."""
    logits = img_feats @ class_texts_emb.T
    e = np.exp(100 * (logits - logits.max(axis=1, keepdims=True)))
    prob = e / e.sum(axis=1, keepdims=True)
    pred_idx = prob.argmax(axis=1)
    return [classes_order[i] for i in pred_idx], prob


def get_class_embeddings(model, tokenizer, classes):
    texts = []
    for c in classes:
        for t in TEMPLATES_ES:
            texts.append(t.format(c=c))
    feats = encode_text(model, tokenizer, texts)
    dim = feats.shape[-1]
    feats = feats.reshape(len(classes), len(TEMPLATES_ES), dim).mean(axis=1)
    feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
    return feats


def metrics_bin(yt, yp):
    if len(set(yt)) < 2:
        return {"Acc@1": accuracy_score(yt, yp)}
    return {
        "Acc@1": accuracy_score(yt, yp),
        "BAcc":  balanced_accuracy_score(yt, yp),
        "W-F1":  f1_score(yt, yp, average="weighted", zero_division=0),
    }


def main():
    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"train={len(train_rows)} test={len(test_rows)}")

    # Construir jerarquía L1 → {L2: {L3}}
    hierarchy = defaultdict(lambda: defaultdict(set))
    for r in train_rows:
        hierarchy[r["ontology_l1"]][r["ontology_l2"]].add(r["ontology_l3"])
    print(f"L1 niveles: {len(hierarchy)}")
    for l1, l2s in hierarchy.items():
        n_l3 = sum(len(l3s) for l3s in l2s.values())
        print(f"  {l1}: {len(l2s)} L2 → {n_l3} L3")

    print("\n=== Cargar DermLIP v2 ===")
    model, tokenizer, preprocess = load_dermlip_v2()

    # Embeddings de las clases en cada nivel (ALL classes)
    all_l1 = sorted(hierarchy.keys())
    all_l2 = sorted({l2 for l2s in hierarchy.values() for l2 in l2s.keys()})
    all_l3 = sorted({l3 for l2s in hierarchy.values() for l3s in l2s.values() for l3 in l3s})

    print(f"\nL1={len(all_l1)} L2={len(all_l2)} L3={len(all_l3)}")
    print("\nEncoding text classes (ES) ...")
    emb_l1 = get_class_embeddings(model, tokenizer, all_l1)
    emb_l2 = get_class_embeddings(model, tokenizer, all_l2)
    emb_l3 = get_class_embeddings(model, tokenizer, all_l3)
    l1_to_idx = {c: i for i, c in enumerate(all_l1)}
    l2_to_idx = {c: i for i, c in enumerate(all_l2)}
    l3_to_idx = {c: i for i, c in enumerate(all_l3)}

    print("\n=== Encoding test images ===")
    test_img_feats = encode_images(model, preprocess, test_rows)
    print(f"  shape: {test_img_feats.shape}")

    # ZS plano L1 (baseline igual a §4.12)
    print("\n=== ZS plano (referencia §4.12) ===")
    pred_l1_flat, _ = zs_predict(test_img_feats, emb_l1, all_l1)
    true_l1 = [r["ontology_l1"] for r in test_rows]
    m_l1_flat = metrics_bin(true_l1, pred_l1_flat)
    print(f"  L1 plano: Acc@1={m_l1_flat['Acc@1']:.3f} BAcc={m_l1_flat['BAcc']:.3f}")

    pred_l2_flat, _ = zs_predict(test_img_feats, emb_l2, all_l2)
    true_l2 = [r["ontology_l2"] for r in test_rows]
    # filtrar test rows con l2 ausente en train
    valid_l2 = [c in all_l2 for c in true_l2]
    pred_l2_flat_v = [p for p, v in zip(pred_l2_flat, valid_l2) if v]
    true_l2_v = [t for t, v in zip(true_l2, valid_l2) if v]
    m_l2_flat = metrics_bin(true_l2_v, pred_l2_flat_v)
    print(f"  L2 plano: Acc@1={m_l2_flat['Acc@1']:.3f} BAcc={m_l2_flat['BAcc']:.3f}")

    pred_l3_flat, _ = zs_predict(test_img_feats, emb_l3, all_l3)
    true_l3 = [r["ontology_l3"] for r in test_rows]
    valid_l3 = [c in all_l3 for c in true_l3]
    pred_l3_flat_v = [p for p, v in zip(pred_l3_flat, valid_l3) if v]
    true_l3_v = [t for t, v in zip(true_l3, valid_l3) if v]
    m_l3_flat = metrics_bin(true_l3_v, pred_l3_flat_v)
    print(f"  L3 plano: Acc@1={m_l3_flat['Acc@1']:.3f} BAcc={m_l3_flat['BAcc']:.3f}")

    # ZS Hierarchical: predecir L1 → restringir L2 a los hijos → restringir L3 a hijos
    print("\n=== ZS Hierarchical (cascada L1→L2→L3) ===")

    pred_l2_hier, pred_l3_hier = [], []
    for i, r in enumerate(test_rows):
        # L1 (con todas las 4 clases) — igual al plano
        l1_pred = pred_l1_flat[i]
        img_feat = test_img_feats[i:i+1]  # (1, dim)

        # L2 restringido a hijos de l1_pred
        candidates_l2 = sorted(hierarchy[l1_pred].keys())
        if candidates_l2:
            sub_emb_l2 = np.stack([emb_l2[l2_to_idx[c]] for c in candidates_l2])
            sims_l2 = (img_feat @ sub_emb_l2.T).flatten()
            l2_pred = candidates_l2[sims_l2.argmax()]
        else:
            l2_pred = pred_l2_flat[i]  # fallback
        pred_l2_hier.append(l2_pred)

        # L3 restringido a hijos de l2_pred
        candidates_l3 = sorted(hierarchy[l1_pred][l2_pred])
        if candidates_l3:
            sub_emb_l3 = np.stack([emb_l3[l3_to_idx[c]] for c in candidates_l3])
            sims_l3 = (img_feat @ sub_emb_l3.T).flatten()
            l3_pred = candidates_l3[sims_l3.argmax()]
        else:
            l3_pred = pred_l3_flat[i]
        pred_l3_hier.append(l3_pred)

    # Filtrar test rows con etiquetas válidas
    pred_l2_hier_v = [p for p, v in zip(pred_l2_hier, valid_l2) if v]
    pred_l3_hier_v = [p for p, v in zip(pred_l3_hier, valid_l3) if v]
    m_l2_hier = metrics_bin(true_l2_v, pred_l2_hier_v)
    m_l3_hier = metrics_bin(true_l3_v, pred_l3_hier_v)
    print(f"  L2 hier: Acc@1={m_l2_hier['Acc@1']:.3f} BAcc={m_l2_hier['BAcc']:.3f}")
    print(f"  L3 hier: Acc@1={m_l3_hier['Acc@1']:.3f} BAcc={m_l3_hier['BAcc']:.3f}")

    # Top-3 hierarchical: para L1, conservar top-3 L1 candidates → unión de L2 hijos → top-1
    print("\n=== ZS Hierarchical top-3 (relaja decisión L1) ===")
    sims_l1_all = test_img_feats @ emb_l1.T  # (n_test, 4)
    top3_l1_idx = np.argsort(-sims_l1_all, axis=1)[:, :3]  # (n_test, 3)

    pred_l2_top3, pred_l3_top3 = [], []
    for i, r in enumerate(test_rows):
        # Unión de hijos L2 de los top-3 L1
        top3_l1 = [all_l1[j] for j in top3_l1_idx[i]]
        cand_l2 = sorted({l2 for l1c in top3_l1 for l2 in hierarchy[l1c].keys()})
        if cand_l2:
            sub_emb = np.stack([emb_l2[l2_to_idx[c]] for c in cand_l2])
            sims = (test_img_feats[i:i+1] @ sub_emb.T).flatten()
            l2_pred = cand_l2[sims.argmax()]
        else:
            l2_pred = pred_l2_flat[i]
        pred_l2_top3.append(l2_pred)

        # L3 condicionado a top-3 L1 + l2_pred
        cand_l3 = sorted({l3 for l1c in top3_l1
                          for l3 in hierarchy[l1c].get(l2_pred, set())})
        if cand_l3:
            sub_emb = np.stack([emb_l3[l3_to_idx[c]] for c in cand_l3])
            sims = (test_img_feats[i:i+1] @ sub_emb.T).flatten()
            l3_pred = cand_l3[sims.argmax()]
        else:
            l3_pred = pred_l3_flat[i]
        pred_l3_top3.append(l3_pred)

    pred_l2_top3_v = [p for p, v in zip(pred_l2_top3, valid_l2) if v]
    pred_l3_top3_v = [p for p, v in zip(pred_l3_top3, valid_l3) if v]
    m_l2_top3 = metrics_bin(true_l2_v, pred_l2_top3_v)
    m_l3_top3 = metrics_bin(true_l3_v, pred_l3_top3_v)
    print(f"  L2 hier top-3: Acc@1={m_l2_top3['Acc@1']:.3f} BAcc={m_l2_top3['BAcc']:.3f}")
    print(f"  L3 hier top-3: Acc@1={m_l3_top3['Acc@1']:.3f} BAcc={m_l3_top3['BAcc']:.3f}")

    # Acc condicional: dado que L1 correcto, ¿cuánto mejora L2/L3?
    correct_l1 = [p == t for p, t in zip(pred_l1_flat, true_l1)]
    pl1_correct = [(p, t) for p, t, c in zip(pred_l2_hier, true_l2, correct_l1) if c]
    if pl1_correct:
        cond_acc_l2 = sum(1 for p, t in pl1_correct if p == t) / len(pl1_correct)
        print(f"\n  Acc L2 hier | L1 correcto: {cond_acc_l2:.3f} (sobre {len(pl1_correct)} casos)")

    # Save
    results = {
        "n_test": len(test_rows),
        "l1": {"flat": m_l1_flat},
        "l2": {"flat": m_l2_flat, "hier": m_l2_hier, "hier_top3": m_l2_top3},
        "l3": {"flat": m_l3_flat, "hier": m_l3_hier, "hier_top3": m_l3_top3},
    }
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "variant", "metric", "value"])
        for lvl, vs in results.items():
            if not isinstance(vs, dict): continue
            for v, m in vs.items():
                if isinstance(m, dict):
                    for k, val in m.items():
                        w.writerow([lvl, v, k, round(val, 4)])
    print(f"\n✓ {OUT_DIR / 'results.json'}")
    print(f"✓ {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
