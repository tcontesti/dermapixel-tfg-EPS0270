# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_faiss_knn.py · N3 · FAISS k-NN retrieval sobre DermapixelAI

Replica módulo M4 del prototipo de producción (RAG sobre Derm1M) sobre
el corpus DermapixelAI 1.0:
  - Indexar embeddings PanDerm Large train (874 imgs) en FAISS IndexFlatIP
    (L2-normalize + producto interno = similitud coseno)
  - Para cada test img: top-k vecinos
  - Predicción: (a) mayoría top-1; (b) majority voting top-5/10;
                (c) weighted softmax con similitud cosénica

Comparativa con LP §4.11 y SpanDerm v0 § 4.14 para los 3 niveles.

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_faiss/{results.json, summary.csv}
"""
from __future__ import annotations
import csv
import json
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import faiss
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
LP_DIR   = ROOT / "output" / "dermapixel_v1_lp"
OUT_DIR  = ROOT / "output" / "dermapixel_v1_faiss"
OUT_DIR.mkdir(parents=True, exist_ok=True)
L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


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


def pack(yt, yp, prob, n_cls):
    base = metrics_all(yt, yp, prob, n_cls)
    out = {"value": {}, "ci": {}}
    for k in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
        m, lo, hi = bootstrap_metric(yt, yp, prob, n_cls, k)
        out["value"][k] = round(base[k], 4)
        out["ci"][k] = [round(lo, 4) if lo else None, round(hi, 4) if hi else None]
    return out


def predict_majority(neighbors_idx, y_train, n_cls):
    """top-k majority voting con prob estimada por frecuencia."""
    n_test = neighbors_idx.shape[0]
    prob = np.zeros((n_test, n_cls))
    y_pred = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        votes = Counter(y_train[neighbors_idx[i]])
        total = sum(votes.values())
        for c, n in votes.items(): prob[i, c] = n / total
        y_pred[i] = votes.most_common(1)[0][0]
    return y_pred, prob


def predict_weighted(neighbors_idx, neighbors_sim, y_train, n_cls):
    """Weighted softmax sobre similitud coseno por clase."""
    n_test = neighbors_idx.shape[0]
    prob = np.zeros((n_test, n_cls))
    y_pred = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        # Softmax con tau=10 (típico en CLIP-style)
        sims = neighbors_sim[i]
        w = np.exp(10 * (sims - sims.max()))
        w = w / w.sum()
        labels = y_train[neighbors_idx[i]]
        for j, c in enumerate(labels):
            prob[i, c] += w[j]
        y_pred[i] = prob[i].argmax()
    return y_pred, prob


def main():
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)

    print("=== Cargar PanDerm Large embeddings ===")
    X = np.load(LP_DIR / "panderm_large_embeddings.npy")
    with (LP_DIR / "panderm_large_filenames.json").open() as f:
        filenames = json.load(f)
    fn2idx = {fn: i for i, fn in enumerate(filenames)}
    print(f"  shape: {X.shape}")

    # L2-normalize
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    results = {"levels": {}}

    for level in ("ontology_l1", "ontology_l2", "ontology_l3"):
        print(f"\n========== {level} ==========")
        tr_rows = [r for r in rows if r["split"] == "train" and r["image_filename"] in fn2idx]
        te_rows = [r for r in rows if r["split"] == "test"  and r["image_filename"] in fn2idx]
        classes = sorted({r[level] for r in tr_rows})
        lab2id = {l: i for i, l in enumerate(classes)}
        n_cls = len(lab2id)

        X_tr = np.array([X_norm[fn2idx[r["image_filename"]]] for r in tr_rows]).astype(np.float32)
        X_te = np.array([X_norm[fn2idx[r["image_filename"]]] for r in te_rows]).astype(np.float32)
        y_tr = np.array([lab2id[r[level]] for r in tr_rows])
        y_te_all = np.array([lab2id.get(r[level], -1) for r in te_rows])
        mask = y_te_all >= 0
        X_te = X_te[mask]; y_te = y_te_all[mask]
        print(f"  train={len(y_tr)} test={len(y_te)} classes={n_cls}")

        # Build FAISS index
        index = faiss.IndexFlatIP(X_tr.shape[1])
        index.add(X_tr)
        print(f"  FAISS IndexFlatIP, ntotal={index.ntotal}")

        level_res = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                     "n_classes": int(n_cls), "variants": {}}

        # Probar varios k
        for k in (1, 5, 10, 20):
            sims, nbr = index.search(X_te, k)  # sims=(n_test, k) cos similitud
            # (a) Majority voting
            y_pred_m, prob_m = predict_majority(nbr, y_tr, n_cls)
            r_m = pack(y_te, y_pred_m, prob_m, n_cls)
            level_res["variants"][f"k={k}_majority"] = r_m
            # (b) Weighted softmax
            y_pred_w, prob_w = predict_weighted(nbr, sims, y_tr, n_cls)
            r_w = pack(y_te, y_pred_w, prob_w, n_cls)
            level_res["variants"][f"k={k}_weighted"] = r_w
            print(f"  k={k:2d} majority: Acc@1={r_m['value']['Acc@1']:.3f} BAcc={r_m['value']['BAcc']:.3f} AUROC={r_m['value']['AUROC']:.3f}")
            print(f"  k={k:2d} weighted: Acc@1={r_w['value']['Acc@1']:.3f} BAcc={r_w['value']['BAcc']:.3f} AUROC={r_w['value']['AUROC']:.3f}")

        results["levels"][level] = level_res

    # Save
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "variant", "metric", "value", "ci_low", "ci_high"])
        for lvl, ldata in results["levels"].items():
            for v, vres in ldata["variants"].items():
                for met in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                    w.writerow([lvl, v, met, vres["value"][met],
                                vres["ci"][met][0], vres["ci"][met][1]])
    print(f"\n✓ {OUT_DIR / 'results.json'}")
    print(f"✓ {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
