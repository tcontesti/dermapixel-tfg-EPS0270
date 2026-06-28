# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_focal_loss.py · Focal loss puro sobre LP de embeddings cacheados

Compara:
  - LP CE estándar (§4.11)
  - LP CE con class_weight='balanced' (§4.11 ya hecho)
  - LP Focal loss γ ∈ {1.0, 2.0, 5.0} (este experimento)

Focal: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

Implementación: optimización manual con torch sobre embeddings cacheados.
Sobre 3 niveles L1/L2/L3 × PanDerm Base/Large.

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_focal/{results.json, summary.csv}
"""
from __future__ import annotations
import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
LP_DIR   = ROOT / "output" / "dermapixel_v1_lp"
OUT_DIR  = ROOT / "output" / "dermapixel_v1_focal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


# Focal loss multi-clase
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def safe_auroc(yt, prob, n_cls):
    aurocs = []
    for c in range(n_cls):
        if (yt == c).sum() == 0 or (yt != c).sum() == 0:
            continue
        try:
            aurocs.append(roc_auc_score((yt == c).astype(int), prob[:, c]))
        except ValueError:
            continue
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
        except (ValueError, IndexError):
            continue
    if not vals: return None, None, None
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def train_lp_focal(X_tr, y_tr, X_te, y_te, n_cls, gamma=2.0, use_class_weight=False,
                    epochs=300, lr=0.1):
    """LP equivalente con Focal loss usando torch optimization."""
    Xt = torch.from_numpy(X_tr).float().to(DEVICE)
    yt = torch.from_numpy(y_tr).long().to(DEVICE)
    Xe = torch.from_numpy(X_te).float().to(DEVICE)

    head = nn.Linear(X_tr.shape[1], n_cls).to(DEVICE)
    optim = torch.optim.LBFGS(head.parameters(), lr=lr, max_iter=epochs,
                               tolerance_grad=1e-6, tolerance_change=1e-9)

    if use_class_weight:
        counts = np.bincount(y_tr, minlength=n_cls)
        w = np.where(counts > 0, len(y_tr) / (n_cls * np.maximum(counts, 1)), 0.0)
        alpha = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    else:
        alpha = None

    loss_fn = FocalLoss(gamma=gamma, alpha=alpha)

    def closure():
        optim.zero_grad()
        logits = head(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        return loss
    optim.step(closure)

    with torch.no_grad():
        logits = head(Xe)
        prob = F.softmax(logits, dim=-1).cpu().numpy()
    y_pred = prob.argmax(axis=1)
    return y_pred, prob


def main():
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)
    print(f"Filas: {len(rows)}")

    results = {"models": {}}

    for enc in ("panderm_large", "panderm_base"):
        emb_path = LP_DIR / f"{enc}_embeddings.npy"
        if not emb_path.exists():
            print(f"  ! {enc} no encontrado, skip")
            continue
        print(f"\n========== {enc} ==========")
        X = np.load(emb_path)
        with (LP_DIR / f"{enc}_filenames.json").open() as f:
            filenames = json.load(f)
        fn2idx = {fn: i for i, fn in enumerate(filenames)}

        model_res = {}
        for level in ("ontology_l1", "ontology_l2", "ontology_l3"):
            print(f"\n--- {level} ---")
            X_tr, y_tr, X_te, y_te = [], [], [], []
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
            print(f"  train={len(y_tr)} test={len(y_te)} classes={n_cls}")

            level_res = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                         "n_classes": int(n_cls), "variants": {}}

            for gamma in (1.0, 2.0, 5.0):
                for cw in (False, True):
                    name = f"focal_g{gamma}{'_balanced' if cw else ''}"
                    y_pred, prob = train_lp_focal(X_tr, y_tr, X_te, y_te, n_cls,
                                                   gamma=gamma, use_class_weight=cw)
                    base = metrics_all(y_te, y_pred, prob, n_cls)
                    out = {"value": {}, "ci": {}}
                    for k in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                        m, lo, hi = bootstrap_metric(y_te, y_pred, prob, n_cls, k)
                        out["value"][k] = round(base[k], 4)
                        out["ci"][k] = [round(lo, 4) if lo else None, round(hi, 4) if hi else None]
                    level_res["variants"][name] = out
                    print(f"    {name:25s} Acc@1={base['Acc@1']:.3f} BAcc={base['BAcc']:.3f} AUROC={base['AUROC']:.3f}")

            model_res[level] = level_res

        results["models"][enc] = model_res

    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "level", "variant", "metric", "value", "ci_low", "ci_high"])
        for enc, levels in results["models"].items():
            for lvl, ldata in levels.items():
                for v, vres in ldata["variants"].items():
                    for met in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                        w.writerow([enc, lvl, v, met, vres["value"][met],
                                    vres["ci"][met][0], vres["ci"][met][1]])
    print(f"\n✓ {OUT_DIR / 'results.json'}")
    print(f"✓ {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
