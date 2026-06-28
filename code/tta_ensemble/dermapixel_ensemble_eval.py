# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_ensemble_eval.py · Ensemble PanDerm Base+Large+DermLIP v2

Estrategias evaluadas:
  - LP individual sobre cada encoder (referencia)
  - avg: promedio de softmax sobre los 3 LPs (clásico)
  - max: máx por clase (útil para safety-screen, recall-oriented)
  - stack: meta-LP entrenado sobre concatenación de las 3 probabilidades

Para DermLIP v2 se reutilizan embeddings cacheados si existen; si no, se
extraen aquí (~30s).

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_ensemble/{results.json, summary.csv}
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
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
LP_DIR   = ROOT / "output" / "dermapixel_v1_lp"
OUT_DIR  = ROOT / "output" / "dermapixel_v1_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


# -----------------------------------------------------------------------------
# Métricas (idénticas)
# -----------------------------------------------------------------------------

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
    if prob.shape[1] < k:
        return 1.0
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


def bootstrap_metric(yt, yp, prob, n_cls, metric_key, n_iter=1000, seed=42):
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
            vals.append(m[metric_key])
        except (ValueError, IndexError):
            continue
    if not vals: return None, None, None
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def pack_metrics(y_te, y_pred, prob, n_cls):
    base = metrics_all(y_te, y_pred, prob, n_cls)
    out = {"value": {}, "ci": {}}
    for k in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
        m, lo, hi = bootstrap_metric(y_te, y_pred, prob, n_cls, k)
        out["value"][k] = round(base[k], 4)
        out["ci"][k] = [round(lo, 4) if lo else None, round(hi, 4) if hi else None]
    return out


# -----------------------------------------------------------------------------
# Embeddings DermLIP v2 (extraer si no están)
# -----------------------------------------------------------------------------

def get_dermlip_embeddings(rows):
    """Devuelve (X, filenames) con embeddings del visual encoder DermLIP v2."""
    cache_emb = LP_DIR / "dermlip_v2_embeddings.npy"
    cache_fn = LP_DIR / "dermlip_v2_filenames.json"
    if cache_emb.exists() and cache_fn.exists():
        print("  DermLIP v2: reutilizando cache")
        X = np.load(cache_emb)
        with cache_fn.open() as f:
            return X, json.load(f)

    print("  DermLIP v2: extrayendo embeddings (no en cache)...")
    derm1m_src = os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "dermfm_zero/src")
    if derm1m_src not in sys.path:
        sys.path.insert(0, derm1m_src)
    for m in [k for k in list(sys.modules) if k.startswith("open_clip")]:
        del sys.modules[m]
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256")
    model.eval().to(DEVICE)

    embeddings, filenames = [], []
    t0 = time.time()
    batch_imgs, batch_fns = [], []
    for i, r in enumerate(rows):
        img_path = DATASET / r["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(preprocess(img))
            batch_fns.append(r["image_filename"])
        except Exception:
            continue
        if len(batch_imgs) >= 32 or i == len(rows) - 1:
            if not batch_imgs: continue
            with torch.no_grad():
                feats = model.encode_image(torch.stack(batch_imgs).to(DEVICE))
            embeddings.append(feats.cpu().numpy())
            filenames.extend(batch_fns)
            batch_imgs, batch_fns = [], []
    X = np.vstack(embeddings)
    print(f"  shape: {X.shape} en {time.time()-t0:.1f}s")
    np.save(cache_emb, X)
    with cache_fn.open("w") as f:
        json.dump(filenames, f)
    del model
    torch.cuda.empty_cache()
    return X, filenames


# -----------------------------------------------------------------------------
# Helpers ensemble
# -----------------------------------------------------------------------------

def fit_lp_get_probs(X_tr, y_tr, X_te, n_cls):
    """Entrena LP estándar y devuelve (prob_train, prob_test, clf)."""
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)
    prob_tr = clf.predict_proba(X_tr)
    prob_te = clf.predict_proba(X_te)
    # Padding por si alguna clase no aparece en train (defensivo)
    for prob in (prob_tr, prob_te):
        if prob.shape[1] < n_cls:
            full = np.zeros((prob.shape[0], n_cls))
            for j, ci in enumerate(clf.classes_):
                full[:, ci] = prob[:, j]
            prob[:] = full[:prob.shape[0]]
    return prob_tr, prob_te


def stack_meta_lp(prob_tr_list, y_tr, prob_te_list, n_cls):
    """Stacking: concatena probs de los 3 LPs como features de un meta-LP."""
    Xs_tr = np.hstack(prob_tr_list)
    Xs_te = np.hstack(prob_te_list)
    meta = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
    meta.fit(Xs_tr, y_tr)
    prob = meta.predict_proba(Xs_te)
    if prob.shape[1] < n_cls:
        full = np.zeros((prob.shape[0], n_cls))
        for j, ci in enumerate(meta.classes_):
            full[:, ci] = prob[:, j]
        prob = full
    return prob


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)
    print(f"Total filas: {len(rows)}")

    # Cargar embeddings de los 3 encoders
    print("\n=== Cargando embeddings ===")
    X_base = np.load(LP_DIR / "panderm_base_embeddings.npy")
    with (LP_DIR / "panderm_base_filenames.json").open() as f:
        fn_base = json.load(f)
    fn2idx_base = {fn: i for i, fn in enumerate(fn_base)}
    print(f"  PanDerm Base: {X_base.shape}")

    X_large = np.load(LP_DIR / "panderm_large_embeddings.npy")
    with (LP_DIR / "panderm_large_filenames.json").open() as f:
        fn_large = json.load(f)
    fn2idx_large = {fn: i for i, fn in enumerate(fn_large)}
    print(f"  PanDerm Large: {X_large.shape}")

    X_dlip, fn_dlip = get_dermlip_embeddings(rows)
    fn2idx_dlip = {fn: i for i, fn in enumerate(fn_dlip)}
    print(f"  DermLIP v2: {X_dlip.shape}")

    results = {"meta": {"n_total": len(rows)}, "levels": {}}

    for level in ("ontology_l1", "ontology_l2", "ontology_l3"):
        print(f"\n========== {level} ==========")
        # Train + test indices y labels
        tr_rows = [r for r in rows if r["split"] == "train"]
        te_rows = [r for r in rows if r["split"] == "test"]

        # Filtrar filas que tengan filename en los 3 encoders
        def in_all(r):
            f = r["image_filename"]
            return (f in fn2idx_base) and (f in fn2idx_large) and (f in fn2idx_dlip)
        tr_rows = [r for r in tr_rows if in_all(r)]
        te_rows = [r for r in te_rows if in_all(r)]

        lab2id = {l: i for i, l in enumerate(sorted({r[level] for r in tr_rows}))}
        n_cls = len(lab2id)

        # Build matrices per encoder
        Xb_tr = np.array([X_base[fn2idx_base[r["image_filename"]]] for r in tr_rows])
        Xb_te = np.array([X_base[fn2idx_base[r["image_filename"]]] for r in te_rows])
        Xl_tr = np.array([X_large[fn2idx_large[r["image_filename"]]] for r in tr_rows])
        Xl_te = np.array([X_large[fn2idx_large[r["image_filename"]]] for r in te_rows])
        Xd_tr = np.array([X_dlip[fn2idx_dlip[r["image_filename"]]] for r in tr_rows])
        Xd_te = np.array([X_dlip[fn2idx_dlip[r["image_filename"]]] for r in te_rows])

        y_tr = np.array([lab2id[r[level]] for r in tr_rows])
        y_te_all = np.array([lab2id.get(r[level], -1) for r in te_rows])
        mask = y_te_all >= 0
        y_te = y_te_all[mask]
        Xb_te, Xl_te, Xd_te = Xb_te[mask], Xl_te[mask], Xd_te[mask]

        print(f"  train={len(y_tr)} test={len(y_te)} classes={n_cls}")

        # LPs individuales
        prob_b_tr, prob_b_te = fit_lp_get_probs(Xb_tr, y_tr, Xb_te, n_cls)
        prob_l_tr, prob_l_te = fit_lp_get_probs(Xl_tr, y_tr, Xl_te, n_cls)
        prob_d_tr, prob_d_te = fit_lp_get_probs(Xd_tr, y_tr, Xd_te, n_cls)

        level_res = {"n_train": len(y_tr), "n_test": len(y_te), "n_classes": n_cls,
                     "methods": {}}

        # Individuales
        for name, prob in (("base", prob_b_te), ("large", prob_l_te), ("dermlip", prob_d_te)):
            y_pred = prob.argmax(axis=1)
            level_res["methods"][f"lp_{name}"] = pack_metrics(y_te, y_pred, prob, n_cls)

        # Avg de los 3
        prob_avg = (prob_b_te + prob_l_te + prob_d_te) / 3.0
        y_pred_avg = prob_avg.argmax(axis=1)
        level_res["methods"]["avg_3"] = pack_metrics(y_te, y_pred_avg, prob_avg, n_cls)

        # Avg sólo Large + DermLIP (excluir Base por ser peor)
        prob_avg_ld = (prob_l_te + prob_d_te) / 2.0
        y_pred_ld = prob_avg_ld.argmax(axis=1)
        level_res["methods"]["avg_large_dermlip"] = pack_metrics(y_te, y_pred_ld, prob_avg_ld, n_cls)

        # Max de los 3 (por clase)
        prob_max = np.maximum.reduce([prob_b_te, prob_l_te, prob_d_te])
        # Re-normalizar
        prob_max = prob_max / prob_max.sum(axis=1, keepdims=True)
        y_pred_max = prob_max.argmax(axis=1)
        level_res["methods"]["max_3"] = pack_metrics(y_te, y_pred_max, prob_max, n_cls)

        # Stacking
        prob_stack = stack_meta_lp(
            [prob_b_tr, prob_l_tr, prob_d_tr], y_tr,
            [prob_b_te, prob_l_te, prob_d_te], n_cls)
        y_pred_stack = prob_stack.argmax(axis=1)
        level_res["methods"]["stack_3"] = pack_metrics(y_te, y_pred_stack, prob_stack, n_cls)

        # Print
        for name, res in level_res["methods"].items():
            v = res["value"]
            print(f"  {name:20s} Acc@1={v['Acc@1']:.3f} BAcc={v['BAcc']:.3f} "
                  f"AUROC={v['AUROC']:.3f} Acc@3={v['Acc@3']:.3f}")

        results["levels"][level] = level_res

    # Save
    out_json = OUT_DIR / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ {out_json}")

    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "method", "metric", "value", "ci_low", "ci_high"])
        for lvl, ldata in results["levels"].items():
            for mname, mres in ldata["methods"].items():
                for met in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                    w.writerow([lvl, mname, met, mres["value"][met],
                                mres["ci"][met][0], mres["ci"][met][1]])
    print(f"✓ {csv_path}")


if __name__ == "__main__":
    main()
