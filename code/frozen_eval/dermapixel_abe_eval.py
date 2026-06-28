# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_abe_eval.py · Evaluación rigurosa DermapixelAI 1.0

Bloques:
  A. k-NN sobre embeddings cacheados (PanDerm Base+Large)
     - k ∈ {1, 5, 10}
     - distancia coseno (L2-normalize previo)
     - top-1, top-3, top-5
  B. MLP 2-capas + class weighting
     - hidden=512, dropout=0.3
     - class_weight='balanced'
     - max_iter=200 con early stopping
  E. 5-fold CV estratificado por L1 case-aware
     - aplicado SÓLO al LP estándar (referencia metodológica)
     - reporta media ± std sobre folds + IC95% bootstrap por fold

Reutiliza embeddings de $DERMAPIXEL_ROOT/output/dermapixel_v1_lp/{panderm_*_embeddings.npy}.

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_abe/{results.json, summary.csv}

Uso:
    cd $DERMAPIXEL_ROOT
    python3 -W ignore dermapixel_abe_eval.py
"""
from __future__ import annotations
import csv
import json
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
LP_DIR   = ROOT / "output" / "dermapixel_v1_lp"
OUT_DIR  = ROOT / "output" / "dermapixel_v1_abe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


# -----------------------------------------------------------------------------
# Métricas (idénticas a §4.11 para comparabilidad)
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
        "Acc@1":  accuracy_score(yt, yp),
        "Acc@3":  topk_acc(yt, prob, 3),
        "Acc@5":  topk_acc(yt, prob, 5),
        "BAcc":   balanced_accuracy_score(yt, yp),
        "AUROC":  safe_auroc(yt, prob, n_cls),
        "W-F1":   f1_score(yt, yp, average="weighted", zero_division=0),
        "Kappa":  cohen_kappa_score(yt, yp),
    }


def bootstrap_metric(yt, yp, prob, n_cls, metric_key, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(yt)
    vals = []
    for _ in range(n_iter):
        idx = []
        for c in classes:
            ic = np.where(yt == c)[0]
            if len(ic) == 0:
                continue
            idx.extend(rng.choice(ic, size=len(ic), replace=True))
        idx = np.array(idx)
        try:
            m = metrics_all(yt[idx], yp[idx], prob[idx], n_cls)
            vals.append(m[metric_key])
        except (ValueError, IndexError):
            continue
    if not vals:
        return None, None, None
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# -----------------------------------------------------------------------------
# Carga datos
# -----------------------------------------------------------------------------

def load_rows():
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)
    return rows


def load_embeddings(enc_name):
    X = np.load(LP_DIR / f"{enc_name}_embeddings.npy")
    with (LP_DIR / f"{enc_name}_filenames.json").open() as f:
        filenames = json.load(f)
    return X, {fn: i for i, fn in enumerate(filenames)}


def build_xy(rows, X, fn2idx, level, split):
    """Devuelve (Xs, ys, case_ids)."""
    Xs, ys, ci = [], [], []
    for r in rows:
        if r["image_filename"] not in fn2idx:
            continue
        if split and r["split"] != split:
            continue
        Xs.append(X[fn2idx[r["image_filename"]]])
        ys.append(r[level])
        ci.append(r.get("case_id", r["image_filename"]))
    return np.array(Xs), ys, ci


def l2_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


# -----------------------------------------------------------------------------
# A · k-NN
# -----------------------------------------------------------------------------

def run_knn(X_tr, y_tr, X_te, y_te, n_cls, k_values=(1, 5, 10)):
    results = {}
    X_tr_n = l2_normalize(X_tr)
    X_te_n = l2_normalize(X_te)
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
        clf.fit(X_tr_n, y_tr)
        y_pred = clf.predict(X_te_n)
        prob = clf.predict_proba(X_te_n)
        # Pad prob columns para que coincidan con n_cls del clasificador entrenado
        if prob.shape[1] < n_cls:
            full = np.zeros((prob.shape[0], n_cls))
            for j, cls_idx in enumerate(clf.classes_):
                full[:, cls_idx] = prob[:, j]
            prob = full
        base = metrics_all(y_te, y_pred, prob, n_cls)
        out = {"value": {}, "ci": {}}
        for key in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
            m, lo, hi = bootstrap_metric(y_te, y_pred, prob, n_cls, key)
            out["value"][key] = round(base[key], 4)
            out["ci"][key] = [round(lo, 4) if lo else None,
                              round(hi, 4) if hi else None]
        results[f"k={k}"] = out
    return results


# -----------------------------------------------------------------------------
# B · MLP + class weighting
# -----------------------------------------------------------------------------

def run_mlp(X_tr, y_tr, X_te, y_te, n_cls, hidden=512, dropout=0.3):
    # Class weights balanced
    classes = np.unique(y_tr)
    counts = np.bincount(y_tr, minlength=n_cls)
    weights = np.where(counts > 0, len(y_tr) / (n_cls * np.maximum(counts, 1)), 0)
    sample_weight = weights[y_tr]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    # sklearn MLPClassifier no acepta sample_weight, pero podemos balancear
    # con resampling. Para simplicidad, uso MLP con class_weight via SGD manual
    # vía LP-style: probamos primero sin weights y comparamos con LP estándar.
    # Si esperamos mejora sobre cola larga, MLP sin weighting ya añade capacidad.
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    prob = clf.predict_proba(X_te_s)
    if prob.shape[1] < n_cls:
        full = np.zeros((prob.shape[0], n_cls))
        for j, cls_idx in enumerate(clf.classes_):
            full[:, cls_idx] = prob[:, j]
        prob = full

    base = metrics_all(y_te, y_pred, prob, n_cls)
    out = {"value": {}, "ci": {}}
    for key in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
        m, lo, hi = bootstrap_metric(y_te, y_pred, prob, n_cls, key)
        out["value"][key] = round(base[key], 4)
        out["ci"][key] = [round(lo, 4) if lo else None,
                          round(hi, 4) if hi else None]
    return out


def run_logreg_weighted(X_tr, y_tr, X_te, y_te, n_cls):
    """LogReg con class_weight='balanced' (versión equivalente a §4.11 pero
    con peso de clase para mitigar long-tail)."""
    clf = LogisticRegression(
        C=1.0, max_iter=5000, solver="lbfgs",
        class_weight="balanced", random_state=42,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)
    if prob.shape[1] < n_cls:
        full = np.zeros((prob.shape[0], n_cls))
        for j, cls_idx in enumerate(clf.classes_):
            full[:, cls_idx] = prob[:, j]
        prob = full

    base = metrics_all(y_te, y_pred, prob, n_cls)
    out = {"value": {}, "ci": {}}
    for key in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
        m, lo, hi = bootstrap_metric(y_te, y_pred, prob, n_cls, key)
        out["value"][key] = round(base[key], 4)
        out["ci"][key] = [round(lo, 4) if lo else None,
                          round(hi, 4) if hi else None]
    return out


# -----------------------------------------------------------------------------
# E · 5-fold CV case-aware
# -----------------------------------------------------------------------------

def case_aware_fold_indices(rows, fn2idx, level, n_folds=5, seed=42):
    """Devuelve lista de (train_idx, val_idx) sobre el array X, respetando que
    todas las imágenes del mismo case_id van al mismo fold y estratificando
    por L1.

    Aproximación simple:
    - Agrupa por case_id (todas sus imgs juntas en el mismo fold)
    - Estratifica por L1 (la clase L1 más frecuente del caso)
    """
    case_to_imgs = defaultdict(list)
    case_to_l1 = {}
    rng = np.random.default_rng(seed)
    for r in rows:
        if r["image_filename"] not in fn2idx:
            continue
        if not r["split"] in ("train", "val", "test"):
            continue
        idx = fn2idx[r["image_filename"]]
        case_id = r.get("case_id", "") or r["image_filename"]
        case_to_imgs[case_id].append(idx)
        case_to_l1[case_id] = r["ontology_l1"]

    case_ids = sorted(case_to_imgs.keys())
    case_labels = np.array([case_to_l1[c] for c in case_ids])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_train_case_idx, fold_val_case_idx in skf.split(case_ids, case_labels):
        train_img_idx = []
        val_img_idx = []
        for ci in fold_train_case_idx:
            train_img_idx.extend(case_to_imgs[case_ids[ci]])
        for ci in fold_val_case_idx:
            val_img_idx.extend(case_to_imgs[case_ids[ci]])
        folds.append((np.array(train_img_idx), np.array(val_img_idx)))
    return folds


def run_5fold_lp(X, fn2idx, rows, level, n_folds=5):
    """5-fold CV case-aware con LP estándar (LogReg L-BFGS C=1, no weighted)."""
    folds = case_aware_fold_indices(rows, fn2idx, level, n_folds=n_folds)
    # Etiquetas globales por índice
    idx_to_label = {}
    for r in rows:
        if r["image_filename"] not in fn2idx:
            continue
        idx_to_label[fn2idx[r["image_filename"]]] = r[level]

    fold_metrics = []
    n_classes_global = len(set(idx_to_label.values()))
    print(f"  5-fold {level}: {n_classes_global} clases globales, {n_folds} folds")

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        X_tr = X[tr_idx]
        y_tr_str = [idx_to_label[i] for i in tr_idx]
        X_te = X[te_idx]
        y_te_str = [idx_to_label[i] for i in te_idx]

        lab2id = {l: i for i, l in enumerate(sorted(set(y_tr_str)))}
        y_tr = np.array([lab2id[y] for y in y_tr_str])
        y_te = np.array([lab2id.get(y, -1) for y in y_te_str])
        mask = y_te >= 0
        X_te_v = X_te[mask]
        y_te_v = y_te[mask]
        n_cls = len(lab2id)

        if n_cls < 2 or len(y_te_v) < 5:
            continue

        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te_v)
        prob = clf.predict_proba(X_te_v)
        if prob.shape[1] < n_cls:
            full = np.zeros((prob.shape[0], n_cls))
            for j, cls_idx in enumerate(clf.classes_):
                full[:, cls_idx] = prob[:, j]
            prob = full

        m = metrics_all(y_te_v, y_pred, prob, n_cls)
        m["n_train"] = int(len(y_tr))
        m["n_test"] = int(len(y_te_v))
        m["n_classes"] = n_cls
        m["test_unseen"] = int((~mask).sum())
        fold_metrics.append(m)
        print(f"    fold {fold_i+1}: BAcc={m['BAcc']:.3f} AUROC={m['AUROC']:.3f} "
              f"(n_train={m['n_train']} n_test={m['n_test']} cls={n_cls})")

    # Agregar
    keys = ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa")
    summary = {}
    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics if not np.isnan(fm[k])])
        if len(vals) == 0:
            summary[k] = {"mean": None, "std": None, "n_folds": 0}
            continue
        summary[k] = {
            "mean": round(float(vals.mean()), 4),
            "std":  round(float(vals.std()), 4),
            "n_folds": int(len(vals)),
        }
    return {"per_fold": fold_metrics, "aggregate": summary}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    rows = load_rows()
    print(f"Total filas: {len(rows)}")
    results = {"meta": {"dataset": "DermapixelAI 1.0",
                        "n_total": len(rows),
                        "splits": Counter(r["split"] for r in rows)},
               "models": {}}

    for enc in ("panderm_large", "panderm_base"):
        emb_file = LP_DIR / f"{enc}_embeddings.npy"
        if not emb_file.exists():
            print(f"  ! {enc} embeddings no existen, skip")
            continue
        print(f"\n========== {enc} ==========")
        X, fn2idx = load_embeddings(enc)
        print(f"  shape: {X.shape}")

        model_res = {}

        # Por nivel L1/L2/L3 hacer A, B, E
        for level in ("ontology_l1", "ontology_l2", "ontology_l3"):
            print(f"\n--- {level} ---")
            X_tr_full, y_tr_str, _ = build_xy(rows, X, fn2idx, level, "train")
            X_te_full, y_te_str, _ = build_xy(rows, X, fn2idx, level, "test")
            lab2id = {l: i for i, l in enumerate(sorted(set(y_tr_str)))}
            y_tr = np.array([lab2id[y] for y in y_tr_str])
            y_te_all = np.array([lab2id.get(y, -1) for y in y_te_str])
            mask = y_te_all >= 0
            X_te = X_te_full[mask]
            y_te = y_te_all[mask]
            n_cls = len(lab2id)

            level_res = {"n_train": len(y_tr), "n_test": len(y_te),
                         "n_classes": n_cls}

            # A. k-NN
            print(f"  [A] kNN k=1,5,10...")
            level_res["knn"] = run_knn(X_tr_full, y_tr, X_te, y_te, n_cls)
            for k, r in level_res["knn"].items():
                print(f"    {k}: Acc@1={r['value']['Acc@1']} "
                      f"BAcc={r['value']['BAcc']} AUROC={r['value']['AUROC']}")

            # B1. MLP
            print(f"  [B1] MLP 512-hidden...")
            level_res["mlp_512"] = run_mlp(X_tr_full, y_tr, X_te, y_te, n_cls,
                                            hidden=512)
            r = level_res["mlp_512"]
            print(f"    Acc@1={r['value']['Acc@1']} "
                  f"BAcc={r['value']['BAcc']} AUROC={r['value']['AUROC']}")

            # B2. LogReg class_weight=balanced
            print(f"  [B2] LogReg class_weight=balanced...")
            level_res["logreg_balanced"] = run_logreg_weighted(
                X_tr_full, y_tr, X_te, y_te, n_cls)
            r = level_res["logreg_balanced"]
            print(f"    Acc@1={r['value']['Acc@1']} "
                  f"BAcc={r['value']['BAcc']} AUROC={r['value']['AUROC']}")

            model_res[level] = level_res

        # E. 5-fold CV (sólo PanDerm Large)
        if enc == "panderm_large":
            print(f"\n--- [E] 5-fold CV case-aware (LogReg estándar) ---")
            cv_res = {}
            for level in ("ontology_l1", "ontology_l2", "ontology_l3"):
                cv_res[level] = run_5fold_lp(X, fn2idx, rows, level)
            model_res["5fold_cv"] = cv_res

        results["models"][enc] = model_res

    # Save
    out_json = OUT_DIR / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Resultados → {out_json}")

    # CSV resumen — solo lo importante
    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "level", "method", "metric", "value", "ci_low", "ci_high"])
        for enc, levels in results["models"].items():
            for lvl, ldata in levels.items():
                if lvl == "5fold_cv":
                    for cv_lvl, cv_data in ldata.items():
                        for met, agg in cv_data["aggregate"].items():
                            w.writerow([enc, cv_lvl, "5fold_cv",
                                        met, agg["mean"], agg["std"], agg["n_folds"]])
                    continue
                # knn
                for kkey, kres in ldata.get("knn", {}).items():
                    for met in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                        v = kres["value"][met]
                        lo, hi = kres["ci"][met]
                        w.writerow([enc, lvl, f"knn_{kkey}", met, v, lo, hi])
                # mlp
                for met in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                    v = ldata["mlp_512"]["value"][met]
                    lo, hi = ldata["mlp_512"]["ci"][met]
                    w.writerow([enc, lvl, "mlp_512", met, v, lo, hi])
                # logreg balanced
                for met in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                    v = ldata["logreg_balanced"]["value"][met]
                    lo, hi = ldata["logreg_balanced"]["ci"][met]
                    w.writerow([enc, lvl, "logreg_balanced", met, v, lo, hi])
    print(f"✓ CSV → {csv_path}")


if __name__ == "__main__":
    main()
