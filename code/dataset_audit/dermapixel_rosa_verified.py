# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_rosa_verified.py · Comparación rosa_verified=True vs False

Una vez calculados los embeddings PanDerm Base/Large (reutilizables desde
output/dermapixel_v1_lp/{name}_embeddings.npy + _filenames.json), se entrena
LP en dos modos:

A) Solo subconjunto rosa_verified=True
B) Subconjunto completo (True+False)

…sobre los mismos splits de train/test del CSV. Output:
  $DERMAPIXEL_ROOT/output/dermapixel_v1_rosa/summary.csv + report.md

Uso:
    cd $DERMAPIXEL_ROOT
    python3 dermapixel_rosa_verified.py
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, cohen_kappa_score, roc_auc_score,
)
from sklearn.utils import resample

ROOT       = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET    = ROOT / "datasets" / "dermapixel_v1" / "dataset_filtered.csv"
LP_DIR     = ROOT / "output" / "dermapixel_v1_lp"
OUT_DIR    = ROOT / "output" / "dermapixel_v1_rosa"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stratified_bootstrap(y_true, y_pred, y_prob, n_classes, fn, n_iter=1000, seed=42):
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
            vals.append(fn(y_true[idx], y_pred[idx], y_prob[idx], n_classes))
        except (ValueError, IndexError):
            continue
    if not vals:
        return None, None, None
    vals = np.array(vals)
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics(y_true, y_pred, y_prob, n_classes):
    out = {
        "Acc": accuracy_score(y_true, y_pred),
        "BAcc": balanced_accuracy_score(y_true, y_pred),
        "W-F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
    }
    try:
        if n_classes == 2:
            out["AUROC"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            out["AUROC"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=list(range(n_classes)),
            )
    except (ValueError, IndexError):
        out["AUROC"] = float("nan")
    return out


def run_lp(X_tr, y_tr, X_te, y_te, n_classes):
    if len(np.unique(y_tr)) < 2:
        return None
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)
    base = metrics(y_te, y_pred, y_prob, n_classes)

    fns = {
        "Acc":   lambda yt, yp, pp, n: accuracy_score(yt, yp),
        "BAcc":  lambda yt, yp, pp, n: balanced_accuracy_score(yt, yp),
        "W-F1":  lambda yt, yp, pp, n: f1_score(yt, yp, average="weighted", zero_division=0),
        "Kappa": lambda yt, yp, pp, n: cohen_kappa_score(yt, yp),
        "AUROC": lambda yt, yp, pp, n: metrics(yt, yp, pp, n)["AUROC"],
    }
    res = {}
    for name, fn in fns.items():
        mean, lo, hi = stratified_bootstrap(y_te, y_pred, y_prob, n_classes, fn)
        res[name] = {"value": base[name], "ci_low": lo, "ci_high": hi}
    return res


def main():
    # 1 · Cargar CSV
    rows = []
    with DATASET.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"Total filas: {len(rows)}")

    rosa_t = sum(1 for r in rows if r.get("rosa_verified", "").lower() == "true")
    rosa_f = sum(1 for r in rows if r.get("rosa_verified", "").lower() == "false")
    print(f"rosa_verified True={rosa_t} False={rosa_f}")

    # 2 · Para cada encoder, cargar embeddings y correr LP en 2 escenarios
    summary = [["model", "subset", "level", "n_train", "n_test", "n_classes",
                "metric", "value", "ci_low", "ci_high"]]

    for enc in ("panderm_large", "panderm_base"):
        emb_path = LP_DIR / f"{enc}_embeddings.npy"
        fn_path  = LP_DIR / f"{enc}_filenames.json"
        if not emb_path.exists():
            print(f"  ! embeddings {enc} no existen, skip")
            continue
        X = np.load(emb_path)
        with fn_path.open() as f:
            filenames = json.load(f)
        fn2idx = {fn: i for i, fn in enumerate(filenames)}
        print(f"\n{enc} | embedding shape={X.shape}")

        for subset_name, subset_filter in (
            ("rosa_true_only", lambda r: r.get("rosa_verified", "").lower() == "true"),
            ("all_verified",   lambda r: True),
        ):
            for lvl in ("ontology_l1", "ontology_l2", "ontology_l3"):
                X_tr, y_tr, X_te, y_te = [], [], [], []
                for r in rows:
                    if not subset_filter(r):
                        continue
                    if r["image_filename"] not in fn2idx:
                        continue
                    xi = X[fn2idx[r["image_filename"]]]
                    yi = r[lvl]
                    if r["split"] == "train":
                        X_tr.append(xi); y_tr.append(yi)
                    elif r["split"] == "test":
                        X_te.append(xi); y_te.append(yi)

                if not X_tr or not X_te:
                    print(f"  {subset_name}/{lvl}: insuficientes datos (tr={len(X_tr)} te={len(X_te)})")
                    continue

                X_tr = np.array(X_tr); X_te = np.array(X_te)
                lab2id = {l: i for i, l in enumerate(sorted(set(y_tr)))}
                y_tr_i = np.array([lab2id[y] for y in y_tr])
                y_te_i = np.array([lab2id.get(y, -1) for y in y_te])
                mask = y_te_i >= 0
                X_te_v = X_te[mask]; y_te_v = y_te_i[mask]
                n_cls = len(lab2id)

                print(f"  {subset_name}/{lvl}: train={len(y_tr_i)} test={len(y_te_v)} cls={n_cls}")
                res = run_lp(X_tr, y_tr_i, X_te_v, y_te_v, n_cls)
                if res is None:
                    continue
                for met, vals in res.items():
                    summary.append([
                        enc, subset_name, lvl, len(y_tr_i), len(y_te_v), n_cls,
                        met,
                        round(vals["value"], 4) if vals["value"] is not None and not np.isnan(vals["value"]) else "nan",
                        round(vals["ci_low"], 4)  if vals["ci_low"]  is not None else "nan",
                        round(vals["ci_high"], 4) if vals["ci_high"] is not None else "nan",
                    ])

    # 3 · Output
    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary)
    print(f"\nResumen: {csv_path}")


if __name__ == "__main__":
    main()
