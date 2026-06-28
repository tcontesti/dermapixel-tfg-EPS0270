# Material reproducible del TFG EPS0270 — DermapixelAI. Pesos y datasets de
# terceros NO incluidos (ver licencias originales). Rutas por variable de
# entorno DERMAPIXEL_ROOT / DERMAPIXEL_OUT.
"""
dermapixel_aug_oof.py · Ablación R0-AUG (resultado complementario)
¿Mejora la head L2 (mlp_512) si se aumenta SOLO el train y se re-extraen
embeddings del encoder PanDerm-Large CONGELADO?  Veredicto: FLAT (sin mejora).

Protocolo:
  - Señal principal = 5-fold CV OOF case-aware sobre los 1062, head mlp_512.
  - Comparación OOF-vs-OOF: baseline NO-aug en el MISMO 5-fold vs N=3/5/10.
  - Augmentation SOLO dentro del train de cada fold; re-extraccion por fold.
  - El fold held-out se evalua SIEMPRE con su embedding original (sin aug).
  - Folds case-aware: ningun case_id en train y held-out a la vez (assert).
  - mlp_512 identica al baseline (StandardScaler fit-en-train, hidden=512,
    alpha=1e-4, lr=1e-3, max_iter=300, early_stopping, seed=42). SIN class weights.
  - Test-36 fijo = referencia SECUNDARIA (ruidoso: 36 imgs / 38 clases).

Salida: $DERMAPIXEL_OUT/dermapixel_v1_aug/{oof_results.json, leakage_sanity.json}

Uso:
    DERMAPIXEL_ROOT=./data DERMAPIXEL_OUT=./outputs python3 -W ignore dermapixel_aug_oof.py [--smoke]
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
OUT_ROOT = Path(os.environ.get("DERMAPIXEL_OUT", "./outputs"))
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "classification"))
import dermapixel_abe_eval as abe   # metrics + folds + load
import dermapixel_lp_eval as lp     # encoder + extract

DATASET_DIR = ROOT / "datasets" / "dermapixel_v1"
LP_DIR = OUT_ROOT / "dermapixel_v1_lp"
OUT = OUT_ROOT / "dermapixel_v1_aug"
OUT.mkdir(parents=True, exist_ok=True)

LEVEL = "ontology_l2"
N_LIST = [3, 5, 10]
MAXN = 10
KEYS = ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- derm-safe augmentation (color es diagnostico: jitter suave, sin hue/sat) ----
AUG_T = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),   # NO hue, NO saturation
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def fit_mlp_global(X_tr, y_tr_g, X_te, n_global):
    """mlp_512 identica al baseline; opera en espacio de etiquetas GLOBAL.
    Devuelve (y_pred_global, prob[len(X_te), n_global])."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = MLPClassifier(
        hidden_layer_sizes=(512,), activation="relu", alpha=1e-4,
        learning_rate_init=1e-3, max_iter=300, early_stopping=True,
        validation_fraction=0.1, random_state=42,
    )
    clf.fit(X_tr_s, y_tr_g)
    prob_local = clf.predict_proba(X_te_s)
    prob = np.zeros((len(X_te), n_global), dtype=np.float64)
    for j, c in enumerate(clf.classes_):
        prob[:, c] = prob_local[:, j]
    y_pred = clf.classes_[np.argmax(prob_local, axis=1)]
    return y_pred, prob


def extract_aug_copies(model, rows_subset, n_copies, base_seed, batch_size=64, tag=""):
    """Re-extrae n_copies embeddings AUMENTADOS por imagen (encoder congelado).
    Devuelve dict filename -> ndarray [n_copies, D]. Determinista por base_seed."""
    out = {r["image_filename"]: [] for r in rows_subset}
    t0 = time.time()
    for c in range(n_copies):
        torch.manual_seed(base_seed * 1000 + c)
        np.random.seed(base_seed * 1000 + c)
        with torch.no_grad():
            batch_t, batch_fn = [], []
            for i, r in enumerate(rows_subset):
                img = Image.open(DATASET_DIR / r["image_path"]).convert("RGB")
                batch_t.append(AUG_T(img)); batch_fn.append(r["image_filename"])
                if len(batch_t) >= batch_size or i == len(rows_subset) - 1:
                    X = torch.stack(batch_t).to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        f = model(X)
                    f = f.float().cpu().numpy()
                    for k, fn in enumerate(batch_fn):
                        out[fn].append(f[k])
                    batch_t, batch_fn = [], []
        print(f"    [{tag}] copia {c+1}/{n_copies} ({time.time()-t0:.0f}s)", flush=True)
    return {fn: np.stack(v) for fn, v in out.items()}


def metrics_row(y_true, y_pred, prob, n_cls):
    m = abe.metrics_all(y_true, y_pred, prob, n_cls)
    return {k: round(float(m[k]), 4) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 fold, N=3, subconjunto")
    args = ap.parse_args()

    rows = abe.load_rows()  # usa dataset_filtered.csv + L2_CANON
    X = np.load(LP_DIR / "panderm_large_embeddings.npy")
    fns = json.load(open(LP_DIR / "panderm_large_filenames.json"))
    fn2idx = {f: i for i, f in enumerate(fns)}

    # mapas por indice de embedding
    idx_row, idx_label, idx_case, idx_split = {}, {}, {}, {}
    for r in rows:
        if r["image_filename"] not in fn2idx:
            continue
        i = fn2idx[r["image_filename"]]
        idx_row[i] = r; idx_label[i] = r[LEVEL]
        idx_case[i] = r.get("case_id", "") or r["image_filename"]
        idx_split[i] = r["split"]

    classes = sorted(set(idx_label.values()))
    lab2id = {l: k for k, l in enumerate(classes)}
    n_cls = len(classes)
    print(f"Clases L2 globales: {n_cls}  | imgs: {len(idx_label)}")

    model, _ = lp.build_encoder("panderm_large")

    n_list = [3] if args.smoke else N_LIST
    maxn = max(n_list)

    # =====================================================================
    # A. OOF 5-fold case-aware (senal principal)
    # =====================================================================
    folds = abe.case_aware_fold_indices(rows, fn2idx, LEVEL, n_folds=5, seed=42)
    if args.smoke:
        folds = folds[:1]

    all_idx = sorted(idx_label.keys())
    y_global = np.array([lab2id[idx_label[i]] for i in range(len(fns))])

    # OOF prob containers (relleno solo en held-out)
    variants = ["baseline"] + [f"N={n}" for n in n_list]
    oof_prob = {v: np.zeros((len(fns), n_cls)) for v in variants}
    oof_pred = {v: -np.ones(len(fns), dtype=int) for v in variants}
    oof_filled = {v: np.zeros(len(fns), dtype=bool) for v in variants}

    leak = {"folds": [], "test36": {}}
    fold_sizes = []

    for fi, (tr_idx, te_idx) in enumerate(folds):
        if args.smoke:
            tr_idx = tr_idx[:120]; te_idx = te_idx[:40]
        tr_cases = {idx_case[i] for i in tr_idx}
        te_cases = {idx_case[i] for i in te_idx}
        inter = tr_cases & te_cases
        assert not inter, f"LEAKAGE fold {fi}: cases en train y held-out: {inter}"
        leak["folds"].append({
            "fold": fi + 1, "n_train": int(len(tr_idx)), "n_held": int(len(te_idx)),
            "case_overlap_train_held": len(inter),
            "n_train_after_byN": {f"N={n}": int(len(tr_idx) * (1 + n)) for n in n_list},
        })
        print(f"\n=== Fold {fi+1}/{len(folds)} | train={len(tr_idx)} held={len(te_idx)} "
              f"| cases train inter held={len(inter)} ===", flush=True)

        X_tr0 = X[tr_idx]
        y_tr0 = y_global[tr_idx]
        X_te = X[te_idx]                       # held-out SIEMPRE original (sin aug)

        # baseline OOF (no-aug)
        yp, pr = fit_mlp_global(X_tr0, y_tr0, X_te, n_cls)
        oof_pred["baseline"][te_idx] = yp; oof_prob["baseline"][te_idx] = pr
        oof_filled["baseline"][te_idx] = True

        # re-extraer aug del TRAIN de este fold (maxn copias, reutiliza prefijos)
        tr_rows = [idx_row[i] for i in tr_idx]
        aug = extract_aug_copies(model, tr_rows, maxn, base_seed=100 + fi, tag=f"fold{fi+1}")
        # assert: solo train images fueron aumentadas
        aug_cases = {idx_case[fn2idx[fn]] for fn in aug}
        assert aug_cases <= tr_cases, "LEAKAGE: aug fuera del train del fold"
        assert not (aug_cases & te_cases), "LEAKAGE: case held-out entre aumentados"
        # apila aug ordenado igual que tr_idx
        aug_stack = np.stack([aug[idx_row[i]["image_filename"]] for i in tr_idx])  # [Ntr, maxn, D]

        for n in n_list:
            X_aug = aug_stack[:, :n, :].reshape(-1, X.shape[1])      # [Ntr*n, D]
            y_aug = np.repeat(y_tr0, n)
            X_tr = np.vstack([X_tr0, X_aug])
            y_tr = np.concatenate([y_tr0, y_aug])
            yp, pr = fit_mlp_global(X_tr, y_tr, X_te, n_cls)
            v = f"N={n}"
            oof_pred[v][te_idx] = yp; oof_prob[v][te_idx] = pr
            oof_filled[v][te_idx] = True
            print(f"    {v}: train {len(tr_idx)}->{len(X_tr)}", flush=True)

    # metricas OOF agregadas sobre las imgs cubiertas
    oof_metrics = {}
    cov = oof_filled["baseline"]
    yt = y_global[cov]
    for v in variants:
        assert (oof_filled[v] == cov).all(), f"cobertura OOF inconsistente en {v}"
        oof_metrics[v] = metrics_row(yt, oof_pred[v][cov], oof_prob[v][cov], n_cls)
    print(f"\nOOF cubre {cov.sum()} imgs")

    # persistir arrays OOF por-muestra para bootstrap pareado
    if not args.smoke:
        vkey = {v: v.replace("=", "").replace("/", "") for v in variants}  # N=3->N3
        np.savez(
            OUT / "oof_arrays.npz",
            y=y_global, cov=cov,
            cases=np.array([idx_case.get(i, "") for i in range(len(fns))], dtype=object),
            n_cls=n_cls, variants=np.array(variants, dtype=object),
            **{f"pred_{vkey[v]}": oof_pred[v] for v in variants},
            **{f"prob_{vkey[v]}": oof_prob[v] for v in variants},
        )
        print(f"OK {OUT}/oof_arrays.npz")

    # =====================================================================
    # B. Test-36 fijo (referencia SECUNDARIA)
    # =====================================================================
    test36 = {}
    if not args.smoke:
        tr_idx = np.array([i for i in all_idx if idx_split[i] == "train"])
        te_idx = np.array([i for i in all_idx if idx_split[i] == "test"])
        tr_cases = {idx_case[i] for i in tr_idx}; te_cases = {idx_case[i] for i in te_idx}
        leak["test36"] = {
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "train_test_case_overlap": len(tr_cases & te_cases),
            "n_train_after_byN": {f"N={n}": int(len(tr_idx) * (1 + n)) for n in n_list},
        }
        assert not (tr_cases & te_cases), "LEAKAGE test36: case en train y test"
        X_tr0 = X[tr_idx]; y_tr0 = y_global[tr_idx]; X_te = X[te_idx]; y_te = y_global[te_idx]
        yp, pr = fit_mlp_global(X_tr0, y_tr0, X_te, n_cls)
        test36["baseline"] = metrics_row(y_te, yp, pr, n_cls)
        tr_rows = [idx_row[i] for i in tr_idx]
        print(f"\n=== Test-36 secundario: aug train {len(tr_idx)} ===", flush=True)
        aug = extract_aug_copies(model, tr_rows, maxn, base_seed=900, tag="test36")
        aug_cases = {idx_case[fn2idx[fn]] for fn in aug}
        assert not (aug_cases & te_cases), "LEAKAGE: case test entre aumentados (test36)"
        aug_stack = np.stack([aug[idx_row[i]["image_filename"]] for i in tr_idx])
        for n in n_list:
            X_aug = aug_stack[:, :n, :].reshape(-1, X.shape[1]); y_aug = np.repeat(y_tr0, n)
            yp, pr = fit_mlp_global(np.vstack([X_tr0, X_aug]),
                                    np.concatenate([y_tr0, y_aug]), X_te, n_cls)
            test36[f"N={n}"] = metrics_row(y_te, yp, pr, n_cls)

    # =====================================================================
    # Guardar
    # =====================================================================
    results = {
        "protocol": "5-fold OOF case-aware (mlp_512); aug solo en train por fold; "
                    "held-out original; OOF-vs-OOF.",
        "n_list": n_list, "n_classes": n_cls, "oof_coverage": int(cov.sum()),
        "oof_primary": oof_metrics,
        "test36_secondary": test36,
        "baseline_published_test36_mlp512": {
            "Acc@1": 0.25, "Acc@3": 0.5556, "BAcc": 0.2647,
            "AUROC": 0.8118, "W-F1": 0.2136, "Kappa": 0.1987},
    }
    # deltas OOF-vs-OOF
    deltas = {}
    base = oof_metrics["baseline"]
    for n in n_list:
        v = f"N={n}"
        deltas[v] = {k: round(oof_metrics[v][k] - base[k], 4) for k in KEYS}
    results["oof_deltas_vs_baseline"] = deltas

    suffix = "_smoke" if args.smoke else ""
    json.dump(results, open(OUT / f"oof_results{suffix}.json", "w"), indent=2, ensure_ascii=False)
    json.dump(leak, open(OUT / f"leakage_sanity{suffix}.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nOK {OUT}/oof_results{suffix}.json")
    print(json.dumps(results["oof_primary"], indent=1))
    print("DELTAS OOF:", json.dumps(deltas, indent=1))
    if test36:
        print("TEST36:", json.dumps(test36, indent=1))


if __name__ == "__main__":
    main()
