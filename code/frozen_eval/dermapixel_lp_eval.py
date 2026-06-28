# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_lp_eval.py · Evaluación LP sobre DermapixelAI 1.0 ($DERMAPIXEL_ROOT/datasets/dermapixel_v1)

Pipeline:
1. Carga PanDerm Base y PanDerm Large desde $DERMAPIXEL_ROOT/weights/weights/.
2. Extrae embeddings sobre las 1062 imágenes filtradas (clinical+dermoscopy, L1≠∅, label_source=ontology).
3. LP con LogReg L-BFGS C=1.0 max_iter=5000 seed=42 sobre L1 (4 cls), L2 (~38 cls), L3 (~250 cls).
4. Bootstrap IC95% (1000 remuestreos estratificados por clase) sobre el split test.
5. Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_lp/results.json + .csv + report.md

Uso:
    cd $DERMAPIXEL_ROOT
    python3 dermapixel_lp_eval.py
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, f1_score, cohen_kappa_score,
)
from sklearn.utils import resample
from torchvision import transforms

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classification"))

DATASET_DIR = ROOT / "datasets" / "dermapixel_v1"
WEIGHTS_DIR = ROOT / "weights" / "weights"
OUT_DIR     = ROOT / "output" / "dermapixel_v1_lp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# -----------------------------------------------------------------------------
# 1 · Carga del CSV filtrado
# -----------------------------------------------------------------------------

def load_dataset_csv():
    rows = []
    with (DATASET_DIR / "dataset_filtered.csv").open(encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    print(f"Filas en dataset_filtered.csv: {len(rows)}")
    return rows


# -----------------------------------------------------------------------------
# 2 · Carga modelo
# -----------------------------------------------------------------------------

def build_encoder(name: str):
    """Carga PanDerm Base/Large usando la API oficial del repo (builder.py).

    - PanDerm Large: panderm_large_patch16_224(); quitar prefijo 'encoder.'.
    - PanDerm Base: panderm_base_patch16_224(); strict=False; head=Identity.
    """
    from models.modeling_finetune import (
        panderm_base_patch16_224, panderm_large_patch16_224,
    )

    ckpt_path = {
        "panderm_base":  WEIGHTS_DIR / "panderm_base.pth",
        "panderm_large": WEIGHTS_DIR / "panderm_large.pth",
    }[name]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    if "state_dict" in sd:
        sd = sd["state_dict"]

    if name == "panderm_large":
        model = panderm_large_patch16_224()
        sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        model.head = torch.nn.Identity()
    else:
        model = panderm_base_patch16_224()
        msg = model.load_state_dict(sd, strict=False)
        model.head = torch.nn.Identity()

    print(f"  {name} | missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"    primeras missing: {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"    primeras unexpected: {msg.unexpected_keys[:5]}")

    model = model.to(device).eval()
    eval_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return model, eval_t


# -----------------------------------------------------------------------------
# 3 · Extracción de embeddings
# -----------------------------------------------------------------------------

def extract_embeddings(model, transform, rows, batch_size=32):
    """Devuelve (X: ndarray [N, D], filenames: list)."""
    n = len(rows)
    feats = []
    filenames = []

    t0 = time.time()
    with torch.no_grad():
        batch_imgs, batch_fns = [], []
        for i, r in enumerate(rows):
            img_path = DATASET_DIR / r["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img)
                batch_imgs.append(tensor)
                batch_fns.append(r["image_filename"])
            except Exception as e:
                print(f"  ! error {img_path}: {e}")
                continue

            if len(batch_imgs) >= batch_size or i == n - 1:
                if not batch_imgs:
                    continue
                X = torch.stack(batch_imgs).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    f = model(X)
                feats.append(f.float().cpu().numpy())
                filenames.extend(batch_fns)
                batch_imgs, batch_fns = [], []
                if (i + 1) % 100 == 0 or i == n - 1:
                    elapsed = time.time() - t0
                    print(f"  procesadas {i+1}/{n} en {elapsed:.1f}s")

    X = np.vstack(feats) if feats else np.empty((0, 0))
    return X, filenames


# -----------------------------------------------------------------------------
# 4 · Linear probing + bootstrap IC95%
# -----------------------------------------------------------------------------

def stratified_bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_iter=1000, seed=42):
    """Devuelve (media, ic_low, ic_high) para metric_fn(y_true, y_pred, y_prob)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    vals = []
    for _ in range(n_iter):
        idx_resamp = []
        for c in classes:
            idx_c = np.where(y_true == c)[0]
            if len(idx_c) == 0:
                continue
            idx_resamp.extend(rng.choice(idx_c, size=len(idx_c), replace=True))
        idx = np.array(idx_resamp)
        try:
            v = metric_fn(y_true[idx], y_pred[idx], y_prob[idx])
            vals.append(v)
        except (ValueError, IndexError):
            continue
    if not vals:
        return None, None, None
    vals = np.array(vals)
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compute_metrics(y_true, y_pred, y_prob):
    n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # AUROC OvR macro custom: ignora clases ausentes en y_true
    try:
        if n_classes == 2:
            auroc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            aurocs = []
            for c in range(n_classes):
                pos = (y_true == c).sum()
                neg = (y_true != c).sum()
                if pos == 0 or neg == 0:
                    continue
                try:
                    aurocs.append(roc_auc_score((y_true == c).astype(int), y_prob[:, c]))
                except ValueError:
                    continue
            auroc = float(np.mean(aurocs)) if aurocs else float("nan")
    except (ValueError, IndexError):
        auroc = float("nan")

    return {"Acc": acc, "BAcc": bacc, "AUROC": auroc, "W-F1": wf1, "Kappa": kappa}


def run_lp(X_train, y_train, X_test, y_test, label_name: str):
    """Devuelve dict con métricas + IC95%."""
    print(f"\n--- LP {label_name} | train={len(y_train)} test={len(y_test)} clases={len(np.unique(y_train))} ---")
    if len(np.unique(y_train)) < 2:
        return {"error": "menos de 2 clases en train"}

    clf = LogisticRegression(
        C=1.0, max_iter=5000, solver="lbfgs",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    base = compute_metrics(y_test, y_pred, y_prob)
    print(f"  Acc={base['Acc']:.3f}  BAcc={base['BAcc']:.3f}  AUROC={base['AUROC']:.3f}  W-F1={base['W-F1']:.3f}")

    # Bootstrap IC95%
    print("  Bootstrap IC95% (1000 remuestreos)...")
    metrics_fn = {
        "Acc":   lambda yt, yp, pp: accuracy_score(yt, yp),
        "BAcc":  lambda yt, yp, pp: balanced_accuracy_score(yt, yp),
        "W-F1":  lambda yt, yp, pp: f1_score(yt, yp, average="weighted", zero_division=0),
        "Kappa": lambda yt, yp, pp: cohen_kappa_score(yt, yp),
        "AUROC": lambda yt, yp, pp: compute_metrics(yt, yp, pp)["AUROC"],
    }
    out = {"n_train": int(len(y_train)), "n_test": int(len(y_test)),
           "n_classes": int(len(np.unique(y_train))), "label": label_name,
           "metrics": {}}
    for name, fn in metrics_fn.items():
        mean, lo, hi = stratified_bootstrap_ci(y_test, y_pred, y_prob, fn,
                                                n_iter=1000, seed=42)
        out["metrics"][name] = {
            "value": round(base[name], 4) if not np.isnan(base[name]) else None,
            "bootstrap_mean": round(mean, 4) if mean is not None else None,
            "ci95_low": round(lo, 4) if lo is not None else None,
            "ci95_high": round(hi, 4) if hi is not None else None,
        }
    return out


# -----------------------------------------------------------------------------
# 5 · Main
# -----------------------------------------------------------------------------

def main():
    rows = load_dataset_csv()
    print(f"Total imágenes: {len(rows)}")

    # Indexar splits
    train_idx = [i for i, r in enumerate(rows) if r["split"] == "train"]
    val_idx   = [i for i, r in enumerate(rows) if r["split"] == "val"]
    test_idx  = [i for i, r in enumerate(rows) if r["split"] == "test"]
    print(f"Train={len(train_idx)} Val={len(val_idx)} Test={len(test_idx)}")

    # Cuentas por nivel
    from collections import Counter
    for lvl in ("ontology_l1", "ontology_l2", "ontology_l3"):
        cnt = Counter(r[lvl] for r in rows)
        print(f"  {lvl}: {len(cnt)} clases únicas")

    results = {"meta": {
        "dataset": "DermapixelAI 1.0",
        "n_total": len(rows),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "filter": "clinical+dermoscopy, label_source=ontology, L1 not empty",
    }, "models": {}}

    encoders = ["panderm_large", "panderm_base"]

    for enc_name in encoders:
        print(f"\n========== {enc_name} ==========")
        try:
            model, transform = build_encoder(enc_name)
        except Exception as e:
            print(f"  ! no se pudo cargar {enc_name}: {e}")
            results["models"][enc_name] = {"error": str(e)}
            continue

        # Extraer embeddings
        emb_file = OUT_DIR / f"{enc_name}_embeddings.npy"
        if emb_file.exists():
            print(f"  → reutilizando {emb_file.name}")
            X = np.load(emb_file)
            with (OUT_DIR / f"{enc_name}_filenames.json").open() as f:
                filenames = json.load(f)
        else:
            print(f"  → extrayendo embeddings...")
            X, filenames = extract_embeddings(model, transform, rows, batch_size=32)
            np.save(emb_file, X)
            with (OUT_DIR / f"{enc_name}_filenames.json").open("w") as f:
                json.dump(filenames, f)
        print(f"  embeddings shape: {X.shape}")

        # Mapeo image_filename -> índice en X
        fn_to_idx = {fn: i for i, fn in enumerate(filenames)}

        # Para cada nivel L1, L2, L3: hacer LP
        model_results = {}
        for lvl in ("ontology_l1", "ontology_l2", "ontology_l3"):
            # Recoger features + labels por split
            X_tr, y_tr, X_te, y_te = [], [], [], []
            for i, r in enumerate(rows):
                if r["image_filename"] not in fn_to_idx:
                    continue
                xi = X[fn_to_idx[r["image_filename"]]]
                yi = r[lvl]
                if r["split"] == "train":
                    X_tr.append(xi); y_tr.append(yi)
                elif r["split"] == "test":
                    X_te.append(xi); y_te.append(yi)

            X_tr, X_te = np.array(X_tr), np.array(X_te)
            # Indexar labels a enteros
            label_to_idx = {l: i for i, l in enumerate(sorted(set(y_tr)))}
            y_tr_idx = np.array([label_to_idx[y] for y in y_tr])
            y_te_idx = np.array([label_to_idx.get(y, -1) for y in y_te])
            # Filtrar test cuyas labels no estén en train
            mask = y_te_idx >= 0
            X_te_v = X_te[mask]
            y_te_v = y_te_idx[mask]

            res = run_lp(X_tr, y_tr_idx, X_te_v, y_te_v, lvl)
            res["test_filtered_size"] = int(mask.sum())
            res["test_original_size"] = int(len(y_te_idx))
            res["test_unseen_labels"] = int((~mask).sum())
            model_results[lvl] = res

        results["models"][enc_name] = model_results

        # Limpiar GPU
        del model
        torch.cuda.empty_cache()

    # Guardar JSON
    out_json = OUT_DIR / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {out_json}")

    # Generar tabla CSV resumen
    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "n_train", "n_test", "n_classes",
                    "metric", "value", "ci95_low", "ci95_high"])
        for enc, lvls in results["models"].items():
            if "error" in lvls:
                continue
            for lvl, res in lvls.items():
                if "error" in res:
                    continue
                for met, vals in res["metrics"].items():
                    w.writerow([enc, lvl, res["n_train"], res["n_test"],
                                res["n_classes"], met,
                                vals["value"], vals["ci95_low"], vals["ci95_high"]])
    print(f"Resumen CSV: {csv_path}")


if __name__ == "__main__":
    main()
