# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_tta_eval.py · Test-Time Augmentation sobre DermapixelAI 1.0

Estrategia:
  - Para cada imagen de TEST extrae 6 embeddings PanDerm:
      [original, hflip, vflip, rot90, rot180, rot270]
  - Promedio L2-normalizado de los 6 → embedding TTA
  - LP entrenado sobre embeddings train ORIGINAL (sin TTA)
  - Predice sobre embeddings TTA del test

Esto es la variante "TTA solo en test" — la más común en la literatura
y la que se aplica en producción (M1 server ya usa TTA con +4,29pp mel recall).

Para train se reutilizan los embeddings cacheados de §4.11.

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_tta/{results.json, summary.csv}

Uso:
    cd $DERMAPIXEL_ROOT
    python3 -W ignore dermapixel_tta_eval.py
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
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
WEIGHTS  = ROOT / "weights" / "weights"
LP_DIR   = ROOT / "output" / "dermapixel_v1_lp"
OUT_DIR  = ROOT / "output" / "dermapixel_v1_tta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classification"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------

def build_encoder(name):
    from models.modeling_finetune import (
        panderm_base_patch16_224, panderm_large_patch16_224,
    )
    ckpt = WEIGHTS / f"{name}.pth"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model" in sd: sd = sd["model"]
    if "state_dict" in sd: sd = sd["state_dict"]

    if name == "panderm_large":
        model = panderm_large_patch16_224()
        sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    else:
        model = panderm_base_patch16_224()
    model.load_state_dict(sd, strict=False)
    model.head = torch.nn.Identity()
    model = model.to(DEVICE).eval()
    return model


BASE_T = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


def apply_augmentations(pil_img):
    """Devuelve lista de 5 tensores (mismo TTA que server M1, sin rot 180).

    rot 180 está documentado en el server como ausente; rotaciones de 90/270
    son las que mejor preservan distribución según testing previo."""
    imgs = [
        pil_img,
        pil_img.transpose(Image.FLIP_LEFT_RIGHT),
        pil_img.transpose(Image.FLIP_TOP_BOTTOM),
        pil_img.rotate(90, expand=True),
        pil_img.rotate(270, expand=True),
    ]
    return [BASE_T(im) for im in imgs]


@torch.no_grad()
def tta_embeddings_per_aug(model, pil_img):
    """Devuelve matriz (5, dim) con embedding por augmentation (SIN promediar).

    El promediado correcto se hace después sobre las PROBABILIDADES del LP,
    no sobre los embeddings (que están en distribuciones distintas tras
    rotaciones)."""
    tensors = apply_augmentations(pil_img)
    batch = torch.stack(tensors).to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        feats = model(batch).float()
    return feats.cpu().numpy()  # (5, dim) sin normalizar individualmente


# -----------------------------------------------------------------------------
# Métricas (idénticas a §4.11)
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Cargar rows + consolidar L2
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)
    print(f"Total filas: {len(rows)}")

    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"Test imgs: {len(test_rows)}")

    results = {"meta": {"n_test": len(test_rows)}, "models": {}}

    for enc_name in ("panderm_large", "panderm_base"):
        print(f"\n========== {enc_name} ==========")
        model = build_encoder(enc_name)

        # Train embeddings reutilizados (sin TTA)
        X_train = np.load(LP_DIR / f"{enc_name}_embeddings.npy")
        with (LP_DIR / f"{enc_name}_filenames.json").open() as f:
            filenames = json.load(f)
        fn2idx = {fn: i for i, fn in enumerate(filenames)}

        # Re-extraer test embeddings CON TTA (5 augmentations por imagen)
        # Guardamos los 5 embeddings por imagen para promediar PROBABILIDADES
        # después (no embeddings).
        print(f"  Extrayendo TTA embeddings test (5 augs × {len(test_rows)} imgs)...")
        t0 = time.time()
        test_tta_emb_per_aug = []  # lista de (5, dim) — 1 por imagen
        for r in test_rows:
            img = Image.open(DATASET / r["image_path"]).convert("RGB")
            embs = tta_embeddings_per_aug(model, img)  # (5, dim)
            test_tta_emb_per_aug.append(embs)
        test_tta_emb_per_aug = np.stack(test_tta_emb_per_aug)  # (N_test, 5, dim)
        print(f"  shape: {test_tta_emb_per_aug.shape} en {time.time()-t0:.1f}s")
        np.save(OUT_DIR / f"{enc_name}_test_tta_per_aug.npy", test_tta_emb_per_aug)

        # Para cada nivel L1/L2/L3
        model_res = {}
        for level in ("ontology_l1", "ontology_l2", "ontology_l3"):
            # Train: usar embeddings train cacheados (sin TTA)
            X_tr, y_tr_str = [], []
            for r in rows:
                if r["split"] != "train": continue
                if r["image_filename"] not in fn2idx: continue
                X_tr.append(X_train[fn2idx[r["image_filename"]]])
                y_tr_str.append(r[level])
            X_tr = np.array(X_tr)
            lab2id = {l: i for i, l in enumerate(sorted(set(y_tr_str)))}
            y_tr = np.array([lab2id[y] for y in y_tr_str])

            # Test: 5 embeddings por imagen, aplicar LP a cada uno y promediar
            # probabilidades. Esto sigue el patrón del server M1.
            y_te_str = [r[level] for r in test_rows]
            y_te_all = np.array([lab2id.get(y, -1) for y in y_te_str])
            mask = y_te_all >= 0
            X_te_aug = test_tta_emb_per_aug[mask]  # (N_te, 5, dim)
            y_te = y_te_all[mask]
            n_cls = len(lab2id)

            # LP estándar entrenado sobre train original (sin TTA)
            clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
            clf.fit(X_tr, y_tr)

            # Para cada imagen: predecir 5 probs (1 por augmentation) y promediar
            n_te, n_augs, dim = X_te_aug.shape
            X_flat = X_te_aug.reshape(n_te * n_augs, dim)  # (N_te*5, dim)
            prob_flat = clf.predict_proba(X_flat)
            if prob_flat.shape[1] < n_cls:
                full = np.zeros((prob_flat.shape[0], n_cls))
                for j, ci in enumerate(clf.classes_):
                    full[:, ci] = prob_flat[:, j]
                prob_flat = full
            prob_per_aug = prob_flat.reshape(n_te, n_augs, n_cls)
            prob = prob_per_aug.mean(axis=1)  # promediar las 5 probs
            y_pred = prob.argmax(axis=1)

            base = metrics_all(y_te, y_pred, prob, n_cls)
            out = {"value": {}, "ci": {}, "n_train": len(y_tr),
                   "n_test": len(y_te), "n_classes": n_cls}
            for k in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                m, lo, hi = bootstrap_metric(y_te, y_pred, prob, n_cls, k)
                out["value"][k] = round(base[k], 4)
                out["ci"][k] = [round(lo, 4) if lo else None,
                                round(hi, 4) if hi else None]
            model_res[level] = out
            print(f"  {level}: Acc@1={out['value']['Acc@1']} "
                  f"BAcc={out['value']['BAcc']} AUROC={out['value']['AUROC']} "
                  f"Acc@3={out['value']['Acc@3']}")

        results["models"][enc_name] = model_res

        del model
        torch.cuda.empty_cache()

    # Output
    out_json = OUT_DIR / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Resultados → {out_json}")

    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "level", "metric", "value", "ci_low", "ci_high"])
        for enc, levels in results["models"].items():
            for lvl, ldata in levels.items():
                for met in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
                    w.writerow([enc, lvl, met, ldata["value"][met],
                                ldata["ci"][met][0], ldata["ci"][met][1]])
    print(f"✓ CSV → {csv_path}")


if __name__ == "__main__":
    main()
