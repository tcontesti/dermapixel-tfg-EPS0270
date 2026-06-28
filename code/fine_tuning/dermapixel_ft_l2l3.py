# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_ft_l2l3.py · Fine-tuning PanDerm Large completo sobre L2 y L3

Misma arquitectura y protocolo que dermapixel_ft_l1.py (§4.11 FT L1):
  - Encoder PanDerm Large, últimas 2 capas descongeladas (25M trainable)
  - Cabeza FC 1024 → N_CLS (38 L2 / 224 L3)
  - 10 épocas, AdamW lr_head=1e-3, lr_encoder=1e-5, weight_decay=1e-4
  - BS 16, cosine warmup 1 ép, class_weight balanced
  - Augmentations train: RandomResizedCrop, hflip, vflip(0.3), ColorJitter
  - Selección por mejor val BAcc + bootstrap IC95% sobre test

LANZA dos entrenamientos secuenciales (L2 y L3) en el mismo run.

Comparativa con baselines existentes:
  - L2: LP 0,265 / Ensemble L+D 0,279 / SpanDerm v0 LoRA 0,363±0,007
  - L3: LP 0,184 / Ensemble L 0,184 / sin LoRA L3 todavía

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_ft_{l2,l3}/ con results.json y summary.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
WEIGHTS  = ROOT / "weights" / "weights"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classification"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
BATCH_SIZE = 16
LR_HEAD = 1e-3
LR_ENCODER = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

T_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])
T_EVAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class DermaSet(Dataset):
    def __init__(self, rows, transform, lab2id, level):
        self.rows = rows
        self.transform = transform
        self.lab2id = lab2id
        self.level = level
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(DATASET / r["image_path"]).convert("RGB")
        x = self.transform(img)
        y = self.lab2id[r[self.level]]
        return x, y


class PanDermClassifier(nn.Module):
    def __init__(self, encoder, n_classes, dim=1024):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, n_classes)
    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats)


def build_model(n_classes):
    from models.modeling_finetune import panderm_large_patch16_224
    sd = torch.load(WEIGHTS / "panderm_large.pth", map_location="cpu", weights_only=False)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    encoder = panderm_large_patch16_224()
    encoder.load_state_dict(sd, strict=False)
    encoder.head = nn.Identity()
    for p in encoder.parameters():
        p.requires_grad = False
    for blk in encoder.blocks[-2:]:
        for p in blk.parameters():
            p.requires_grad = True
    for name in ("norm", "fc_norm"):
        attr = getattr(encoder, name, None)
        if attr is not None and hasattr(attr, "parameters"):
            for p in attr.parameters():
                p.requires_grad = True
    return PanDermClassifier(encoder, n_classes=n_classes).to(DEVICE)


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


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_y, all_yp, all_prob = [], [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        prob = F.softmax(logits, dim=-1).cpu().numpy()
        all_y.append(y.numpy())
        all_prob.append(prob)
        all_yp.append(prob.argmax(axis=1))
    return np.concatenate(all_y), np.concatenate(all_yp), np.concatenate(all_prob)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits.float(), y)
        loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item() * x.size(0); total_n += x.size(0)
    return total_loss / total_n


def run_level(level):
    """Entrena FT denso sobre el level dado y guarda resultados."""
    out_dir = ROOT / "output" / f"dermapixel_v1_ft_{level[-1]}"  # ft_l2 / ft_l3
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n=== FT denso PanDerm Large sobre {level} ===\n{'='*60}\n")

    # Cargar rows
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            if not r[level]:
                continue
            rows.append(r)

    tr_rows = [r for r in rows if r["split"] == "train"]
    va_rows = [r for r in rows if r["split"] == "val"]
    te_rows = [r for r in rows if r["split"] == "test"]

    classes = sorted(set(r[level] for r in tr_rows))
    lab2id = {c: i for i, c in enumerate(classes)}
    n_classes = len(lab2id)

    # Filtrar val/test a clases visibles en train
    va_rows = [r for r in va_rows if r[level] in lab2id]
    te_rows = [r for r in te_rows if r[level] in lab2id]
    print(f"train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)} | {n_classes} clases")

    # Class weights
    counts = np.bincount([lab2id[r[level]] for r in tr_rows], minlength=n_classes)
    cw = np.where(counts > 0, len(tr_rows) / (n_classes * np.maximum(counts, 1)), 0.0)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    tr_ds = DermaSet(tr_rows, T_TRAIN, lab2id, level)
    va_ds = DermaSet(va_rows, T_EVAL, lab2id, level)
    te_ds = DermaSet(te_rows, T_EVAL, lab2id, level)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(n_classes)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_trainable/1e6:.2f} M")

    head_params = list(model.head.parameters())
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD},
        {"params": encoder_params, "lr": LR_ENCODER},
    ], weight_decay=WEIGHT_DECAY)
    n_steps = len(tr_loader) * EPOCHS
    warmup_steps = len(tr_loader)
    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(weight=cw_t)

    history = []
    best_val_bacc = -1
    best_state = None
    t_start = time.time()
    for ep in range(EPOCHS):
        t0 = time.time()
        loss = train_one_epoch(model, tr_loader, optimizer, scheduler, loss_fn)
        y_v, yp_v, prob_v = evaluate(model, va_loader)
        y_t, yp_t, prob_t = evaluate(model, te_loader)
        m_v = metrics_all(y_v, yp_v, prob_v, n_classes)
        m_t = metrics_all(y_t, yp_t, prob_t, n_classes)
        elapsed = time.time() - t0
        history.append({"epoch": ep+1, "loss": loss, "val": m_v, "test": m_t, "secs": elapsed})
        print(f"  ep{ep+1:2d} loss={loss:.3f} | val BAcc={m_v['BAcc']:.3f} AUROC={m_v['AUROC']:.3f} | "
              f"test BAcc={m_t['BAcc']:.3f} AUROC={m_t['AUROC']:.3f} | {elapsed:.1f}s")
        if m_v["BAcc"] > best_val_bacc:
            best_val_bacc = m_v["BAcc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nTraining {level}: {(time.time()-t_start)/60:.1f} min")
    if best_state is not None:
        model.load_state_dict(best_state)

    y_t, yp_t, prob_t = evaluate(model, te_loader)
    final = metrics_all(y_t, yp_t, prob_t, n_classes)
    print(f"\n=== Test FINAL {level} (best val BAcc={best_val_bacc:.3f}) ===")
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")

    ci = {}
    print("\nBootstrap IC95%...")
    for k in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
        m, lo, hi = bootstrap_metric(y_t, yp_t, prob_t, n_classes, k)
        ci[k] = {"value": round(final[k], 4) if not np.isnan(final[k]) else None,
                 "ci95_low": round(lo, 4) if lo else None,
                 "ci95_high": round(hi, 4) if hi else None}
        print(f"  {k}: {final[k]:.4f} [{lo:.4f}, {hi:.4f}]")

    results = {
        "config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                   "lr_head": LR_HEAD, "lr_encoder": LR_ENCODER,
                   "weight_decay": WEIGHT_DECAY,
                   "unfrozen_last_blocks": 2,
                   "n_trainable_M": round(n_trainable/1e6, 2),
                   "augmentations": ["RandomResizedCrop", "HFlip", "VFlip(0.3)", "ColorJitter"]},
        "level": level, "n_classes": n_classes,
        "n_train": len(tr_rows), "n_val": len(va_rows), "n_test": len(te_rows),
        "best_val_bacc": float(best_val_bacc),
        "final_test_metrics": ci,
        "history": history,
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_low", "ci_high"])
        for k, v in ci.items():
            w.writerow([k, v["value"], v["ci95_low"], v["ci95_high"]])
    print(f"\n✓ {out_dir / 'results.json'}\n✓ {out_dir / 'summary.csv'}")

    # Liberar
    del model; torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=["ontology_l2", "ontology_l3"])
    args = parser.parse_args()
    for level in args.levels:
        run_level(level)


if __name__ == "__main__":
    main()
