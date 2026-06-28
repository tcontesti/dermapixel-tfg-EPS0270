# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_ft_l1.py · Fine-tuning PanDerm Large head L1 sobre DermapixelAI 1.0

Protocolo:
  - Encoder PanDerm Large (1024-d), últimas 2 capas descongeladas
  - Head FC 1024 → 4 (L1: Patología inflamatoria/infecciosa/tumoral/Genodermatosis)
  - 10 épocas, AdamW, lr_head=1e-3, lr_encoder=1e-5, weight_decay=1e-4
  - BS=16, cosine warmup 1 época
  - Augmentations train: RandomResizedCrop 224, hflip, vflip, ColorJitter
  - Augmentations eval: Resize 256 + CenterCrop 224
  - class_weight balanced en CrossEntropy

Métricas vs LP baseline §4.11 + bootstrap IC95% 1000.

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_ft_l1/{results.json, summary.csv}
"""
from __future__ import annotations
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
    accuracy_score, balanced_accuracy_score,
    cohen_kappa_score, f1_score, roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
WEIGHTS  = ROOT / "weights" / "weights"
OUT_DIR  = ROOT / "output" / "dermapixel_v1_ft_l1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classification"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS       = 10
BATCH_SIZE   = 16
LR_HEAD      = 1e-3
LR_ENCODER   = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4

L1_CLASSES = sorted([
    "Patología inflamatoria", "Patología infecciosa",
    "Patología tumoral", "Genodermatosis",
])
L1_TO_ID = {c: i for i, c in enumerate(L1_CLASSES)}
N_CLASSES = len(L1_CLASSES)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)

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
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_path = DATASET / r["image_path"]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = L1_TO_ID[r["ontology_l1"]]
        return x, y


# -----------------------------------------------------------------------------
# Modelo: encoder PanDerm Large + head FC
# -----------------------------------------------------------------------------

class PanDermClassifier(nn.Module):
    def __init__(self, encoder, n_classes=4, dim=1024):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats)


def build_model():
    from models.modeling_finetune import panderm_large_patch16_224
    sd = torch.load(WEIGHTS / "panderm_large.pth", map_location="cpu", weights_only=False)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    encoder = panderm_large_patch16_224()
    encoder.load_state_dict(sd, strict=False)
    encoder.head = nn.Identity()

    # Congelar todo el encoder excepto las últimas 2 capas (block.22, block.23 en ViT-L)
    for p in encoder.parameters():
        p.requires_grad = False
    # Descongelar últimas 2 capas
    n_unfreeze = 2
    if hasattr(encoder, "blocks"):
        for blk in encoder.blocks[-n_unfreeze:]:
            for p in blk.parameters():
                p.requires_grad = True
    # Descongelar norm final si existe y no es None
    for name in ("norm", "fc_norm"):
        attr = getattr(encoder, name, None)
        if attr is not None and hasattr(attr, "parameters"):
            for p in attr.parameters():
                p.requires_grad = True

    model = PanDermClassifier(encoder, n_classes=N_CLASSES)
    model = model.to(DEVICE)
    return model


# -----------------------------------------------------------------------------
# Métricas
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


def metrics_all(yt, yp, prob, n_cls=N_CLASSES):
    return {
        "Acc@1": accuracy_score(yt, yp),
        "Acc@3": topk_acc(yt, prob, 3),
        "BAcc":  balanced_accuracy_score(yt, yp),
        "AUROC": safe_auroc(yt, prob, n_cls),
        "W-F1":  f1_score(yt, yp, average="weighted", zero_division=0),
        "Kappa": cohen_kappa_score(yt, yp),
    }


def bootstrap_metric(yt, yp, prob, metric_key, n_iter=1000, seed=42):
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
            m = metrics_all(yt[idx], yp[idx], prob[idx])
            vals.append(m[metric_key])
        except (ValueError, IndexError):
            continue
    if not vals: return None, None, None
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# -----------------------------------------------------------------------------
# Train / Eval
# -----------------------------------------------------------------------------

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
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_yp)
    prob = np.concatenate(all_prob)
    return y_true, y_pred, prob


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn):
    model.train()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            if r["ontology_l1"] not in L1_TO_ID:
                continue
            rows.append(r)
    print(f"Total filas válidas: {len(rows)}")

    tr_rows = [r for r in rows if r["split"] == "train"]
    va_rows = [r for r in rows if r["split"] == "val"]
    te_rows = [r for r in rows if r["split"] == "test"]
    print(f"train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)}")

    # Class counts en train
    counts = np.bincount([L1_TO_ID[r["ontology_l1"]] for r in tr_rows], minlength=N_CLASSES)
    print(f"Class counts train: {dict(zip(L1_CLASSES, counts))}")
    class_weights = np.where(counts > 0,
                              len(tr_rows) / (N_CLASSES * np.maximum(counts, 1)),
                              0.0)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    tr_ds = DermaSet(tr_rows, T_TRAIN)
    va_ds = DermaSet(va_rows, T_EVAL)
    te_ds = DermaSet(te_rows, T_EVAL)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model()
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo: {n_total/1e6:.1f}M total, {n_trainable/1e6:.1f}M entrenables "
          f"({100*n_trainable/n_total:.1f}%)")

    # Optimizer con grupos diferenciados
    head_params = list(model.head.parameters())
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD},
        {"params": encoder_params, "lr": LR_ENCODER},
    ], weight_decay=WEIGHT_DECAY)

    n_steps = len(tr_loader) * EPOCHS
    warmup_steps = len(tr_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    # Train loop
    history = []
    best_val_bacc = -1
    best_state = None
    t_start = time.time()
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, tr_loader, optimizer, scheduler, None, loss_fn)

        # Eval val
        y_v, yp_v, prob_v = evaluate(model, va_loader)
        m_v = metrics_all(y_v, yp_v, prob_v)
        # Eval test
        y_t, yp_t, prob_t = evaluate(model, te_loader)
        m_t = metrics_all(y_t, yp_t, prob_t)

        elapsed = time.time() - t0
        history.append({"epoch": epoch+1, "train_loss": train_loss,
                        "val": m_v, "test": m_t, "secs": elapsed})
        print(f"  ep{epoch+1:2d} | loss={train_loss:.4f} | "
              f"val BAcc={m_v['BAcc']:.3f} AUROC={m_v['AUROC']:.3f} | "
              f"test BAcc={m_t['BAcc']:.3f} AUROC={m_t['AUROC']:.3f} | "
              f"{elapsed:.1f}s")

        if m_v["BAcc"] > best_val_bacc:
            best_val_bacc = m_v["BAcc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nTotal training: {(time.time()-t_start)/60:.1f} min")

    # Eval final con mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)
    y_t, yp_t, prob_t = evaluate(model, te_loader)
    final = metrics_all(y_t, yp_t, prob_t)
    print(f"\n=== Test FINAL (best val) ===")
    print(f"  Acc@1={final['Acc@1']:.4f}  BAcc={final['BAcc']:.4f}  "
          f"AUROC={final['AUROC']:.4f}  Acc@3={final['Acc@3']:.4f}")

    # Bootstrap IC95%
    print("\nBootstrap IC95% (1000 remuestreos)...")
    bootstrap_ci = {}
    for k in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
        m, lo, hi = bootstrap_metric(y_t, yp_t, prob_t, k)
        bootstrap_ci[k] = {"value": round(final[k], 4),
                           "ci95_low": round(lo, 4) if lo else None,
                           "ci95_high": round(hi, 4) if hi else None}
        print(f"  {k}: {final[k]:.4f} [{lo:.4f}, {hi:.4f}]")

    # Save
    results = {
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr_head": LR_HEAD, "lr_encoder": LR_ENCODER,
            "weight_decay": WEIGHT_DECAY,
            "unfrozen_last_blocks": 2,
            "n_trainable_M": round(n_trainable/1e6, 2),
            "augmentations": ["RandomResizedCrop", "HFlip", "VFlip(0.3)", "ColorJitter"],
        },
        "n_train": len(tr_rows), "n_val": len(va_rows), "n_test": len(te_rows),
        "classes": L1_CLASSES,
        "class_counts_train": dict(zip(L1_CLASSES, counts.tolist())),
        "final_test_metrics": bootstrap_ci,
        "history": history,
        "best_val_bacc": float(best_val_bacc),
    }
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ {OUT_DIR / 'results.json'}")

    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "ci_low", "ci_high"])
        for k, v in bootstrap_ci.items():
            w.writerow([k, v["value"], v["ci95_low"], v["ci95_high"]])
    print(f"✓ {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
