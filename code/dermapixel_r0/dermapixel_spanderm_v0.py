# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_spanderm_v0.py · SpanDerm v0 — LoRA L2 + ranking L3

Arquitectura:
  - Encoder PanDerm Large (1024-d) con LoRA r=16 alpha=32 sobre blocks.22-23
  - Head FC 1024 → 38 clases L2 (consolidación queratinización)
  - Ranking L3 derivado: similitud coseno sobre prototipos L3 calculados
    como media de embeddings train por clase (250 clases globales)

Entrenamiento:
  - 15 épocas, AdamW
  - LR head 1e-3, LR LoRA 5e-4, weight_decay 1e-4
  - BS 16, cosine warmup 1 época
  - Augmentations: RandomResizedCrop, hflip, vflip(0.3), ColorJitter
  - CrossEntropy con class_weight balanced en L2
  - Best checkpoint por val BAcc L2

Métricas:
  - L2: Acc@1, Acc@3, BAcc, AUROC OvR macro + bootstrap IC95%
  - L3 ranking: Acc@1, Acc@3, recall@5 sobre prototipos
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
OUT_DIR  = ROOT / "output" / "dermapixel_v1_spanderm_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classification"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS       = 15
BATCH_SIZE   = 16
LR_HEAD      = 1e-3
LR_LORA      = 5e-4
WEIGHT_DECAY = 1e-4
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
N_LAST_BLOCKS_TO_ADAPT = 2  # blocks.22 and blocks.23

L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


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
    def __init__(self, rows, transform, l2_to_id):
        self.rows = rows
        self.transform = transform
        self.l2_to_id = l2_to_id

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(DATASET / r["image_path"]).convert("RGB")
        x = self.transform(img)
        y_l2 = self.l2_to_id[r["ontology_l2"]]
        return x, y_l2


# -----------------------------------------------------------------------------
# Modelo: encoder PanDerm Large + LoRA + head L2
# -----------------------------------------------------------------------------

class SpanDermV0(nn.Module):
    def __init__(self, encoder, n_classes_l2, dim=1024):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, n_classes_l2)

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.head(feats)
        return logits, feats


def build_spanderm_v0(n_classes_l2):
    """Cargar PanDerm Large + aplicar LoRA a últimas 2 capas con PEFT."""
    from models.modeling_finetune import panderm_large_patch16_224
    from peft import LoraConfig, get_peft_model

    # Cargar encoder
    sd = torch.load(WEIGHTS / "panderm_large.pth", map_location="cpu", weights_only=False)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    encoder = panderm_large_patch16_224()
    encoder.load_state_dict(sd, strict=False)
    encoder.head = nn.Identity()

    # Congelar todo el encoder
    for p in encoder.parameters():
        p.requires_grad = False

    # PEFT LoRA sobre las últimas 2 capas (blocks.22, blocks.23)
    n_blocks = len(encoder.blocks)
    target_blocks = list(range(n_blocks - N_LAST_BLOCKS_TO_ADAPT, n_blocks))
    target_modules = []
    for b in target_blocks:
        for sub in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            target_modules.append(f"blocks.{b}.{sub}")

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, bias="none",
        modules_to_save=[],  # no extra saved modules
    )
    encoder = get_peft_model(encoder, lora_cfg)
    encoder.print_trainable_parameters()

    model = SpanDermV0(encoder, n_classes_l2)
    return model.to(DEVICE)


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
    if prob.shape[1] < k: return 1.0
    topk = np.argsort(-prob, axis=1)[:, :k]
    return float(np.mean([yt[i] in topk[i] for i in range(len(yt))]))


def metrics_l2(yt, yp, prob, n_cls):
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
            m = metrics_l2(yt[idx], yp[idx], prob[idx], n_cls)
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
def evaluate(model, loader, n_cls_l2):
    model.eval()
    all_y, all_yp, all_prob, all_feats = [], [], [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, feats = model(x)
            logits = logits.float()
            feats = feats.float()
        prob = F.softmax(logits, dim=-1).cpu().numpy()
        all_y.append(y.numpy())
        all_prob.append(prob)
        all_yp.append(prob.argmax(axis=1))
        all_feats.append(feats.cpu().numpy())
    return (np.concatenate(all_y), np.concatenate(all_yp),
            np.concatenate(all_prob), np.concatenate(all_feats))


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = loss_fn(logits.float(), y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n


# -----------------------------------------------------------------------------
# Ranking L3 por similitud coseno
# -----------------------------------------------------------------------------

class FeatExtractSet(Dataset):
    """Dataset puro de extracción: solo imagen, sin label."""
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(DATASET / r["image_path"]).convert("RGB")
        return self.transform(img), idx


@torch.no_grad()
def extract_features(model, rows):
    """Extrae embeddings normalizados sobre rows."""
    model.eval()
    ds = FeatExtractSet(rows, T_EVAL)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    all_feats = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, feats = model(x)
            feats = feats.float()
        feats_n = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats_n.cpu().numpy())
    return np.vstack(all_feats)


def compute_l3_prototypes(model, train_rows, l3_to_id):
    """Calcula prototipos L3 como media de embeddings train por clase."""
    all_feats = extract_features(model, train_rows)
    all_l3 = np.array([l3_to_id[r["ontology_l3"]] for r in train_rows])
    n_classes_l3 = len(l3_to_id)
    dim = all_feats.shape[1]
    protos = np.zeros((n_classes_l3, dim))
    for c in range(n_classes_l3):
        mask = all_l3 == c
        if mask.sum() == 0: continue
        protos[c] = all_feats[mask].mean(axis=0)
    norms = np.linalg.norm(protos, axis=1, keepdims=True)
    return protos / np.maximum(norms, 1e-8)


def eval_l3_ranking(model, test_rows, l3_to_id, protos):
    """Embedding test → similitud coseno con prototipos → top-k."""
    all_feats = extract_features(model, test_rows)
    sims = all_feats @ protos.T  # (n_test, n_classes_l3)

    y_true = np.array([l3_to_id.get(r["ontology_l3"], -1) for r in test_rows])
    valid = y_true >= 0
    sims = sims[valid]; y_true = y_true[valid]

    if len(y_true) == 0:
        return {"n_test": 0}

    pred = sims.argmax(axis=1)
    return {
        "n_test": int(len(y_true)),
        "n_classes_l3": int(len(l3_to_id)),
        "Acc@1": float(accuracy_score(y_true, pred)),
        "Acc@3": float(topk_acc(y_true, sims, 3)),
        "Acc@5": float(topk_acc(y_true, sims, 5)),
        "BAcc":  float(balanced_accuracy_score(y_true, pred)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Carga rows
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)
    print(f"Total filas: {len(rows)}")

    tr_rows = [r for r in rows if r["split"] == "train"]
    va_rows = [r for r in rows if r["split"] == "val"]
    te_rows = [r for r in rows if r["split"] == "test"]
    print(f"train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)}")

    # Mapeos
    l2_classes = sorted({r["ontology_l2"] for r in tr_rows})
    l2_to_id = {c: i for i, c in enumerate(l2_classes)}
    n_l2 = len(l2_classes)
    print(f"L2 classes: {n_l2}")

    l3_classes = sorted({r["ontology_l3"] for r in rows})
    l3_to_id = {c: i for i, c in enumerate(l3_classes)}
    n_l3 = len(l3_classes)
    print(f"L3 classes total: {n_l3}")

    # Class weights L2
    counts = np.bincount([l2_to_id[r["ontology_l2"]] for r in tr_rows], minlength=n_l2)
    print(f"L2 counts top-5: {Counter(dict(zip(l2_classes, counts.tolist()))).most_common(5)}")
    class_weights = np.where(counts > 0,
                              len(tr_rows) / (n_l2 * np.maximum(counts, 1)),
                              0.0)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    # Datasets
    tr_ds = DermaSet(tr_rows, T_TRAIN, l2_to_id)
    va_ds = DermaSet(va_rows, T_EVAL, l2_to_id)
    te_ds = DermaSet(te_rows, T_EVAL, l2_to_id)

    # Filtrar val/test: solo filas con L2 conocida en train
    va_ds.rows = [r for r in va_ds.rows if r["ontology_l2"] in l2_to_id]
    te_ds.rows = [r for r in te_ds.rows if r["ontology_l2"] in l2_to_id]
    print(f"val (con L2 visible): {len(va_ds.rows)}, test: {len(te_ds.rows)}")

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)

    # Modelo
    print("\n=== Building SpanDerm v0 ===")
    model = build_spanderm_v0(n_l2)

    # Optimizer
    head_params = list(model.head.parameters())
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]
    print(f"Head params: {sum(p.numel() for p in head_params)/1e3:.1f} K")
    print(f"LoRA params: {sum(p.numel() for p in lora_params)/1e6:.2f} M")

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD},
        {"params": lora_params, "lr": LR_LORA},
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

    # Train
    history = []
    best_val = -1
    best_state = None
    t_start = time.time()
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, tr_loader, optimizer, scheduler, loss_fn)
        y_v, yp_v, prob_v, _ = evaluate(model, va_loader, n_l2)
        y_t, yp_t, prob_t, _ = evaluate(model, te_loader, n_l2)
        m_v = metrics_l2(y_v, yp_v, prob_v, n_l2)
        m_t = metrics_l2(y_t, yp_t, prob_t, n_l2)
        elapsed = time.time() - t0
        history.append({"epoch": epoch+1, "loss": train_loss,
                        "val": m_v, "test": m_t, "secs": elapsed})
        print(f"  ep{epoch+1:2d} | loss={train_loss:.4f} | "
              f"val BAcc={m_v['BAcc']:.3f} AUROC={m_v['AUROC']:.3f} | "
              f"test BAcc={m_t['BAcc']:.3f} AUROC={m_t['AUROC']:.3f} | "
              f"{elapsed:.1f}s")

        if m_v["BAcc"] > best_val:
            best_val = m_v["BAcc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nTotal training: {(time.time()-t_start)/60:.1f} min")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Eval final L2
    y_t, yp_t, prob_t, feats_t = evaluate(model, te_loader, n_l2)
    final_l2 = metrics_l2(y_t, yp_t, prob_t, n_l2)
    print(f"\n=== Test FINAL L2 (best val BAcc={best_val:.3f}) ===")
    print(f"  Acc@1={final_l2['Acc@1']:.4f}  BAcc={final_l2['BAcc']:.4f}  "
          f"AUROC={final_l2['AUROC']:.4f}  Acc@3={final_l2['Acc@3']:.4f}")

    # Bootstrap IC95
    print("\nBootstrap IC95% L2...")
    ci = {}
    for k in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa"):
        m, lo, hi = bootstrap_metric(y_t, yp_t, prob_t, n_l2, k)
        ci[k] = {"value": round(final_l2[k], 4),
                 "ci95_low": round(lo, 4) if lo else None,
                 "ci95_high": round(hi, 4) if hi else None}
        print(f"  {k}: {final_l2[k]:.4f} [{lo:.4f}, {hi:.4f}]")

    # Ranking L3
    print("\n=== Ranking L3 por similitud coseno ===")
    print("  Computando prototipos L3 con embeddings train...")
    t0 = time.time()
    protos = compute_l3_prototypes(model, tr_rows, l3_to_id)
    print(f"  Prototipos: {protos.shape} en {time.time()-t0:.1f}s")
    l3_metrics = eval_l3_ranking(model, te_rows, l3_to_id, protos)
    print(f"  L3 ranking: Acc@1={l3_metrics.get('Acc@1',0):.4f}  "
          f"Acc@3={l3_metrics.get('Acc@3',0):.4f}  "
          f"Acc@5={l3_metrics.get('Acc@5',0):.4f}  "
          f"BAcc={l3_metrics.get('BAcc',0):.4f}")

    # Save
    results = {
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr_head": LR_HEAD, "lr_lora": LR_LORA,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_blocks": list(range(24-N_LAST_BLOCKS_TO_ADAPT, 24)),
        },
        "n_train": len(tr_rows), "n_val_used": len(va_ds.rows),
        "n_test_used": len(te_ds.rows),
        "n_classes_l2": n_l2, "n_classes_l3": n_l3,
        "best_val_bacc_l2": float(best_val),
        "final_test_l2": ci,
        "l3_ranking_test": l3_metrics,
        "history": history,
    }
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ {OUT_DIR / 'results.json'}")

    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "metric", "value", "ci_low", "ci_high"])
        for k, v in ci.items():
            w.writerow(["L2", k, v["value"], v["ci95_low"], v["ci95_high"]])
        for k, v in l3_metrics.items():
            if k.startswith("Acc") or k == "BAcc":
                w.writerow(["L3_ranking", k, v, "", ""])
    print(f"✓ {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    from collections import Counter
    main()
