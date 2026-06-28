# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_spanderm_v0_multiseed.py · SpanDerm v0 con 3 seeds para robustez

Mismo protocolo que dermapixel_spanderm_v0.py pero:
  - 3 seeds (42, 43, 44)
  - Selección de checkpoint: best val BAcc + cifra alternativa "media últimas 5 épocas"
  - Reporta media ± std entre seeds para cada métrica
  - L3 ranking promediado entre seeds
"""
from __future__ import annotations
import csv
import json
import math
import os
import sys
import time
import warnings
import random
from collections import Counter, defaultdict
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
OUT_DIR  = ROOT / "output" / "dermapixel_v1_spanderm_v0_multiseed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classification"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15
BATCH_SIZE = 16
LR_HEAD = 1e-3
LR_LORA = 5e-4
WEIGHT_DECAY = 1e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
N_LAST_BLOCKS_TO_ADAPT = 2
SEEDS = [42, 43, 44]

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class FeatExtractSet(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows; self.transform = transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(DATASET / r["image_path"]).convert("RGB")
        return self.transform(img), idx


class SpanDermV0(nn.Module):
    def __init__(self, encoder, n_classes_l2, dim=1024):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, n_classes_l2)
    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats), feats


def build_spanderm_v0(n_classes_l2):
    from models.modeling_finetune import panderm_large_patch16_224
    from peft import LoraConfig, get_peft_model
    sd = torch.load(WEIGHTS / "panderm_large.pth", map_location="cpu", weights_only=False)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    encoder = panderm_large_patch16_224()
    encoder.load_state_dict(sd, strict=False)
    encoder.head = nn.Identity()
    for p in encoder.parameters():
        p.requires_grad = False
    n_blocks = len(encoder.blocks)
    target_modules = []
    for b in range(n_blocks - N_LAST_BLOCKS_TO_ADAPT, n_blocks):
        for sub in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            target_modules.append(f"blocks.{b}.{sub}")
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, bias="none",
    )
    encoder = get_peft_model(encoder, lora_cfg)
    return SpanDermV0(encoder, n_classes_l2).to(DEVICE)


def safe_auroc(yt, prob, n_cls):
    aurocs = []
    for c in range(n_cls):
        if (yt == c).sum() == 0 or (yt != c).sum() == 0: continue
        try: aurocs.append(roc_auc_score((yt == c).astype(int), prob[:, c]))
        except ValueError: continue
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


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_y, all_yp, all_prob = [], [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            logits = logits.float()
        prob = F.softmax(logits, dim=-1).cpu().numpy()
        all_y.append(y.numpy()); all_prob.append(prob); all_yp.append(prob.argmax(axis=1))
    return np.concatenate(all_y), np.concatenate(all_yp), np.concatenate(all_prob)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = loss_fn(logits.float(), y)
        loss.backward(); optimizer.step(); scheduler.step()
        total_loss += loss.item() * x.size(0); total_n += x.size(0)
    return total_loss / total_n


@torch.no_grad()
def extract_features(model, rows):
    model.eval()
    ds = FeatExtractSet(rows, T_EVAL)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    all_feats = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, feats = model(x); feats = feats.float()
        feats_n = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats_n.cpu().numpy())
    return np.vstack(all_feats)


def l3_ranking(model, tr_rows, te_rows, l3_to_id):
    tr_feats = extract_features(model, tr_rows)
    tr_l3 = np.array([l3_to_id[r["ontology_l3"]] for r in tr_rows])
    n_cls = len(l3_to_id); dim = tr_feats.shape[1]
    protos = np.zeros((n_cls, dim))
    for c in range(n_cls):
        mask = tr_l3 == c
        if mask.sum() == 0: continue
        protos[c] = tr_feats[mask].mean(axis=0)
    norms = np.linalg.norm(protos, axis=1, keepdims=True)
    protos = protos / np.maximum(norms, 1e-8)

    te_feats = extract_features(model, te_rows)
    sims = te_feats @ protos.T
    y_true = np.array([l3_to_id.get(r["ontology_l3"], -1) for r in te_rows])
    valid = y_true >= 0
    sims = sims[valid]; y_true = y_true[valid]
    pred = sims.argmax(axis=1)
    return {
        "Acc@1": float(accuracy_score(y_true, pred)),
        "Acc@3": float(topk_acc(y_true, sims, 3)),
        "Acc@5": float(topk_acc(y_true, sims, 5)),
        "BAcc":  float(balanced_accuracy_score(y_true, pred)),
        "n_test": int(len(y_true)),
    }


def train_one_seed(seed, tr_rows, va_rows, te_rows, l2_to_id, l3_to_id, class_weights_t):
    set_seed(seed)
    n_l2 = len(l2_to_id)

    tr_ds = DermaSet(tr_rows, T_TRAIN, l2_to_id)
    va_ds = DermaSet([r for r in va_rows if r["ontology_l2"] in l2_to_id], T_EVAL, l2_to_id)
    te_ds = DermaSet([r for r in te_rows if r["ontology_l2"] in l2_to_id], T_EVAL, l2_to_id)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_spanderm_v0(n_l2)
    head_params = list(model.head.parameters())
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD},
        {"params": lora_params, "lr": LR_LORA},
    ], weight_decay=WEIGHT_DECAY)
    n_steps = len(tr_loader) * EPOCHS
    warmup_steps = len(tr_loader)
    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    history = []
    best_val_bacc = -1
    best_state = None
    for ep in range(EPOCHS):
        loss = train_one_epoch(model, tr_loader, optimizer, scheduler, loss_fn)
        y_v, yp_v, prob_v = evaluate(model, va_loader)
        y_t, yp_t, prob_t = evaluate(model, te_loader)
        m_v = metrics_l2(y_v, yp_v, prob_v, n_l2)
        m_t = metrics_l2(y_t, yp_t, prob_t, n_l2)
        history.append({"epoch": ep+1, "loss": loss, "val": m_v, "test": m_t})
        if m_v["BAcc"] > best_val_bacc:
            best_val_bacc = m_v["BAcc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    seed{seed} ep{ep+1:2d} loss={loss:.3f} val_BAcc={m_v['BAcc']:.3f} test_BAcc={m_t['BAcc']:.3f} test_AUROC={m_t['AUROC']:.3f}")

    # Cifra 1: best val checkpoint
    model.load_state_dict(best_state)
    y_t, yp_t, prob_t = evaluate(model, te_loader)
    test_best_val = metrics_l2(y_t, yp_t, prob_t, n_l2)

    # Cifra 2: media de últimas 5 épocas
    last5 = history[-5:]
    test_last5_mean = {
        k: float(np.mean([h["test"][k] for h in last5]))
        for k in ("Acc@1", "Acc@3", "BAcc", "AUROC", "W-F1", "Kappa")
    }

    # L3 ranking sobre best val checkpoint
    l3_metrics = l3_ranking(model, tr_rows, te_rows, l3_to_id)

    del model
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "test_best_val": test_best_val,
        "test_last5_mean": test_last5_mean,
        "l3_ranking": l3_metrics,
        "history": history,
    }


def aggregate_metrics(per_seed, key):
    """Agrega métricas entre seeds: media ± std."""
    keys = per_seed[0][key].keys()
    agg = {}
    for k in keys:
        if not isinstance(per_seed[0][key][k], (int, float)): continue
        vals = np.array([s[key][k] for s in per_seed if isinstance(s[key].get(k), (int, float))
                         and not np.isnan(s[key][k])])
        if len(vals) == 0:
            agg[k] = {"mean": None, "std": None}
            continue
        agg[k] = {"mean": round(float(vals.mean()), 4),
                  "std":  round(float(vals.std()), 4),
                  "n_seeds": int(len(vals))}
    return agg


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

    l2_classes = sorted({r["ontology_l2"] for r in tr_rows})
    l2_to_id = {c: i for i, c in enumerate(l2_classes)}
    n_l2 = len(l2_classes)
    l3_classes = sorted({r["ontology_l3"] for r in rows})
    l3_to_id = {c: i for i, c in enumerate(l3_classes)}
    n_l3 = len(l3_classes)
    print(f"L2 classes: {n_l2}, L3 classes: {n_l3}")

    counts = np.bincount([l2_to_id[r["ontology_l2"]] for r in tr_rows], minlength=n_l2)
    class_weights = np.where(counts > 0, len(tr_rows) / (n_l2 * np.maximum(counts, 1)), 0.0)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    # Train 3 seeds
    per_seed = []
    t_start = time.time()
    for seed in SEEDS:
        print(f"\n========== SEED {seed} ==========")
        result = train_one_seed(seed, tr_rows, va_rows, te_rows, l2_to_id, l3_to_id, class_weights_t)
        per_seed.append(result)
        print(f"  best_val test BAcc={result['test_best_val']['BAcc']:.4f} AUROC={result['test_best_val']['AUROC']:.4f}")
        print(f"  last5_mean test BAcc={result['test_last5_mean']['BAcc']:.4f} AUROC={result['test_last5_mean']['AUROC']:.4f}")
        print(f"  L3 ranking Acc@1={result['l3_ranking']['Acc@1']:.4f} Acc@3={result['l3_ranking']['Acc@3']:.4f}")

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal training time: {elapsed:.1f} min")

    # Aggregate
    agg_best_val = aggregate_metrics(per_seed, "test_best_val")
    agg_last5 = aggregate_metrics(per_seed, "test_last5_mean")
    agg_l3 = aggregate_metrics(per_seed, "l3_ranking")

    print("\n=== AGREGADO 3 SEEDS ===")
    print("Best val checkpoint:")
    for k, v in agg_best_val.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
    print("\nMedia últimas 5 épocas:")
    for k, v in agg_last5.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
    print("\nL3 ranking:")
    for k, v in agg_l3.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Save
    results = {
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr_head": LR_HEAD, "lr_lora": LR_LORA,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "seeds": SEEDS,
            "n_classes_l2": n_l2, "n_classes_l3": n_l3,
            "n_train": len(tr_rows), "n_test": len(te_rows),
        },
        "aggregate_best_val": agg_best_val,
        "aggregate_last5_mean": agg_last5,
        "aggregate_l3_ranking": agg_l3,
        "per_seed": per_seed,
        "total_time_min": elapsed,
    }
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ {OUT_DIR / 'results.json'}")

    # CSV summary
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["selection", "metric", "mean", "std", "n_seeds"])
        for k, v in agg_best_val.items():
            w.writerow(["best_val", k, v["mean"], v["std"], v["n_seeds"]])
        for k, v in agg_last5.items():
            w.writerow(["last5_mean", k, v["mean"], v["std"], v["n_seeds"]])
        for k, v in agg_l3.items():
            w.writerow(["l3_ranking", k, v["mean"], v["std"], v["n_seeds"]])
    print(f"✓ {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
