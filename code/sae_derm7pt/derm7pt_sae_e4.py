# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
derm7pt_sae_e4.py · E4 · Fine-tuning multitarea PanDerm Large

Réplica conceptual de Kawahara et al. 2019 con encoder fundacional:
  - Encoder PanDerm Large (1024d) con LoRA r=16 últimas 2 capas (igual SpanDerm v0)
  - 8 cabezas:
     7 cabezas conceptos (clasificación multi-clase por criterio Derm7pt)
     1 cabeza melanoma binario
  - Loss: suma de CE de las 8 cabezas (weighted opcional)

Splits oficiales: train 413, val 203, test 395
Métricas multitarea + AUROC test por cabeza + bootstrap IC95%

Comparativa con E2 (CBM sobre SAE) y baselines.

Salida: $DERMAPIXEL_ROOT/output/derm7pt_sae_e4/{results.json, summary.csv, report.md}
"""
from __future__ import annotations
import csv
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, cohen_kappa_score,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DERM7PT_DIR = ROOT / "dermfm_zero/data/PanDerm-2-Eval/multimodal_finetune/multimodal_finetune/derm7pt"
META_CSV = DERM7PT_DIR / "meta" / "meta.csv"
IMG_DIR  = DERM7PT_DIR / "images"
TRAIN_IDX_CSV = DERM7PT_DIR / "meta" / "train_indexes.csv"
VALID_IDX_CSV = DERM7PT_DIR / "meta" / "valid_indexes.csv"
TEST_IDX_CSV  = DERM7PT_DIR / "meta" / "test_indexes.csv"
WEIGHTS = ROOT / "weights" / "weights"
OUT_DIR = ROOT / "output" / "derm7pt_sae_e4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classification"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15
BATCH_SIZE = 16
LR_HEADS = 1e-3
LR_LORA = 5e-4
WEIGHT_DECAY = 1e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
N_LAST_BLOCKS = 2
SEED = 42

# 7 criterios + valores admisibles
CRIT_VALS = {
    "pigment_network":       ["absent", "typical", "atypical"],
    "streaks":               ["absent", "regular", "irregular"],
    "pigmentation":          ["absent", "diffuse regular", "diffuse irregular",
                              "localized regular", "localized irregular"],
    "regression_structures": ["absent", "blue areas", "combinations", "white areas"],
    "dots_and_globules":     ["absent", "regular", "irregular"],
    "blue_whitish_veil":     ["absent", "present"],
    "vascular_structures":   ["absent", "arborizing", "comma", "dotted",
                              "hairpin", "linear irregular", "within regression", "wreath"],
}
SEVEN_PT = list(CRIT_VALS.keys())

MELANOMA_DXS = {"melanoma", "melanoma (in situ)", "melanoma (less than 0.76 mm)",
                "melanoma (0.76 to 1.5 mm)", "melanoma (more than 1.5 mm)"}


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


def load_meta():
    rows = []
    with META_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_idx_csv(p):
    out = []
    with p.open() as f:
        for r in csv.reader(f):
            if r and r[0] != "indexes":
                try: out.append(int(r[0]))
                except ValueError: continue
    return out


class Derm7Set(Dataset):
    def __init__(self, rows, transform, val2id):
        self.rows = rows
        self.transform = transform
        self.val2id = val2id  # dict crit -> {value: id}
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(IMG_DIR / r["derm"]).convert("RGB")
        x = self.transform(img)
        # 7 conceptos
        y_concepts = torch.tensor([
            self.val2id[c].get(r[c], 0) for c in SEVEN_PT
        ], dtype=torch.long)
        # melanoma
        y_mel = torch.tensor(1 if r["diagnosis"] in MELANOMA_DXS else 0,
                              dtype=torch.long)
        return x, y_concepts, y_mel


# -----------------------------------------------------------------------------
# Modelo multitarea
# -----------------------------------------------------------------------------

class MultitaskDerm7(nn.Module):
    def __init__(self, encoder, n_per_crit, dim=1024):
        """encoder: PanDerm Large + LoRA.
        n_per_crit: lista de 7 ints con n_classes por criterio."""
        super().__init__()
        self.encoder = encoder
        self.heads_concept = nn.ModuleList([
            nn.Linear(dim, n) for n in n_per_crit
        ])
        self.head_mel = nn.Linear(dim, 2)

    def forward(self, x):
        feats = self.encoder(x)
        logits_concepts = [h(feats) for h in self.heads_concept]
        logits_mel = self.head_mel(feats)
        return logits_concepts, logits_mel, feats


def build_model():
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
    for b in range(n_blocks - N_LAST_BLOCKS, n_blocks):
        for sub in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            target_modules.append(f"blocks.{b}.{sub}")
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, bias="none",
    )
    encoder = get_peft_model(encoder, lora_cfg)
    n_per_crit = [len(CRIT_VALS[c]) for c in SEVEN_PT]
    model = MultitaskDerm7(encoder, n_per_crit).to(DEVICE)
    return model


# -----------------------------------------------------------------------------
# Train / Eval
# -----------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler,
                concept_weights, mel_weight, lambda_concept=1.0):
    model.train()
    total_loss, total_n = 0.0, 0
    for x, y_concepts, y_mel in loader:
        x = x.to(DEVICE, non_blocking=True)
        y_concepts = y_concepts.to(DEVICE, non_blocking=True)
        y_mel = y_mel.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_c, logits_m, _ = model(x)
            losses_c = []
            for i, lc in enumerate(logits_c):
                w = concept_weights[i].to(lc.device)
                losses_c.append(F.cross_entropy(lc.float(), y_concepts[:, i], weight=w))
            loss_c = sum(losses_c) / len(losses_c)
            loss_m = F.cross_entropy(logits_m.float(), y_mel, weight=mel_weight.to(logits_m.device))
            loss = lambda_concept * loss_c + loss_m

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_y_c, all_pred_c, all_prob_c = [[] for _ in SEVEN_PT], [[] for _ in SEVEN_PT], [[] for _ in SEVEN_PT]
    all_y_m, all_pred_m, all_prob_m = [], [], []
    for x, y_c, y_m in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_c, logits_m, _ = model(x)
        for i, lc in enumerate(logits_c):
            p = F.softmax(lc.float(), dim=-1).cpu().numpy()
            all_y_c[i].append(y_c[:, i].numpy())
            all_pred_c[i].append(p.argmax(axis=1))
            all_prob_c[i].append(p)
        pm = F.softmax(logits_m.float(), dim=-1).cpu().numpy()
        all_y_m.append(y_m.numpy())
        all_pred_m.append(pm.argmax(axis=1))
        all_prob_m.append(pm[:, 1])

    def concat(L): return np.concatenate(L)
    yc = [concat(L) for L in all_y_c]
    pc = [concat(L) for L in all_pred_c]
    probc = [np.vstack(L) for L in all_prob_c]
    ym = concat(all_y_m); pm = concat(all_pred_m); probm = concat(all_prob_m)
    return yc, pc, probc, ym, pm, probm


def metrics_concept(y, p, prob):
    n_cls = prob.shape[1]
    out = {"Acc@1": float(accuracy_score(y, p)),
           "BAcc":  float(balanced_accuracy_score(y, p)),
           "W-F1":  float(f1_score(y, p, average="weighted", zero_division=0))}
    # AUROC: present (any != absent) binarizado
    y_bin = (y != 0).astype(int)
    prob_present = prob[:, 1:].sum(axis=1) if prob.shape[1] > 1 else prob[:, 0]
    try:
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            out["AUROC_present"] = float(roc_auc_score(y_bin, prob_present))
        else:
            out["AUROC_present"] = float("nan")
    except ValueError:
        out["AUROC_present"] = float("nan")
    return out


def metrics_mel(y, p, prob):
    out = {
        "Acc@1": float(accuracy_score(y, p)),
        "BAcc":  float(balanced_accuracy_score(y, p)),
        "W-F1":  float(f1_score(y, p, average="weighted", zero_division=0)),
        "Kappa": float(cohen_kappa_score(y, p)),
    }
    try:
        out["AUROC"] = float(roc_auc_score(y, prob))
    except ValueError:
        out["AUROC"] = float("nan")
    return out


def bootstrap_ci(y, p, prob, metric_fn, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    vals = []
    for _ in range(n_iter):
        idx = []
        for c in classes:
            ic = np.where(y == c)[0]
            if len(ic) == 0: continue
            idx.extend(rng.choice(ic, size=len(ic), replace=True))
        idx = np.array(idx)
        try:
            v = metric_fn(y[idx], p[idx], prob[idx])
            vals.append(v)
        except (ValueError, IndexError):
            continue
    if not vals: return None, None, None
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    rows = load_meta()
    print(f"Total meta: {len(rows)}")

    tr_idx = load_idx_csv(TRAIN_IDX_CSV)
    va_idx = load_idx_csv(VALID_IDX_CSV)
    te_idx = load_idx_csv(TEST_IDX_CSV)
    print(f"Splits: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    val2id = {c: {v: i for i, v in enumerate(CRIT_VALS[c])} for c in SEVEN_PT}
    n_per_crit = [len(CRIT_VALS[c]) for c in SEVEN_PT]
    print(f"Concept heads: {dict(zip(SEVEN_PT, n_per_crit))}")

    tr_rows = [rows[i] for i in tr_idx]
    va_rows = [rows[i] for i in va_idx]
    te_rows = [rows[i] for i in te_idx]

    # class weights por criterio (balanced)
    concept_weights = []
    for c in SEVEN_PT:
        ys = [val2id[c].get(r[c], 0) for r in tr_rows]
        n_cls = n_per_crit[SEVEN_PT.index(c)]
        counts = np.bincount(ys, minlength=n_cls)
        w = np.where(counts > 0, len(ys) / (n_cls * np.maximum(counts, 1)), 0.0)
        concept_weights.append(torch.tensor(w, dtype=torch.float32))

    # melanoma weights
    y_mel_tr = [1 if r["diagnosis"] in MELANOMA_DXS else 0 for r in tr_rows]
    counts_m = np.bincount(y_mel_tr, minlength=2)
    w_m = len(y_mel_tr) / (2 * np.maximum(counts_m, 1))
    mel_weight = torch.tensor(w_m, dtype=torch.float32)
    print(f"Melanoma train: {sum(y_mel_tr)}/{len(y_mel_tr)} | weights={w_m}")

    tr_ds = Derm7Set(tr_rows, T_TRAIN, val2id)
    va_ds = Derm7Set(va_rows, T_EVAL, val2id)
    te_ds = Derm7Set(te_rows, T_EVAL, val2id)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)

    print("\n=== Build model ===")
    model = build_model()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable/1e6:.2f} M / {n_total/1e6:.1f} M ({100*n_trainable/n_total:.2f}%)")

    head_params = []
    for h in model.heads_concept: head_params += list(h.parameters())
    head_params += list(model.head_mel.parameters())
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEADS},
        {"params": lora_params, "lr": LR_LORA},
    ], weight_decay=WEIGHT_DECAY)

    n_steps = len(tr_loader) * EPOCHS
    warmup_steps = len(tr_loader)
    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Train
    history = []
    best_val_mel_auroc = -1
    best_state = None
    t_start = time.time()
    for ep in range(EPOCHS):
        t0 = time.time()
        loss = train_epoch(model, tr_loader, optimizer, scheduler,
                            concept_weights, mel_weight, lambda_concept=1.0)
        # Eval val
        yc_v, pc_v, probc_v, ym_v, pm_v, probm_v = evaluate(model, va_loader)
        # Eval test
        yc_t, pc_t, probc_t, ym_t, pm_t, probm_t = evaluate(model, te_loader)

        mel_v = metrics_mel(ym_v, pm_v, probm_v)
        mel_t = metrics_mel(ym_t, pm_t, probm_t)

        elapsed = time.time() - t0
        history.append({"epoch": ep+1, "loss": loss,
                        "val_mel": mel_v, "test_mel": mel_t,
                        "secs": elapsed})
        print(f"  ep{ep+1:2d} loss={loss:.3f} | "
              f"val mel AUROC={mel_v['AUROC']:.3f} BAcc={mel_v['BAcc']:.3f} | "
              f"test mel AUROC={mel_t['AUROC']:.3f} BAcc={mel_t['BAcc']:.3f} | "
              f"{elapsed:.1f}s")

        if mel_v["AUROC"] > best_val_mel_auroc:
            best_val_mel_auroc = mel_v["AUROC"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nTotal training: {(time.time()-t_start)/60:.1f} min")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Eval final test
    yc_t, pc_t, probc_t, ym_t, pm_t, probm_t = evaluate(model, te_loader)
    final_mel = metrics_mel(ym_t, pm_t, probm_t)
    print(f"\n=== Test FINAL melanoma (best val AUROC={best_val_mel_auroc:.3f}) ===")
    for k, v in final_mel.items():
        print(f"  {k}: {v:.4f}")

    # Bootstrap melanoma
    print("\nBootstrap IC95% melanoma (1000)...")
    mel_ci = {}
    for key, fn in [
        ("Acc@1", lambda y, p, pp: float(accuracy_score(y, p))),
        ("BAcc",  lambda y, p, pp: float(balanced_accuracy_score(y, p))),
        ("AUROC", lambda y, p, pp: float(roc_auc_score(y, pp))),
        ("W-F1",  lambda y, p, pp: float(f1_score(y, p, average="weighted", zero_division=0))),
        ("Kappa", lambda y, p, pp: float(cohen_kappa_score(y, p))),
    ]:
        m, lo, hi = bootstrap_ci(ym_t, pm_t, probm_t, fn)
        mel_ci[key] = {"value": round(final_mel[key], 4),
                       "ci95_low": round(lo, 4) if lo else None,
                       "ci95_high": round(hi, 4) if hi else None}
        print(f"  {key}: {final_mel[key]:.4f} [{lo:.4f}, {hi:.4f}]")

    # Métricas por concepto en test
    print("\n=== Métricas por concepto (test) ===")
    concept_metrics = {}
    for i, c in enumerate(SEVEN_PT):
        m = metrics_concept(yc_t[i], pc_t[i], probc_t[i])
        concept_metrics[c] = m
        print(f"  {c}: Acc={m['Acc@1']:.3f} BAcc={m['BAcc']:.3f} "
              f"AUROC_present={m['AUROC_present']:.3f}")

    # Comparativa con baselines
    print(f"\n=== Comparativa melanoma test (E2 vs E4) ===")
    e2_results_path = ROOT / "output" / "derm7pt_sae_e2" / "results.json"
    if e2_results_path.exists():
        with e2_results_path.open() as f:
            e2 = json.load(f)
        print(f"  E2 Direct LP SAE:  AUROC={e2['direct_lp_sae']['AUROC']:.4f}")
        print(f"  E2 CBM 7 cpts:     AUROC={e2['cbm_7_concepts']['AUROC']:.4f}")
        print(f"  E4 FT multitarea:  AUROC={final_mel['AUROC']:.4f}")

    # Save
    results = {
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr_heads": LR_HEADS, "lr_lora": LR_LORA,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "n_last_blocks": N_LAST_BLOCKS,
            "n_trainable_M": round(n_trainable/1e6, 2),
            "lambda_concept": 1.0,
        },
        "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
        "best_val_mel_auroc": float(best_val_mel_auroc),
        "final_test_melanoma": mel_ci,
        "concept_metrics_test": concept_metrics,
        "history": history,
    }
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSV summary
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "name", "metric", "value", "ci_low", "ci_high"])
        for k, v in mel_ci.items():
            w.writerow(["melanoma", "mel", k, v["value"], v["ci95_low"], v["ci95_high"]])
        for c, m in concept_metrics.items():
            for k, v in m.items():
                w.writerow(["concept", c, k, round(v, 4) if isinstance(v, float) else v, "", ""])
    print(f"\n✓ {OUT_DIR / 'results.json'}")
    print(f"✓ {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
