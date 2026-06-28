# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
persist_models_for_prototype.py · Reentrena SpanDerm v0 + E4 y persiste
los checkpoints en disco para integrar al servicio dermapixel-server.

Outputs:
  $DERMAPIXEL_ROOT/output/dermapixel_v1_spanderm_v0_multiseed/
    ├── best_seed42.pth            (state_dict completo encoder LoRA + head FC L2)
    ├── best_seed42_l2_mapping.json (id↔nombre L2)
    └── best_seed42_metrics.json   (métricas test del best val checkpoint)

  $DERMAPIXEL_ROOT/output/derm7pt_sae_e4/
    ├── best_model.pth             (state_dict 8 heads + LoRA)
    ├── best_concept_mapping.json  (CRIT_VALS por concepto)
    └── best_metrics.json          (AUROC melanoma + por concepto)

Reutiliza la lógica de dermapixel_spanderm_v0_multiseed.py y derm7pt_sae_e4.py
pero solo seed=42 y con persistencia del best_state.
"""
from __future__ import annotations
import csv
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
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
DERM7PT_DIR = ROOT / "dermfm_zero/data/PanDerm-2-Eval/multimodal_finetune/multimodal_finetune/derm7pt"
WEIGHTS  = ROOT / "weights" / "weights"
SPDERM_OUT = ROOT / "output" / "dermapixel_v1_spanderm_v0_multiseed"
E4_OUT   = ROOT / "output" / "derm7pt_sae_e4"
SPDERM_OUT.mkdir(parents=True, exist_ok=True)
E4_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classification"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}

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


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ====================================================================
# SpanDerm v0 LoRA L2
# ====================================================================

EPOCHS_SP = 15; BS_SP = 16; LR_HEAD_SP = 1e-3; LR_LORA_SP = 5e-4
WD_SP = 1e-4; LORA_R = 16; LORA_ALPHA = 32; LORA_DROPOUT = 0.1
N_LAST_BLOCKS = 2


class DermaSetL2(Dataset):
    def __init__(self, rows, transform, l2_to_id):
        self.rows = rows; self.transform = transform; self.l2_to_id = l2_to_id
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(DATASET / r["image_path"]).convert("RGB")
        x = self.transform(img)
        y = self.l2_to_id[r["ontology_l2"]]
        return x, y


class SpanDermV0(nn.Module):
    def __init__(self, encoder, n_classes_l2, dim=1024):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, n_classes_l2)
    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats), feats


def build_spanderm(n_classes):
    from models.modeling_finetune import panderm_large_patch16_224
    from peft import LoraConfig, get_peft_model
    sd = torch.load(WEIGHTS / "panderm_large.pth", map_location="cpu", weights_only=False)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    encoder = panderm_large_patch16_224()
    encoder.load_state_dict(sd, strict=False)
    encoder.head = nn.Identity()
    for p in encoder.parameters(): p.requires_grad = False

    target_modules = []
    n_blocks = len(encoder.blocks)
    for b in range(n_blocks - N_LAST_BLOCKS, n_blocks):
        for sub in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            target_modules.append(f"blocks.{b}.{sub}")
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, bias="none",
    )
    encoder = get_peft_model(encoder, lora_cfg)
    return SpanDermV0(encoder, n_classes).to(DEVICE)


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


def metrics_l2_full(yt, yp, prob, n_cls):
    return {
        "Acc@1": float(accuracy_score(yt, yp)),
        "Acc@3": float(topk_acc(yt, prob, 3)),
        "BAcc":  float(balanced_accuracy_score(yt, yp)),
        "AUROC": float(safe_auroc(yt, prob, n_cls)),
        "W-F1":  float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "Kappa": float(cohen_kappa_score(yt, yp)),
    }


@torch.no_grad()
def eval_spanderm(model, loader):
    model.eval()
    ys, yps, probs = [], [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x); logits = logits.float()
        p = F.softmax(logits, dim=-1).cpu().numpy()
        ys.append(y.numpy()); yps.append(p.argmax(axis=1)); probs.append(p)
    return np.concatenate(ys), np.concatenate(yps), np.concatenate(probs)


def train_spanderm():
    print(f"\n{'='*60}\n=== Reentrenar SpanDerm v0 (seed=42) con persistencia ===\n{'='*60}")
    set_seed(42)

    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)

    tr_rows = [r for r in rows if r["split"] == "train"]
    va_rows = [r for r in rows if r["split"] == "val"]
    te_rows = [r for r in rows if r["split"] == "test"]

    l2_classes = sorted({r["ontology_l2"] for r in tr_rows})
    l2_to_id = {c: i for i, c in enumerate(l2_classes)}
    n_l2 = len(l2_to_id)
    va_rows = [r for r in va_rows if r["ontology_l2"] in l2_to_id]
    te_rows = [r for r in te_rows if r["ontology_l2"] in l2_to_id]
    print(f"train={len(tr_rows)} val={len(va_rows)} test={len(te_rows)} | L2 classes: {n_l2}")

    counts = np.bincount([l2_to_id[r["ontology_l2"]] for r in tr_rows], minlength=n_l2)
    cw = np.where(counts > 0, len(tr_rows) / (n_l2 * np.maximum(counts, 1)), 0.0)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

    tr_ds = DermaSetL2(tr_rows, T_TRAIN, l2_to_id)
    va_ds = DermaSetL2(va_rows, T_EVAL, l2_to_id)
    te_ds = DermaSetL2(te_rows, T_EVAL, l2_to_id)
    tr_loader = DataLoader(tr_ds, batch_size=BS_SP, shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BS_SP, shuffle=False, num_workers=4)
    te_loader = DataLoader(te_ds, batch_size=BS_SP, shuffle=False, num_workers=4)

    model = build_spanderm(n_l2)
    head_params = list(model.head.parameters())
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora" in n.lower()]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_trainable/1e6:.2f} M params (LoRA + head FC)")

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD_SP},
        {"params": lora_params, "lr": LR_LORA_SP},
    ], weight_decay=WD_SP)
    n_steps = len(tr_loader) * EPOCHS_SP
    warmup = len(tr_loader)
    def lrl(step):
        if step < warmup: return step / max(1, warmup)
        p = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrl)
    loss_fn = nn.CrossEntropyLoss(weight=cw_t)

    best_val_bacc = -1
    best_state = None
    for ep in range(EPOCHS_SP):
        model.train()
        total_loss, total_n = 0.0, 0
        for x, y in tr_loader:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = loss_fn(logits.float(), y)
            loss.backward(); optimizer.step(); scheduler.step()
            total_loss += loss.item() * x.size(0); total_n += x.size(0)
        avg_loss = total_loss / total_n

        y_v, yp_v, prob_v = eval_spanderm(model, va_loader)
        y_t, yp_t, prob_t = eval_spanderm(model, te_loader)
        m_v = metrics_l2_full(y_v, yp_v, prob_v, n_l2)
        m_t = metrics_l2_full(y_t, yp_t, prob_t, n_l2)
        print(f"  ep{ep+1:2d} loss={avg_loss:.3f} val BAcc={m_v['BAcc']:.3f} "
              f"test BAcc={m_t['BAcc']:.3f} AUROC={m_t['AUROC']:.3f}")
        if m_v["BAcc"] > best_val_bacc:
            best_val_bacc = m_v["BAcc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Cargar best y evaluar
    model.load_state_dict(best_state)
    y_t, yp_t, prob_t = eval_spanderm(model, te_loader)
    final = metrics_l2_full(y_t, yp_t, prob_t, n_l2)
    print(f"\n=== SpanDerm v0 FINAL (best val BAcc={best_val_bacc:.3f}) ===")
    for k, v in final.items(): print(f"  {k}: {v:.4f}")

    # Persistir
    ckpt_path = SPDERM_OUT / "best_seed42.pth"
    torch.save({
        "state_dict": best_state,
        "n_classes_l2": n_l2,
        "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                        "n_last_blocks": N_LAST_BLOCKS},
        "best_val_bacc": float(best_val_bacc),
        "test_metrics": final,
        "trainable_params_M": round(n_trainable/1e6, 3),
        "epoch_count": EPOCHS_SP,
        "seed": 42,
    }, ckpt_path)
    print(f"  ✓ {ckpt_path} ({ckpt_path.stat().st_size/1e6:.1f} MB)")

    # Persistir mapping L2
    mapping_path = SPDERM_OUT / "best_seed42_l2_mapping.json"
    id_to_l2 = {i: c for c, i in l2_to_id.items()}
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump({"l2_to_id": l2_to_id, "id_to_l2": id_to_l2}, f, indent=2,
                  ensure_ascii=False)
    print(f"  ✓ {mapping_path}")

    # Persistir metrics
    metrics_path = SPDERM_OUT / "best_seed42_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"final_test": final, "best_val_bacc": float(best_val_bacc)},
                  f, indent=2, ensure_ascii=False)
    print(f"  ✓ {metrics_path}")

    del model; torch.cuda.empty_cache()
    return final


# ====================================================================
# E4 SAE-Derm7pt multitarea
# ====================================================================

EPOCHS_E4 = 15; BS_E4 = 16; LR_HEADS_E4 = 1e-3; LR_LORA_E4 = 5e-4

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


class Derm7Set(Dataset):
    def __init__(self, rows, transform, val2id):
        self.rows = rows; self.transform = transform; self.val2id = val2id
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(DERM7PT_DIR / "images" / r["derm"]).convert("RGB")
        x = self.transform(img)
        y_c = torch.tensor([self.val2id[c].get(r[c], 0) for c in SEVEN_PT], dtype=torch.long)
        y_m = torch.tensor(1 if r["diagnosis"] in MELANOMA_DXS else 0, dtype=torch.long)
        return x, y_c, y_m


class MultitaskDerm7(nn.Module):
    def __init__(self, encoder, n_per_crit, dim=1024):
        super().__init__()
        self.encoder = encoder
        self.heads_concept = nn.ModuleList([nn.Linear(dim, n) for n in n_per_crit])
        self.head_mel = nn.Linear(dim, 2)
    def forward(self, x):
        feats = self.encoder(x)
        return [h(feats) for h in self.heads_concept], self.head_mel(feats), feats


def build_e4_model():
    from models.modeling_finetune import panderm_large_patch16_224
    from peft import LoraConfig, get_peft_model
    sd = torch.load(WEIGHTS / "panderm_large.pth", map_location="cpu", weights_only=False)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    encoder = panderm_large_patch16_224()
    encoder.load_state_dict(sd, strict=False)
    encoder.head = nn.Identity()
    for p in encoder.parameters(): p.requires_grad = False
    target_modules = []
    n_blocks = len(encoder.blocks)
    for b in range(n_blocks - N_LAST_BLOCKS, n_blocks):
        for sub in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            target_modules.append(f"blocks.{b}.{sub}")
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                          lora_dropout=LORA_DROPOUT,
                          target_modules=target_modules, bias="none")
    encoder = get_peft_model(encoder, lora_cfg)
    n_per_crit = [len(CRIT_VALS[c]) for c in SEVEN_PT]
    return MultitaskDerm7(encoder, n_per_crit).to(DEVICE), n_per_crit


@torch.no_grad()
def eval_e4(model, loader):
    model.eval()
    all_y_m, all_pred_m, all_prob_m = [], [], []
    for x, y_c, y_m in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, logits_m, _ = model(x)
        pm = F.softmax(logits_m.float(), dim=-1).cpu().numpy()
        all_y_m.append(y_m.numpy()); all_pred_m.append(pm.argmax(axis=1))
        all_prob_m.append(pm[:, 1])
    ym = np.concatenate(all_y_m); pm = np.concatenate(all_pred_m); probm = np.concatenate(all_prob_m)
    auroc = float(roc_auc_score(ym, probm)) if len(np.unique(ym)) > 1 else float("nan")
    return ym, pm, probm, auroc


def load_idx_csv(p):
    out = []
    with p.open() as f:
        for r in csv.reader(f):
            if r and r[0] != "indexes":
                try: out.append(int(r[0]))
                except ValueError: continue
    return out


def train_e4():
    print(f"\n{'='*60}\n=== Reentrenar E4 multitarea con persistencia ===\n{'='*60}")
    set_seed(42)
    META_CSV = DERM7PT_DIR / "meta" / "meta.csv"
    rows = []
    with META_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f): rows.append(r)
    tr_idx = load_idx_csv(DERM7PT_DIR / "meta" / "train_indexes.csv")
    va_idx = load_idx_csv(DERM7PT_DIR / "meta" / "valid_indexes.csv")
    te_idx = load_idx_csv(DERM7PT_DIR / "meta" / "test_indexes.csv")
    val2id = {c: {v: i for i, v in enumerate(CRIT_VALS[c])} for c in SEVEN_PT}
    print(f"train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    tr_rows = [rows[i] for i in tr_idx]
    va_rows = [rows[i] for i in va_idx]
    te_rows = [rows[i] for i in te_idx]

    cw_list = []
    for c in SEVEN_PT:
        ys = [val2id[c].get(r[c], 0) for r in tr_rows]
        n = len(CRIT_VALS[c])
        cnt = np.bincount(ys, minlength=n)
        w = np.where(cnt > 0, len(ys) / (n * np.maximum(cnt, 1)), 0.0)
        cw_list.append(torch.tensor(w, dtype=torch.float32))
    y_mel_tr = [1 if r["diagnosis"] in MELANOMA_DXS else 0 for r in tr_rows]
    cnt_m = np.bincount(y_mel_tr, minlength=2)
    w_m = len(y_mel_tr) / (2 * np.maximum(cnt_m, 1))
    mw = torch.tensor(w_m, dtype=torch.float32)

    tr_ds = Derm7Set(tr_rows, T_TRAIN, val2id)
    va_ds = Derm7Set(va_rows, T_EVAL, val2id)
    te_ds = Derm7Set(te_rows, T_EVAL, val2id)
    tr_loader = DataLoader(tr_ds, batch_size=BS_E4, shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BS_E4, shuffle=False, num_workers=4)
    te_loader = DataLoader(te_ds, batch_size=BS_E4, shuffle=False, num_workers=4)

    model, n_per_crit = build_e4_model()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_trainable/1e6:.2f} M")

    head_params = []
    for h in model.heads_concept: head_params += list(h.parameters())
    head_params += list(model.head_mel.parameters())
    lora_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and "lora" in n.lower()]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEADS_E4},
        {"params": lora_params, "lr": LR_LORA_E4},
    ], weight_decay=WD_SP)
    n_steps = len(tr_loader) * EPOCHS_E4
    warmup = len(tr_loader)
    def lrl(step):
        if step < warmup: return step / max(1, warmup)
        p = (step - warmup) / max(1, n_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrl)

    best_val_auroc = -1
    best_state = None
    for ep in range(EPOCHS_E4):
        model.train()
        tl, tn = 0.0, 0
        for x, y_c, y_m in tr_loader:
            x = x.to(DEVICE, non_blocking=True); y_c = y_c.to(DEVICE); y_m = y_m.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_c, logits_m, _ = model(x)
                losses_c = [F.cross_entropy(lc.float(), y_c[:, i],
                                              weight=cw_list[i].to(lc.device))
                            for i, lc in enumerate(logits_c)]
                loss_c = sum(losses_c) / len(losses_c)
                loss_m = F.cross_entropy(logits_m.float(), y_m, weight=mw.to(logits_m.device))
                loss = loss_c + loss_m
            loss.backward(); optimizer.step(); scheduler.step()
            tl += loss.item() * x.size(0); tn += x.size(0)
        ym_v, pm_v, probm_v, auroc_v = eval_e4(model, va_loader)
        ym_t, pm_t, probm_t, auroc_t = eval_e4(model, te_loader)
        print(f"  ep{ep+1:2d} loss={tl/tn:.3f} val mel AUROC={auroc_v:.3f} "
              f"test mel AUROC={auroc_t:.3f}")
        if auroc_v > best_val_auroc:
            best_val_auroc = auroc_v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Eval final
    model.load_state_dict(best_state)
    ym_t, pm_t, probm_t, auroc_final = eval_e4(model, te_loader)
    print(f"\n=== E4 FINAL (best val AUROC={best_val_auroc:.3f}) ===")
    print(f"  test mel AUROC: {auroc_final:.4f}")

    # Persistir
    ckpt_path = E4_OUT / "best_model.pth"
    torch.save({
        "state_dict": best_state,
        "n_per_crit": n_per_crit,
        "concept_names": SEVEN_PT,
        "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                        "n_last_blocks": N_LAST_BLOCKS},
        "best_val_auroc": float(best_val_auroc),
        "test_mel_auroc": float(auroc_final),
        "trainable_params_M": round(n_trainable/1e6, 3),
        "seed": 42,
    }, ckpt_path)
    print(f"  ✓ {ckpt_path} ({ckpt_path.stat().st_size/1e6:.1f} MB)")

    # Persistir concept mapping
    mapping_path = E4_OUT / "best_concept_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump({"CRIT_VALS": CRIT_VALS, "melanoma_dxs": list(MELANOMA_DXS)},
                  f, indent=2, ensure_ascii=False)
    print(f"  ✓ {mapping_path}")

    # Persistir metrics
    metrics_path = E4_OUT / "best_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"test_mel_auroc": float(auroc_final),
                   "best_val_auroc": float(best_val_auroc)},
                  f, indent=2, ensure_ascii=False)
    print(f"  ✓ {metrics_path}")

    del model; torch.cuda.empty_cache()
    return auroc_final


# ====================================================================
# Main
# ====================================================================

def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()

    sp_metrics = train_spanderm()
    e4_auroc = train_e4()

    print(f"\n{'='*60}\nResumen total ({(time.time()-t_start)/60:.1f} min)\n{'='*60}")
    print(f"SpanDerm v0 test L2: BAcc={sp_metrics['BAcc']:.4f} AUROC={sp_metrics['AUROC']:.4f}")
    print(f"E4 multitarea test melanoma AUROC={e4_auroc:.4f}")

    # Verificación de carga
    print("\n=== Verificación carga checkpoints ===")
    for path in [
        SPDERM_OUT / "best_seed42.pth",
        E4_OUT / "best_model.pth",
    ]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  ✓ {path.name}: {len(ckpt['state_dict'])} keys")


if __name__ == "__main__":
    main()
