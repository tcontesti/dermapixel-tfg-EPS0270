# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
derm7pt_sae_e1.py · E1 · Validación SAE sobre los 7 criterios dermatoscópicos

Para cada una de las 7 anotaciones del Seven-Point Checklist en Derm7pt:
  - Binarizar a presente (cualquier valor != "absent") vs ausente
  - Para cada feature SAE (16.384), calcular AUROC sobre activaciones
  - Reportar top-10 features con AUROC más alto por criterio

Encoder: PanDerm Large (1024d) → SAE Large (16384 sparse features)
Imágenes: 1011 lesiones del split test+train+val (meta.csv completo)

Salida: $DERMAPIXEL_ROOT/output/derm7pt_sae_e1/{report.md, top_features.csv, all_aurocs.npy}
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
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DERM7PT_DIR = ROOT / "dermfm_zero/data/PanDerm-2-Eval/multimodal_finetune/multimodal_finetune/derm7pt"
META_CSV = DERM7PT_DIR / "meta" / "meta.csv"
IMG_DIR = DERM7PT_DIR / "images"
WEIGHTS = ROOT / "weights" / "weights"
SAE_PATH = ROOT / "output" / "sae_large" / "sae_large_best.pth"
OUT_DIR = ROOT / "output" / "derm7pt_sae_e1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "classification"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEVEN_PT = [
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
]


# -----------------------------------------------------------------------------
# Encoder + SAE
# -----------------------------------------------------------------------------

def build_panderm_large():
    from models.modeling_finetune import panderm_large_patch16_224
    sd = torch.load(WEIGHTS / "panderm_large.pth", map_location="cpu", weights_only=False)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    model = panderm_large_patch16_224()
    model.load_state_dict(sd, strict=False)
    model.head = nn.Identity()
    return model.to(DEVICE).eval()


class SparseAutoencoder(nn.Module):
    def __init__(self, n_input=1024, n_learned=16384):
        super().__init__()
        self.pre_bias = nn.Parameter(torch.zeros(n_input))
        self.encoder = nn.Linear(n_input, n_learned)
        self.decoder = nn.Linear(n_learned, n_input, bias=False)
        self.post_bias = nn.Parameter(torch.zeros(n_input))

    def forward(self, x):
        x_centered = x - self.pre_bias
        z = torch.relu(self.encoder(x_centered))
        x_hat = self.decoder(z) + self.post_bias
        return x_hat, z

    def encode(self, x):
        x_centered = x - self.pre_bias
        return torch.relu(self.encoder(x_centered))


def load_sae():
    ckpt = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sae = SparseAutoencoder(n_input=cfg["n_input"], n_learned=cfg["n_learned"])
    sae.load_state_dict(ckpt["model_state_dict"])
    return sae.to(DEVICE).eval(), cfg


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

T_EVAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def load_meta():
    rows = []
    with META_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


@torch.no_grad()
def extract_sae_features(encoder, sae, rows, batch_size=16):
    feats_all = []
    valid_idx = []
    t0 = time.time()
    batch_tensors, batch_idx = [], []
    for i, r in enumerate(rows):
        img_path = IMG_DIR / r["derm"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            batch_tensors.append(T_EVAL(img))
            batch_idx.append(i)
        except Exception as e:
            print(f"  ! error {img_path.name}: {e}")
            continue

        if len(batch_tensors) >= batch_size or i == len(rows) - 1:
            if not batch_tensors:
                continue
            x = torch.stack(batch_tensors).to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = encoder(x).float()
                z = sae.encode(emb)
            feats_all.append(z.float().cpu().numpy())
            valid_idx.extend(batch_idx)
            batch_tensors, batch_idx = [], []
            if (len(valid_idx) % 200) < batch_size:
                print(f"  procesadas {len(valid_idx)}/{len(rows)} en {time.time()-t0:.1f}s")

    feats_all = np.vstack(feats_all) if feats_all else np.empty((0, 16384))
    return feats_all, valid_idx


def vectorized_auroc(features, y_binary):
    """AUROC para cada columna de features (vectorizado).

    Implementa Mann-Whitney U directamente con argsort. Mucho más rápido que
    iterar sklearn.roc_auc_score sobre 16K features.
    """
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(features.shape[1], np.nan)

    # Ranks por columna
    ranks = np.argsort(np.argsort(features, axis=0), axis=0).astype(np.float32) + 1

    # Suma de ranks de positivos por columna
    sum_ranks_pos = ranks[y_binary == 1].sum(axis=0)

    # U = sum_ranks - n_pos*(n_pos+1)/2
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2

    # AUROC = U / (n_pos * n_neg)
    auroc = U / (n_pos * n_neg)
    return auroc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    rows = load_meta()
    print(f"Total entradas meta.csv: {len(rows)}")

    print("\n=== Cargando encoder + SAE ===")
    encoder = build_panderm_large()
    sae, sae_cfg = load_sae()
    print(f"  PanDerm Large + SAE ({sae_cfg['n_learned']} features)")

    # Verificar valores únicos por criterio
    print("\n=== Distribución de valores por criterio ===")
    for crit in SEVEN_PT:
        from collections import Counter
        counts = Counter(r[crit] for r in rows)
        present_pct = 100 * sum(v for k, v in counts.items() if k != "absent") / len(rows)
        print(f"  {crit}: {dict(counts)} → present {present_pct:.1f}%")

    # Extraer features SAE
    print(f"\n=== Extrayendo embeddings SAE para {len(rows)} imágenes derm ===")
    features, valid_idx = extract_sae_features(encoder, sae, rows)
    print(f"  Features shape: {features.shape}, valid {len(valid_idx)}/{len(rows)}")
    np.save(OUT_DIR / "sae_features.npy", features)
    valid_rows = [rows[i] for i in valid_idx]

    # Activación media por feature (¿cuántas activas?)
    activation_freq = (features > 0).mean(axis=0)
    print(f"  Features activas en >5% de imgs: {(activation_freq > 0.05).sum()}")
    print(f"  Features activas en >50% imgs:   {(activation_freq > 0.5).sum()}")
    print(f"  Features 'dead' (0 activación):   {(activation_freq == 0).sum()}")

    # AUROC por feature × criterio
    print(f"\n=== AUROC vectorizado por feature × criterio ===")
    results = {}
    top_features_all = []

    for crit in SEVEN_PT:
        y_str = np.array([r[crit] for r in valid_rows])
        y_bin = (y_str != "absent").astype(int)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)
        print(f"\n  --- {crit} ---  present={n_pos}, absent={n_neg}")
        if n_pos < 5 or n_neg < 5:
            print(f"    ! demasiado desbalanceado, skip")
            continue

        t0 = time.time()
        aurocs = vectorized_auroc(features, y_bin)
        # roc_auc considera el lado: si features bajas son positivas, AUROC < 0.5
        # tomamos max(auroc, 1-auroc) para captar discriminación en ambas direcciones
        aurocs_eff = np.maximum(aurocs, 1 - aurocs)
        print(f"    AUROC max: {np.nanmax(aurocs):.4f} | min: {np.nanmin(aurocs):.4f}")
        print(f"    AUROC efectivo top-1: {np.nanmax(aurocs_eff):.4f}  ({time.time()-t0:.1f}s)")

        # Top-20 features
        ranked = np.argsort(-aurocs_eff)
        top20 = ranked[:20]
        results[crit] = {
            "n_present": n_pos, "n_absent": n_neg,
            "auroc_top1": float(aurocs_eff[top20[0]]),
            "auroc_top5_mean": float(aurocs_eff[ranked[:5]].mean()),
            "auroc_top10_mean": float(aurocs_eff[ranked[:10]].mean()),
            "top20_features": [
                {"feat_id": int(f), "auroc_eff": float(aurocs_eff[f]),
                 "auroc_raw": float(aurocs[f]),
                 "direction": "high" if aurocs[f] >= 0.5 else "low",
                 "activation_freq": float(activation_freq[f])}
                for f in top20
            ],
        }
        # Para tabla CSV
        for rank, f in enumerate(top20[:10]):
            top_features_all.append({
                "criterio": crit,
                "rank": rank+1,
                "feat_id": int(f),
                "auroc_eff": round(float(aurocs_eff[f]), 4),
                "direction": "high" if aurocs[f] >= 0.5 else "low",
                "activation_freq": round(float(activation_freq[f]), 4),
            })

    # Save
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump({
            "n_total": len(rows),
            "n_valid": len(valid_idx),
            "sae_config": sae_cfg,
            "criteria": SEVEN_PT,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    # Top features CSV
    with (OUT_DIR / "top_features.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["criterio", "rank", "feat_id", "auroc_eff",
                                          "direction", "activation_freq"])
        w.writeheader()
        w.writerows(top_features_all)

    # Report MD
    md = ["# E1 · SAE features vs Seven-Point Checklist (Derm7pt)",
          "",
          f"**Fecha**: 2026-06-02",
          f"**Encoder**: PanDerm Large (1024d) + SAE Large ({sae_cfg['n_learned']} features)",
          f"**Imágenes**: {len(valid_idx)}/{len(rows)} dermoscopic lesions",
          "",
          "## Resumen por criterio (AUROC efectivo = max(auroc, 1-auroc))",
          "",
          "| Criterio | N present | N absent | AUROC top-1 | AUROC top-5 mean | AUROC top-10 mean |",
          "|---|---:|---:|---:|---:|---:|"]
    for crit in SEVEN_PT:
        if crit not in results:
            md.append(f"| {crit} | — | — | (skip, desbalance) | — | — |")
            continue
        r = results[crit]
        md.append(f"| {crit} | {r['n_present']} | {r['n_absent']} | "
                  f"**{r['auroc_top1']:.4f}** | {r['auroc_top5_mean']:.4f} | "
                  f"{r['auroc_top10_mean']:.4f} |")

    md.extend([
        "",
        "## Top-10 features por criterio",
        "",
    ])
    for crit in SEVEN_PT:
        if crit not in results: continue
        r = results[crit]
        md.append(f"### {crit} (present={r['n_present']}, absent={r['n_absent']})")
        md.append("")
        md.append("| Rank | Feat ID | AUROC eff | Direction | Act freq |")
        md.append("|---:|---:|---:|---|---:|")
        for rank, feat_info in enumerate(r["top20_features"][:10]):
            md.append(f"| {rank+1} | {feat_info['feat_id']} | "
                      f"{feat_info['auroc_eff']:.4f} | {feat_info['direction']} | "
                      f"{feat_info['activation_freq']:.4f} |")
        md.append("")

    (OUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n✓ Resultados:")
    print(f"  - {OUT_DIR / 'results.json'}")
    print(f"  - {OUT_DIR / 'top_features.csv'}")
    print(f"  - {OUT_DIR / 'report.md'}")
    print(f"  - {OUT_DIR / 'sae_features.npy'} ({features.nbytes / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
