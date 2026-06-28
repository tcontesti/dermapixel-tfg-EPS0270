# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
derm7pt_sae_e2.py · E2 · Concept Bottleneck Model jerárquico Derm7pt

Pipeline:
  1. Cargar SAE features (cacheados de E1)
  2. Para cada uno de los 7 criterios, entrenar LogReg sobre features SAE
     (entrenamiento sobre train split, evaluación val/test split)
  3. Con las 7 probabilidades de criterio (CBM bottleneck), entrenar
     un meta-clasificador para melanoma binario
  4. Comparar:
     a) Direct: LP sobre features SAE → melanoma binario
     b) CBM-7: features SAE → 7 conceptos → LP → melanoma
     c) Baseline: LP sobre embedding PanDerm 1024d puro → melanoma
  5. Reportar AUROC + BAcc por método

Splits: usar train_indexes.csv (413), valid_indexes.csv (203), test_indexes.csv (395).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, cohen_kappa_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DERM7PT_DIR = ROOT / "dermfm_zero/data/PanDerm-2-Eval/multimodal_finetune/multimodal_finetune/derm7pt"
META_CSV = DERM7PT_DIR / "meta" / "meta.csv"
TRAIN_IDX_CSV = DERM7PT_DIR / "meta" / "train_indexes.csv"
VALID_IDX_CSV = DERM7PT_DIR / "meta" / "valid_indexes.csv"
TEST_IDX_CSV  = DERM7PT_DIR / "meta" / "test_indexes.csv"
E1_DIR = ROOT / "output" / "derm7pt_sae_e1"
OUT_DIR = ROOT / "output" / "derm7pt_sae_e2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEVEN_PT = [
    "pigment_network", "streaks", "pigmentation", "regression_structures",
    "dots_and_globules", "blue_whitish_veil", "vascular_structures",
]

# Diagnoses considerados MELANOMA en derm7pt meta.csv
MELANOMA_DXS = {"melanoma", "melanoma (in situ)", "melanoma (less than 0.76 mm)",
                "melanoma (0.76 to 1.5 mm)", "melanoma (more than 1.5 mm)"}


def load_meta():
    rows = []
    with META_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_idx_csv(p):
    idx = []
    with p.open() as f:
        for r in csv.reader(f):
            if r and r[0] != "indexes":
                try:
                    idx.append(int(r[0]))
                except ValueError:
                    continue
    return np.array(idx)


def metrics_bin(y_true, y_pred, y_prob):
    out = {
        "Acc@1": float(accuracy_score(y_true, y_pred)),
        "BAcc":  float(balanced_accuracy_score(y_true, y_pred)),
        "W-F1":  float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    try:
        out["AUROC"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["AUROC"] = float("nan")
    return out


def main():
    print("=== Cargar meta + splits ===")
    rows = load_meta()
    print(f"  meta.csv: {len(rows)} entradas")
    tr_idx = load_idx_csv(TRAIN_IDX_CSV)
    va_idx = load_idx_csv(VALID_IDX_CSV)
    te_idx = load_idx_csv(TEST_IDX_CSV)
    print(f"  train idx: {len(tr_idx)}, val: {len(va_idx)}, test: {len(te_idx)}")

    # Features SAE de E1
    features = np.load(E1_DIR / "sae_features.npy")
    print(f"  SAE features: {features.shape}")
    # Ojo: features está en el orden de `rows` filtrado por imágenes válidas.
    # Si E1 saltó imágenes, hay que mapear. Por simplicidad asumimos que se
    # procesaron todas las 1011 en orden.
    if len(features) != len(rows):
        print(f"  ! AVISO: features {len(features)} != rows {len(rows)}")

    # Etiquetas de melanoma
    y_mel = np.array([1 if r["diagnosis"] in MELANOMA_DXS else 0 for r in rows])
    print(f"  melanoma global: {y_mel.sum()}/{len(rows)} ({100*y_mel.mean():.1f}%)")

    # Etiquetas binarias de cada criterio (present vs absent)
    crit_labels = {}
    for crit in SEVEN_PT:
        y_str = np.array([r[crit] for r in rows])
        crit_labels[crit] = (y_str != "absent").astype(int)

    # Comparativa (a) Direct LP SAE → melanoma
    print(f"\n=== (a) Direct LP sobre SAE features → melanoma ===")
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
    clf.fit(features[tr_idx], y_mel[tr_idx])
    y_pred = clf.predict(features[te_idx])
    y_prob = clf.predict_proba(features[te_idx])[:, 1]
    direct_metrics = metrics_bin(y_mel[te_idx], y_pred, y_prob)
    print(f"  test: Acc={direct_metrics['Acc@1']:.4f} BAcc={direct_metrics['BAcc']:.4f} AUROC={direct_metrics['AUROC']:.4f}")

    # Comparativa (b) CBM: features → 7 conceptos → melanoma
    print(f"\n=== (b) CBM: SAE → 7 conceptos → melanoma ===")

    # Paso 1: para cada criterio, entrenar LP separado sobre features SAE
    crit_probs_train = np.zeros((len(rows), len(SEVEN_PT)))
    crit_probs_test_aurocs = {}
    print(f"  Entrenando 7 LPs por concepto sobre train ({len(tr_idx)})...")
    for ci, crit in enumerate(SEVEN_PT):
        y_c = crit_labels[crit]
        if y_c[tr_idx].sum() < 5 or (1 - y_c[tr_idx]).sum() < 5:
            print(f"    {crit}: skip (desbalance)")
            continue
        clf_c = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
        clf_c.fit(features[tr_idx], y_c[tr_idx])
        # Inferir prob present para TODOS los splits (train, val, test)
        probs_all = clf_c.predict_proba(features)[:, 1]
        crit_probs_train[:, ci] = probs_all
        # AUROC concepto sobre test
        try:
            auc_c = roc_auc_score(y_c[te_idx], probs_all[te_idx])
        except ValueError:
            auc_c = float("nan")
        crit_probs_test_aurocs[crit] = float(auc_c)
        print(f"    {crit}: AUROC test = {auc_c:.4f}")

    # Paso 2: meta-LP sobre las 7 probs de concepto → melanoma
    print(f"\n  Meta-LP: 7 probs → melanoma")
    meta = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs", random_state=42)
    meta.fit(crit_probs_train[tr_idx], y_mel[tr_idx])
    y_pred_cbm = meta.predict(crit_probs_train[te_idx])
    y_prob_cbm = meta.predict_proba(crit_probs_train[te_idx])[:, 1]
    cbm_metrics = metrics_bin(y_mel[te_idx], y_pred_cbm, y_prob_cbm)
    print(f"  CBM test: Acc={cbm_metrics['Acc@1']:.4f} BAcc={cbm_metrics['BAcc']:.4f} AUROC={cbm_metrics['AUROC']:.4f}")
    print(f"  Pesos meta-LP por concepto: {dict(zip(SEVEN_PT, meta.coef_[0].tolist()))}")

    # Comparativa (c) Baseline: LP sobre embedding PanDerm 1024d puro → melanoma
    # No tenemos embeddings PanDerm puros cacheados; los features SAE son post-SAE.
    # Para comparativa rápida, usar las primeras 1024 dims del SAE espera no es válido.
    # Mejor cargar embeddings PanDerm Large extraídos previamente (si los hay).
    # Buscar embeddings_eval_dermoscopic-melanoma.pt en sae_large/
    print(f"\n=== (c) Baseline: PanDerm Large 1024d puro → melanoma ===")
    import torch
    raw_emb_path = ROOT / "output" / "sae_large" / "embeddings_eval_dermoscopic-melanoma.pt"
    if raw_emb_path.exists():
        raw_embs = torch.load(raw_emb_path, map_location="cpu", weights_only=False)
        print(f"  Encontrado embeddings cacheados: {raw_embs.shape if hasattr(raw_embs, 'shape') else type(raw_embs)}")
        # Estos son embeddings de un eval distinto (sae_base), no de Derm7pt completo
        print(f"  No mapean directamente a meta.csv, skip baseline (c)")
        raw_metrics = {"note": "embeddings PanDerm puros sobre Derm7pt no cacheados; baseline (c) requeriría re-extracción"}
    else:
        raw_metrics = {"note": "embeddings PanDerm raw no encontrados"}

    # Salida
    results = {
        "n_total": len(rows),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "n_melanoma_total": int(y_mel.sum()),
        "n_melanoma_test": int(y_mel[te_idx].sum()),
        "direct_lp_sae": direct_metrics,
        "cbm_7_concepts": cbm_metrics,
        "baseline_raw_panderm": raw_metrics,
        "concept_aurocs_test": crit_probs_test_aurocs,
        "meta_lp_coefs": {c: float(w) for c, w in zip(SEVEN_PT, meta.coef_[0])},
    }
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Report MD
    md = ["# E2 · Concept Bottleneck Model jerárquico Derm7pt",
          "",
          f"**Fecha**: 2026-06-02",
          f"**Total**: {len(rows)} | Train: {len(tr_idx)} | Val: {len(va_idx)} | Test: {len(te_idx)}",
          f"**Melanoma global**: {y_mel.sum()}/{len(rows)} ({100*y_mel.mean():.1f}%)",
          f"**Melanoma test**: {int(y_mel[te_idx].sum())}/{len(te_idx)} ({100*y_mel[te_idx].mean():.1f}%)",
          "",
          "## Comparativa de arquitecturas",
          "",
          "| Método | Acc@1 | BAcc | AUROC | W-F1 | Kappa |",
          "|---|---:|---:|---:|---:|---:|",
          f"| Direct LP (SAE 16k → mel) | {direct_metrics['Acc@1']:.4f} | {direct_metrics['BAcc']:.4f} | "
          f"**{direct_metrics['AUROC']:.4f}** | {direct_metrics['W-F1']:.4f} | {direct_metrics['Kappa']:.4f} |",
          f"| **CBM** (SAE → 7 cpts → mel) | {cbm_metrics['Acc@1']:.4f} | {cbm_metrics['BAcc']:.4f} | "
          f"**{cbm_metrics['AUROC']:.4f}** | {cbm_metrics['W-F1']:.4f} | {cbm_metrics['Kappa']:.4f} |",
          "",
          "## AUROC por concepto (cabeza intermedia)",
          "",
          "| Concepto | AUROC test |",
          "|---|---:|",
          ]
    for crit in SEVEN_PT:
        auc = crit_probs_test_aurocs.get(crit, "—")
        if isinstance(auc, float):
            md.append(f"| {crit} | {auc:.4f} |")
        else:
            md.append(f"| {crit} | {auc} |")
    md.extend([
        "",
        "## Pesos del meta-clasificador (importancia relativa)",
        "",
        "| Concepto | Peso meta-LP (mel) |",
        "|---|---:|",
    ])
    for crit, w in zip(SEVEN_PT, meta.coef_[0]):
        md.append(f"| {crit} | {w:+.4f} |")

    (OUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n✓ {OUT_DIR / 'results.json'}")
    print(f"✓ {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
