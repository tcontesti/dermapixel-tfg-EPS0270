# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
derm7pt_sae_e3.py · E3 · Cruce conceptos Rosa (Grupo A, 16 dermatoscópicos)
                          ↔ Seven-Point Checklist (Derm7pt)

Estrategia:
  1. Para cada concepto Rosa con match en Derm7pt (directo o por subcategoría):
     - Binarizar y_concept según valores específicos del criterio
     - Calcular AUROC del mejor feature SAE individual
  2. Tabla final: 16 conceptos × {Derm7pt match, N positivos, AUROC SAE}
  3. Reportar cobertura conceptual: cuántos de los 16 quedan cubiertos a AUROC ≥ 0.7

Reutiliza $DERMAPIXEL_ROOT/output/derm7pt_sae_e1/sae_features.npy
"""
from __future__ import annotations
import csv
import json
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DERM7PT_DIR = ROOT / "dermfm_zero/data/PanDerm-2-Eval/multimodal_finetune/multimodal_finetune/derm7pt"
META_CSV = DERM7PT_DIR / "meta" / "meta.csv"
E1_DIR = ROOT / "output" / "derm7pt_sae_e1"
OUT_DIR = ROOT / "output" / "derm7pt_sae_e3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Mapping Rosa Grupo A → Derm7pt (criterio + valores que constituyen positivo)
# null = no mapeo directo (concepto Rosa no representado en Derm7pt)
ROSA_TO_DERM7PT = {
    1:  ("Red dermatoscópica regular",
         "pigment_network", ["typical"], "high"),
    2:  ("Red dermatoscópica atípica",
         "pigment_network", ["atypical"], "high"),
    3:  ("Glóbulos pigmentarios",
         "dots_and_globules", ["regular", "irregular"], "medium"),  # ambos
    4:  ("Puntos pigmentarios",
         "dots_and_globules", ["regular", "irregular"], "low"),  # mismo criterio, ambiguo
    5:  ("Estrías radiales (streaks)",
         "streaks", ["regular", "irregular"], "high"),
    6:  ("Pseudopodios",
         "streaks", ["irregular"], "medium"),  # pseudopodios suelen ser asimétricos
    7:  ("Velo azul-blanquecino",
         "blue_whitish_veil", ["present"], "high"),  # ← MATCH DIRECTO
    8:  ("Estructuras de regresión",
         "regression_structures",
         ["blue areas", "combinations", "white areas"], "high"),  # ← MATCH DIRECTO
    9:  ("Patrón en empedrado (cobblestone)",
         None, [], "no_match"),
    10: ("Patrón paralelo de surcos",
         None, [], "no_match"),
    11: ("Patrón paralelo de crestas",
         None, [], "no_match"),
    12: ("Lagunas vasculares",
         None, [], "no_match"),  # no subcategoría específica en Derm7pt
    13: ("Vasos polimorfos",
         "vascular_structures",
         ["dotted", "comma", "linear irregular", "hairpin", "arborizing"], "low"),
    14: ("Vasos en horquilla",
         "vascular_structures", ["hairpin"], "high"),  # ← MATCH DIRECTO
    15: ("Vasos en corona",
         None, [], "no_match"),  # no en taxonomía Derm7pt
    16: ("Patrón homogéneo desestructurado",
         "pigmentation", ["diffuse regular", "diffuse irregular"], "medium"),
}


def vectorized_auroc(features, y_binary):
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(features.shape[1], np.nan)
    ranks = np.argsort(np.argsort(features, axis=0), axis=0).astype(np.float32) + 1
    sum_ranks_pos = ranks[y_binary == 1].sum(axis=0)
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return U / (n_pos * n_neg)


def main():
    # Cargar features
    features = np.load(E1_DIR / "sae_features.npy")
    print(f"Features SAE: {features.shape}")

    # Cargar meta
    rows = []
    with META_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"Meta: {len(rows)} entradas")
    if len(rows) != len(features):
        print(f"  ! AVISO: features {len(features)} != rows {len(rows)}")

    # Para cada concepto Rosa con mapping, calcular AUROC
    print(f"\n=== AUROC SAE por concepto Rosa (Grupo A) ===")
    results = []
    for cid, (cname, crit, pos_vals, confidence) in ROSA_TO_DERM7PT.items():
        if crit is None:
            results.append({
                "id": cid, "concept_rosa": cname,
                "derm7pt_criterion": "—",
                "positive_values": "—",
                "n_pos": 0, "n_neg": 0,
                "auroc_eff": None,
                "auroc_raw": None,
                "feat_id_top": None,
                "confidence": confidence,
                "covered": False,
            })
            print(f"  {cid:2d}. {cname[:40]:<40} | no_match")
            continue

        y_str = np.array([r[crit] for r in rows])
        y_bin = np.isin(y_str, pos_vals).astype(int)
        n_pos = int(y_bin.sum()); n_neg = int(len(y_bin) - n_pos)

        if n_pos < 5 or n_neg < 5:
            results.append({
                "id": cid, "concept_rosa": cname,
                "derm7pt_criterion": crit, "positive_values": "+".join(pos_vals),
                "n_pos": n_pos, "n_neg": n_neg,
                "auroc_eff": None, "auroc_raw": None, "feat_id_top": None,
                "confidence": confidence, "covered": False,
            })
            print(f"  {cid:2d}. {cname[:40]:<40} | {crit:<22} | desbalanceado n+={n_pos}")
            continue

        aurocs = vectorized_auroc(features, y_bin)
        aurocs_eff = np.maximum(aurocs, 1 - aurocs)
        top = int(np.nanargmax(aurocs_eff))
        auc_eff = float(aurocs_eff[top])
        auc_raw = float(aurocs[top])
        covered = auc_eff >= 0.70

        results.append({
            "id": cid, "concept_rosa": cname,
            "derm7pt_criterion": crit,
            "positive_values": "+".join(pos_vals),
            "n_pos": n_pos, "n_neg": n_neg,
            "auroc_eff": round(auc_eff, 4),
            "auroc_raw": round(auc_raw, 4),
            "feat_id_top": top,
            "confidence": confidence,
            "covered": covered,
        })
        flag = "✓" if covered else "·"
        print(f"  {cid:2d}. {cname[:40]:<40} | {crit:<22} | n+={n_pos:3d} | AUROC eff={auc_eff:.4f} {flag}")

    # CSV
    keys = ["id", "concept_rosa", "derm7pt_criterion", "positive_values",
            "n_pos", "n_neg", "auroc_eff", "auroc_raw", "feat_id_top",
            "confidence", "covered"]
    with (OUT_DIR / "rosa_derm7pt_crossmap.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(results)

    # JSON
    n_covered = sum(1 for r in results if r["covered"])
    n_matched = sum(1 for r in results if r["derm7pt_criterion"] != "—")
    summary = {
        "n_concepts_rosa": len(results),
        "n_matched_derm7pt": n_matched,
        "n_covered_auroc_gte_07": n_covered,
        "n_no_match": sum(1 for r in results if r["derm7pt_criterion"] == "—"),
        "matched_concepts": [r for r in results if r["derm7pt_criterion"] != "—"],
        "unmapped_concepts": [r["concept_rosa"] for r in results if r["derm7pt_criterion"] == "—"],
    }
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Report MD
    md = ["# E3 · Cruce conceptos Rosa (Grupo A) ↔ Seven-Point Checklist (Derm7pt)",
          "",
          f"**Fecha**: 2026-06-02",
          f"**Total conceptos Rosa A**: {len(results)} (16 dermatoscópicos)",
          f"**Con mapping a Derm7pt**: {n_matched}/16",
          f"**Cubiertos por SAE (AUROC ≥ 0,70)**: {n_covered}/{n_matched}",
          "",
          "## Tabla de mapping",
          "",
          "| # | Concepto Rosa | Criterio Derm7pt | Valor(es) positivo(s) | N+ | N− | AUROC eff | Cobertura |",
          "|---:|---|---|---|---:|---:|---:|:---:|"]
    for r in results:
        cov = "✓" if r["covered"] else ("—" if r["derm7pt_criterion"] == "—" else "·")
        auc = f"{r['auroc_eff']:.3f}" if r["auroc_eff"] is not None else "—"
        md.append(f"| {r['id']} | {r['concept_rosa']} | "
                  f"{r['derm7pt_criterion']} | {r['positive_values']} | "
                  f"{r['n_pos']} | {r['n_neg']} | {auc} | {cov} |")
    md.extend([
        "",
        "## Hallazgos",
        "",
        f"- **{n_matched} de 16 conceptos Rosa Grupo A mapean a Derm7pt** (directo o por subcategoría).",
        f"- **{n_covered} conceptos cubiertos por SAE con AUROC ≥ 0,70**.",
        "- Los **3 conceptos con MATCH DIRECTO** son Velo azul-blanquecino, Estructuras de regresión y Vasos en horquilla.",
        "- Conceptos **NO mapeables** (5): cobblestone, paralelo de surcos/crestas, lagunas vasculares, vasos en corona — son específicos del corpus Rosa y NO están en Seven-Point Checklist.",
        "",
        "## Implicación",
        "",
        f"El SAE Large (16.384 features) cubre {n_covered}/{n_matched} conceptos Rosa con match en Derm7pt. Para los {n_matched - n_covered} restantes la cobertura es parcial: el SAE captura el concepto Derm7pt agregado pero no las subcategorías Rosa específicas. Los 5 conceptos sin mapping son material para validar específicamente con anotaciones de la Dra. Taberner sobre las 48 dermatoscopias del Grupo A de DermapixelAI.",
    ])
    (OUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n✓ Resumen: {n_matched}/16 mapeados, {n_covered}/{n_matched} cubiertos AUROC ≥ 0,70")
    print(f"✓ {OUT_DIR / 'results.json'}")
    print(f"✓ {OUT_DIR / 'rosa_derm7pt_crossmap.csv'}")
    print(f"✓ {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
