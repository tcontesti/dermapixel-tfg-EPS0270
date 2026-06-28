# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_eda.py · EDA exhaustivo DermapixelAI 1.0

Análisis:
  1. Volumen y filtrado (1089 total, 1062 filtered)
  2. Distribución de tipos de imagen
  3. Distribución por nivel L1/L2/L3 + cola larga
  4. Análisis de splits (train/val/test) — balance por L1/L2
  5. Casos: imgs/caso, casos/L1, distribución case_text_length
  6. Calidad etiqueta: label_source + rosa_verified + diagnosis_source
  7. Distribución temporal (año)
  8. Ontología vs corpus: cobertura
  9. Comparación con datasets de referencia (HAM/BCN/ISIC/Derm7pt)

Salida:
  $DERMAPIXEL_ROOT/output/dermapixel_v1_eda/
    ├── eda_report.md        — reporte ejecutivo
    ├── tables/              — CSVs por análisis
    ├── figures/             — PNGs
    └── eda_results.json     — todas las cifras programáticamente

Uso (Spark):
    python3 dermapixel_eda.py
"""
from __future__ import annotations
import csv
import json
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
OUT_DIR  = ROOT / "output" / "dermapixel_v1_eda"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_table(rows, name):
    if not rows: return
    p = OUT_DIR / "tables" / f"{name}.csv"
    keys = list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


# -----------------------------------------------------------------------------
# Análisis
# -----------------------------------------------------------------------------

def main():
    # Cargar todo
    rows_all = load_csv(DATASET / "dataset.csv")
    rows_filt = load_csv(DATASET / "dataset_filtered.csv")
    cases = load_csv(DATASET / "metadata" / "cases.csv")
    ontology = load_csv(DATASET / "metadata" / "ontology.csv")
    excluded = load_csv(DATASET / "metadata" / "excluded_images.csv")

    # Consolidar L2 ortográfico
    for r in rows_all + rows_filt:
        r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])

    R = {}  # results dict

    # === 1. Volumen y filtrado ===
    R["volumen"] = {
        "n_total": len(rows_all),
        "n_filtered": len(rows_filt),
        "n_excluded": len(excluded),
        "n_cases": len(cases),
    }
    print(f"=== 1. Volumen ===")
    print(f"  Dataset total:    {len(rows_all)} imgs")
    print(f"  Dataset filtered: {len(rows_filt)} imgs (exclusión {len(rows_all)-len(rows_filt)})")
    print(f"  Excluded total:   {len(excluded)} imgs")
    print(f"  Casos clínicos:   {len(cases)}")

    # Razones de exclusión
    excl_reasons = Counter(r.get("exclusion_reason", "?") for r in excluded)
    print(f"  Razones exclusión: {dict(excl_reasons)}")
    R["volumen"]["exclusion_reasons"] = dict(excl_reasons)

    # === 2. Tipos de imagen ===
    types_all = Counter(r["image_type"] for r in rows_all)
    types_filt = Counter(r["image_type"] for r in rows_filt)
    R["image_types"] = {
        "all": dict(types_all),
        "filtered": dict(types_filt),
    }
    print(f"\n=== 2. Tipos de imagen ===")
    print(f"  Dataset total:    {dict(types_all)}")
    print(f"  Filtered:         {dict(types_filt)}")

    # === 3. Distribución L1 / L2 / L3 ===
    print(f"\n=== 3. Distribución ontológica (filtered) ===")
    R["distribuciones"] = {}
    for lvl in ("ontology_l1", "ontology_l2", "ontology_l3"):
        counts = Counter(r[lvl] for r in rows_filt if r[lvl])
        # Sobre vocabulario completo
        vocab_size = {
            "ontology_l1": 4,
            "ontology_l2": 43,
            "ontology_l3": 367,
        }[lvl]
        present = len(counts)
        coverage = present / vocab_size * 100
        R["distribuciones"][lvl] = {
            "vocab_size": vocab_size,
            "n_distinct_present": present,
            "coverage_pct": round(coverage, 2),
            "counts": dict(counts),
        }
        print(f"  {lvl}: {present}/{vocab_size} clases presentes ({coverage:.1f}%)")
        if lvl == "ontology_l1":
            for k, v in counts.most_common():
                print(f"    {k}: {v} ({100*v/len(rows_filt):.1f}%)")

    # === 4. Cola larga L3 ===
    l3_counts = R["distribuciones"]["ontology_l3"]["counts"]
    distrib_brackets = {
        "1 img": 0, "2-5 imgs": 0, "6-10 imgs": 0,
        "11-20 imgs": 0, "21+ imgs": 0,
    }
    for c, n in l3_counts.items():
        if n == 1: distrib_brackets["1 img"] += 1
        elif n <= 5: distrib_brackets["2-5 imgs"] += 1
        elif n <= 10: distrib_brackets["6-10 imgs"] += 1
        elif n <= 20: distrib_brackets["11-20 imgs"] += 1
        else: distrib_brackets["21+ imgs"] += 1
    R["cola_larga_l3"] = distrib_brackets
    print(f"\n=== 4. Cola larga L3 ===")
    for k, v in distrib_brackets.items():
        print(f"  {k}: {v} clases L3 ({100*v/len(l3_counts):.1f}%)")

    # Top-10 L3 más frecuentes
    top10 = l3_counts and sorted(l3_counts.items(), key=lambda x: -x[1])[:10]
    R["top10_l3"] = top10
    print(f"\n  Top-10 L3: {top10}")

    # === 5. Splits ===
    print(f"\n=== 5. Splits ===")
    splits = Counter(r["split"] for r in rows_filt)
    print(f"  Filtered splits: {dict(splits)}")
    R["splits"] = dict(splits)

    # Balance L1 por split
    R["balance_l1_por_split"] = {}
    print(f"\n  Balance L1 por split:")
    for sp in ("train", "val", "test"):
        sp_rows = [r for r in rows_filt if r["split"] == sp]
        l1_dist = Counter(r["ontology_l1"] for r in sp_rows)
        R["balance_l1_por_split"][sp] = dict(l1_dist)
        print(f"    {sp} ({len(sp_rows)}): {dict(l1_dist)}")

    # L2 sin presencia en train
    train_l2 = set(r["ontology_l2"] for r in rows_filt if r["split"] == "train")
    test_l2 = set(r["ontology_l2"] for r in rows_filt if r["split"] == "test")
    val_l2 = set(r["ontology_l2"] for r in rows_filt if r["split"] == "val")
    R["l2_split_overlap"] = {
        "n_train": len(train_l2),
        "n_val": len(val_l2),
        "n_test": len(test_l2),
        "test_not_in_train": sorted(list(test_l2 - train_l2)),
        "val_not_in_train": sorted(list(val_l2 - train_l2)),
        "train_not_in_test": sorted(list(train_l2 - test_l2)),
    }

    # === 6. Casos clínicos ===
    print(f"\n=== 6. Casos clínicos ===")
    case_imgs = defaultdict(int)
    for r in rows_filt:
        case_imgs[r["case_id"]] += 1
    img_per_case = list(case_imgs.values())
    R["casos"] = {
        "n_casos_total_en_cases_csv": len(cases),
        "n_casos_en_filtered": len(case_imgs),
        "imgs_por_caso_media": round(float(np.mean(img_per_case)), 2),
        "imgs_por_caso_mediana": int(np.median(img_per_case)),
        "imgs_por_caso_max": max(img_per_case),
        "casos_con_1_img": sum(1 for v in img_per_case if v == 1),
        "casos_con_2_imgs": sum(1 for v in img_per_case if v == 2),
        "casos_con_3plus_imgs": sum(1 for v in img_per_case if v >= 3),
    }
    print(f"  Casos en filtered: {len(case_imgs)}")
    print(f"  Imgs/caso: media {R['casos']['imgs_por_caso_media']:.2f}, "
          f"mediana {R['casos']['imgs_por_caso_mediana']}, max {R['casos']['imgs_por_caso_max']}")
    print(f"  Casos 1 img: {R['casos']['casos_con_1_img']}, "
          f"2 imgs: {R['casos']['casos_con_2_imgs']}, "
          f"3+: {R['casos']['casos_con_3plus_imgs']}")

    # case_text disponibilidad
    case_text_len = [int(c.get("case_text_length") or 0) for c in cases]
    R["case_text"] = {
        "mean": round(float(np.mean(case_text_len)), 1),
        "median": int(np.median(case_text_len)),
        "max": max(case_text_len),
        "n_empty": sum(1 for v in case_text_len if v == 0),
    }
    print(f"  case_text length: media {R['case_text']['mean']}, "
          f"mediana {R['case_text']['median']}, max {R['case_text']['max']}, "
          f"empty {R['case_text']['n_empty']}")

    # === 7. Calidad etiqueta ===
    print(f"\n=== 7. Calidad etiqueta ===")
    label_source = Counter(r["label_source"] for r in rows_all)
    rosa_v = Counter(r["rosa_verified"] for r in rows_all)
    diag_source = Counter(r["diagnosis_source"] for r in rows_all)
    R["calidad_etiqueta"] = {
        "label_source": dict(label_source),
        "rosa_verified": dict(rosa_v),
        "diagnosis_source": dict(diag_source),
        "pct_ontology": round(100 * label_source.get("ontology", 0) / len(rows_all), 2),
        "pct_rosa_True": round(100 * rosa_v.get("True", 0) / len(rows_all), 2),
        "pct_expert_v3": round(100 * diag_source.get("expert_v3", 0) / len(rows_all), 2),
    }
    print(f"  label_source: {dict(label_source)} | ontology={R['calidad_etiqueta']['pct_ontology']}%")
    print(f"  rosa_verified: {dict(rosa_v)} | True={R['calidad_etiqueta']['pct_rosa_True']}%")
    print(f"  diagnosis_source: {dict(diag_source)} | expert_v3={R['calidad_etiqueta']['pct_expert_v3']}%")

    # === 8. Distribución temporal ===
    print(f"\n=== 8. Temporal ===")
    years = [int(r.get("year") or 0) for r in rows_filt if r.get("year") and r["year"] != ""]
    year_counts = Counter(years)
    R["temporal"] = {
        "year_min": min(year_counts.keys()) if year_counts else None,
        "year_max": max(year_counts.keys()) if year_counts else None,
        "year_distrib": dict(sorted(year_counts.items())),
        "year_peak": year_counts.most_common(1)[0] if year_counts else None,
    }
    print(f"  Rango: {R['temporal']['year_min']}-{R['temporal']['year_max']}")
    print(f"  Pico: {R['temporal']['year_peak']}")

    # === 9. Ontología completa vs corpus ===
    print(f"\n=== 9. Ontología vs corpus ===")
    vocab_l3 = set(r["nivel_3_diagnostico"] for r in ontology if r["nivel_3_diagnostico"])
    corpus_l3 = set(r["ontology_l3"] for r in rows_filt if r["ontology_l3"])
    R["ontologia_cobertura"] = {
        "vocab_l3_size": len(vocab_l3),
        "corpus_l3_size": len(corpus_l3),
        "coverage_pct": round(100 * len(corpus_l3 & vocab_l3) / max(1, len(vocab_l3)), 2),
        "l3_in_vocab_not_corpus": len(vocab_l3 - corpus_l3),
        "l3_in_corpus_not_vocab": len(corpus_l3 - vocab_l3),
    }
    print(f"  Vocabulario L3: {R['ontologia_cobertura']['vocab_l3_size']} entradas")
    print(f"  Corpus L3:      {R['ontologia_cobertura']['corpus_l3_size']} entradas")
    print(f"  Cobertura:      {R['ontologia_cobertura']['coverage_pct']}%")

    # === 10. Imagen tamaños ===
    # Demasiado costoso abrir 1062 imgs aquí, saltar y notar como pendiente

    # ========================================================================
    # FIGURAS
    # ========================================================================
    print(f"\n=== Generando figuras ===")

    # Fig 1: Histograma L1
    l1_data = Counter(r["ontology_l1"] for r in rows_filt if r["ontology_l1"])
    fig, ax = plt.subplots(figsize=(7, 4))
    items = sorted(l1_data.items(), key=lambda x: -x[1])
    bars = ax.barh([k for k, _ in items], [v for _, v in items], color="#3b6ea5")
    ax.set_xlabel("Imágenes")
    ax.set_title("Distribución L1 etiológico (n=1062)")
    for b, (_, v) in zip(bars, items):
        ax.text(b.get_width() + 5, b.get_y() + b.get_height()/2,
                f"{v} ({100*v/len(rows_filt):.1f}%)", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "fig01_l1_distribucion.png", dpi=140)
    plt.close()

    # Fig 2: Top-20 L2
    l2_data = Counter(r["ontology_l2"] for r in rows_filt if r["ontology_l2"])
    top20_l2 = sorted(l2_data.items(), key=lambda x: -x[1])[:20]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh([k for k, _ in reversed(top20_l2)], [v for _, v in reversed(top20_l2)],
            color="#5da566")
    ax.set_xlabel("Imágenes")
    ax.set_title(f"Top-20 L2 (de {len(l2_data)} efectivas)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "fig02_l2_top20.png", dpi=140)
    plt.close()

    # Fig 3: Cola larga L3
    fig, ax = plt.subplots(figsize=(6.5, 4))
    keys = list(distrib_brackets.keys())
    vals = list(distrib_brackets.values())
    bars = ax.bar(keys, vals, color="#c75450")
    ax.set_ylabel("Nº clases L3")
    ax.set_title(f"Cola larga L3 (corpus: {len(l3_counts)} clases efectivas)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "fig03_l3_cola_larga.png", dpi=140)
    plt.close()

    # Fig 4: Histograma L3 ranked
    l3_counts_sorted = sorted(l3_counts.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(l3_counts_sorted, color="#c75450", linewidth=1)
    ax.fill_between(range(len(l3_counts_sorted)), l3_counts_sorted, alpha=0.2, color="#c75450")
    ax.set_yscale("log")
    ax.set_xlabel(f"Rank de clase L3 (1-{len(l3_counts_sorted)})")
    ax.set_ylabel("Imágenes (log)")
    ax.set_title("Distribución L3 ordenada (cola larga visible)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "fig04_l3_rank.png", dpi=140)
    plt.close()

    # Fig 5: Splits por L1 stacked
    splits_l1_matrix = {}
    l1_order = sorted(set(r["ontology_l1"] for r in rows_filt if r["ontology_l1"]))
    for sp in ("train", "val", "test"):
        splits_l1_matrix[sp] = [
            sum(1 for r in rows_filt if r["split"] == sp and r["ontology_l1"] == l)
            for l in l1_order
        ]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(l1_order))
    width = 0.27
    colors = {"train": "#3b6ea5", "val": "#e8a44c", "test": "#c75450"}
    for i, sp in enumerate(("train", "val", "test")):
        ax.bar(x + (i-1)*width, splits_l1_matrix[sp], width,
               label=f"{sp} (N={sum(splits_l1_matrix[sp])})", color=colors[sp])
    ax.set_xticks(x); ax.set_xticklabels(l1_order, rotation=18, ha="right")
    ax.set_ylabel("Imágenes")
    ax.legend()
    ax.set_title("Splits por L1 etiológico")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "fig05_splits_l1.png", dpi=140)
    plt.close()

    # Fig 6: imgs/caso histograma
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(img_per_case, bins=range(1, max(img_per_case)+2), color="#7a5dc7",
            edgecolor="white")
    ax.set_xlabel("Imágenes por caso")
    ax.set_ylabel("Nº de casos")
    ax.set_title(f"Imgs/caso ({len(case_imgs)} casos, media {np.mean(img_per_case):.2f})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "fig06_imgs_por_caso.png", dpi=140)
    plt.close()

    # Fig 7: Temporal
    if year_counts:
        yr_sorted = sorted(year_counts.items())
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar([y for y, _ in yr_sorted], [c for _, c in yr_sorted], color="#4a8d8e")
        ax.set_xlabel("Año")
        ax.set_ylabel("Imágenes")
        ax.set_title("Distribución temporal")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "figures" / "fig07_temporal.png", dpi=140)
        plt.close()

    # Fig 8: Calidad etiqueta (pie chart triple)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (title, counter) in zip(axes, (
            ("label_source", label_source),
            ("rosa_verified", rosa_v),
            ("diagnosis_source", diag_source))):
        labels = list(counter.keys())
        sizes = list(counter.values())
        colors_p = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        ax.pie(sizes, labels=labels, autopct="%.1f%%", colors=colors_p,
               textprops={"fontsize": 8})
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "fig08_calidad_etiqueta.png", dpi=140)
    plt.close()

    print(f"  Generadas 8 figuras en {OUT_DIR / 'figures'}")

    # ========================================================================
    # Tablas CSV
    # ========================================================================
    print(f"\n=== Generando tablas CSV ===")

    save_table([{"l1": k, "n_imgs": v,
                 "pct": round(100*v/len(rows_filt), 2)}
                for k, v in l1_data.most_common()], "tab_l1_distribucion")

    save_table([{"l2": k, "n_imgs": v,
                 "pct": round(100*v/len(rows_filt), 2)}
                for k, v in l2_data.most_common()], "tab_l2_distribucion")

    save_table([{"l3": k, "n_imgs": v}
                for k, v in sorted(l3_counts.items(), key=lambda x: -x[1])],
               "tab_l3_distribucion")

    save_table([{"rango": k, "n_clases_l3": v,
                 "pct": round(100*v/len(l3_counts), 2)}
                for k, v in distrib_brackets.items()], "tab_l3_cola_larga")

    save_table([
        {"split": sp,
         "n_imgs": splits[sp],
         "n_casos": len(set(r["case_id"] for r in rows_filt if r["split"] == sp)),
         "n_l1": len(set(r["ontology_l1"] for r in rows_filt if r["split"] == sp)),
         "n_l2": len(set(r["ontology_l2"] for r in rows_filt if r["split"] == sp)),
         "n_l3": len(set(r["ontology_l3"] for r in rows_filt if r["split"] == sp)),
        } for sp in ("train", "val", "test")
    ], "tab_splits_resumen")

    print(f"  Tablas en {OUT_DIR / 'tables'}")

    # ========================================================================
    # JSON final + reporte MD
    # ========================================================================
    with (OUT_DIR / "eda_results.json").open("w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ JSON: {OUT_DIR / 'eda_results.json'}")

    # Reporte MD
    md_lines = [
        "# EDA exhaustivo DermapixelAI 1.0",
        "",
        f"**Fecha**: 2026-06-02",
        f"**Fuente**: `$DERMAPIXEL_ROOT/datasets/dermapixel_v1/`",
        "",
        "## 1. Volumen y filtrado",
        f"- Total imágenes: **{R['volumen']['n_total']:,}**",
        f"- Imágenes filtered (clinical+dermoscopy, label_source=ontology, L1≠∅): **{R['volumen']['n_filtered']:,}**",
        f"- Imágenes excluidas: {R['volumen']['n_excluded']}",
        f"- Casos clínicos: **{R['volumen']['n_cases']}**",
        "",
        "## 2. Tipos de imagen",
        "| Tipo | Total | Filtered |",
        "|---|---:|---:|",
    ]
    for t in sorted(set(R["image_types"]["all"]) | set(R["image_types"]["filtered"])):
        a = R["image_types"]["all"].get(t, 0)
        f_ = R["image_types"]["filtered"].get(t, 0)
        md_lines.append(f"| {t} | {a:,} | {f_:,} |")
    md_lines.extend([
        "",
        "## 3. Distribución ontológica",
        "| Nivel | Vocab teórico | Presente en corpus | Cobertura |",
        "|---|---:|---:|---:|",
    ])
    for lvl in ("ontology_l1", "ontology_l2", "ontology_l3"):
        d = R["distribuciones"][lvl]
        md_lines.append(f"| {lvl} | {d['vocab_size']} | {d['n_distinct_present']} | {d['coverage_pct']}% |")
    md_lines.extend([
        "",
        "## 4. Cola larga L3",
        f"De las {R['distribuciones']['ontology_l3']['n_distinct_present']} clases L3 efectivas:",
        "",
        "| Rango | Nº clases L3 |",
        "|---|---:|",
    ])
    for k, v in R["cola_larga_l3"].items():
        md_lines.append(f"| {k} | {v} |")
    md_lines.extend([
        "",
        "**Top-10 diagnósticos L3 más frecuentes:**",
        "",
    ])
    for k, v in R.get("top10_l3", []):
        md_lines.append(f"- {k}: {v} imgs")

    md_lines.extend([
        "",
        "## 5. Splits",
        "| Split | Imágenes | Casos | L1 | L2 | L3 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for sp in ("train", "val", "test"):
        n_imgs = R["splits"].get(sp, 0)
        n_casos = len(set(r["case_id"] for r in rows_filt if r["split"] == sp))
        n_l1 = len(set(r["ontology_l1"] for r in rows_filt if r["split"] == sp))
        n_l2 = len(set(r["ontology_l2"] for r in rows_filt if r["split"] == sp))
        n_l3 = len(set(r["ontology_l3"] for r in rows_filt if r["split"] == sp))
        md_lines.append(f"| {sp} | {n_imgs} | {n_casos} | {n_l1} | {n_l2} | {n_l3} |")

    md_lines.extend([
        "",
        f"**L2 no vista en test**: {len(R['l2_split_overlap']['train_not_in_test'])} clases L2 están en train pero no aparecen en test.",
        f"**L2 no vista en train (test)**: {len(R['l2_split_overlap']['test_not_in_train'])} clases L2 en test sin presencia en train (filtrado obligado).",
        "",
        "## 6. Casos clínicos",
        f"- Imágenes/caso: media **{R['casos']['imgs_por_caso_media']}**, mediana {R['casos']['imgs_por_caso_mediana']}, max {R['casos']['imgs_por_caso_max']}",
        f"- Casos 1 img: {R['casos']['casos_con_1_img']}, 2 imgs: {R['casos']['casos_con_2_imgs']}, 3+ imgs: {R['casos']['casos_con_3plus_imgs']}",
        f"- `case_text` length: media {R['case_text']['mean']} chars, max {R['case_text']['max']}, vacíos: {R['case_text']['n_empty']}",
        "",
        "## 7. Calidad de etiqueta",
        f"- `label_source=ontology`: **{R['calidad_etiqueta']['pct_ontology']}%** ({R['calidad_etiqueta']['label_source'].get('ontology', 0)} imgs)",
        f"- `rosa_verified=True`: **{R['calidad_etiqueta']['pct_rosa_True']}%** ({R['calidad_etiqueta']['rosa_verified'].get('True', 0)} imgs)",
        f"- `diagnosis_source=expert_v3`: **{R['calidad_etiqueta']['pct_expert_v3']}%** ({R['calidad_etiqueta']['diagnosis_source'].get('expert_v3', 0)} imgs)",
        "",
        "## 8. Distribución temporal",
        f"- Rango: **{R['temporal']['year_min']}–{R['temporal']['year_max']}**",
        f"- Año pico: {R['temporal']['year_peak'][0]} ({R['temporal']['year_peak'][1]} imgs)" if R['temporal']['year_peak'] else "",
        "",
        "## 9. Ontología vs corpus",
        f"- Vocabulario teórico L3: **{R['ontologia_cobertura']['vocab_l3_size']}** entradas",
        f"- Presentes en corpus: **{R['ontologia_cobertura']['corpus_l3_size']}** entradas",
        f"- Cobertura: **{R['ontologia_cobertura']['coverage_pct']}%**",
        f"- L3 en vocab pero NO en corpus: {R['ontologia_cobertura']['l3_in_vocab_not_corpus']}",
        f"- L3 en corpus pero NO en vocab: {R['ontologia_cobertura']['l3_in_corpus_not_vocab']}",
        "",
        "## Figuras generadas",
        "",
    ])
    for fname in sorted((OUT_DIR / "figures").iterdir()):
        md_lines.append(f"- `figures/{fname.name}`")

    with (OUT_DIR / "eda_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"✓ MD: {OUT_DIR / 'eda_report.md'}")


if __name__ == "__main__":
    main()
