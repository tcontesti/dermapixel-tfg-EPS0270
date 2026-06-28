# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
dermapixel_md5_audit.py · Audit overlap MD5 DermapixelAI vs (Derm1M, HAM10000, BCN20000)

Lee los MD5 que ya están en dataset_filtered.csv (col image_md5) y los compara
contra los MD5 conocidos de cada dataset.

Salida: $DERMAPIXEL_ROOT/output/dermapixel_v1_md5_audit/report.md + overlap.csv

Uso:
    cd $DERMAPIXEL_ROOT
    python3 dermapixel_md5_audit.py
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT       = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET    = ROOT / "datasets" / "dermapixel_v1" / "dataset_filtered.csv"
OUT_DIR    = ROOT / "output" / "dermapixel_v1_md5_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Catálogos MD5 esperados (los datasets ya están extraídos).
# Si alguno no existe, se omite y se reporta.
CATALOGS = {
    "HAM10000":        ROOT / "datasets" / "HAM10000_clean"     / "md5_all.txt",
    "BCN20000":        ROOT / "datasets" / "BCN20000"           / "md5_all.txt",
    "Derm1M":          ROOT / "datasets" / "Derm1M_images"      / "md5_all.txt",
    "Derm7pt":         ROOT / "datasets" / "Derm7pt"            / "md5_all.txt",
    "ISIC2018":        ROOT / "datasets" / "ISIC2018"           / "md5_all.txt",
    "ISIC2017":        ROOT / "datasets" / "ISIC2017"           / "md5_all.txt",
    "PH2":             ROOT / "datasets" / "PH2"                / "md5_all.txt",
    "PAD-UFES":        ROOT / "datasets" / "pad-ufes"           / "md5_all.txt",
    "DDI":             ROOT / "datasets" / "DDI"                / "md5_all.txt",
    "Dermnet":         ROOT / "datasets" / "Dermnet"            / "md5_all.txt",
    "Fitzpatrick17k":  ROOT / "datasets" / "fitzpatrick17k_full"/ "md5_all.txt",
    "HIBA":            ROOT / "datasets" / "HIBA"               / "md5_all.txt",
    "MSKCC":           ROOT / "datasets" / "MSKCC"              / "md5_all.txt",
}


def md5_of(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(search_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    for p in search_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield str(p)


def build_catalog(root_dir: Path, cache_path: Path, n_workers: int = 16) -> set[str]:
    """Calcula MD5 paralelizado de las imágenes en root_dir."""
    if cache_path.exists():
        with cache_path.open() as f:
            return {line.strip() for line in f if line.strip()}
    if not root_dir.exists():
        return set()

    img_dir = root_dir / "images"
    search = img_dir if img_dir.exists() else root_dir
    paths = list(list_images(search))
    print(f"  [{root_dir.name}] {len(paths)} imágenes encontradas; calculando MD5 ({n_workers} workers)...")

    md5s = set()
    if not paths:
        return md5s

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, m in enumerate(ex.map(md5_of, paths, chunksize=64)):
            md5s.add(m)
            if (i + 1) % 20000 == 0:
                print(f"    {i+1}/{len(paths)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for m in sorted(md5s):
            f.write(m + "\n")
    return md5s


def main():
    # 1 · MD5 DermapixelAI
    derm_md5 = []
    with DATASET.open(encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r.get("image_md5"):
                derm_md5.append((r["image_filename"], r["image_md5"],
                                 r.get("ontology_l1", ""),
                                 r.get("split", "")))
    print(f"DermapixelAI: {len(derm_md5)} MD5")
    derm_set = {m for _, m, _, _ in derm_md5}
    print(f"  únicos: {len(derm_set)}")

    # Detección duplicados internos
    md5_counter = Counter(m for _, m, _, _ in derm_md5)
    internal_dups = {m: c for m, c in md5_counter.items() if c > 1}
    if internal_dups:
        print(f"  ! duplicados internos en DermapixelAI: {len(internal_dups)} MD5")

    # 2 · Cargar catálogos
    overlaps = {}
    for name, cache in CATALOGS.items():
        root_dir = cache.parent  # $DERMAPIXEL_ROOT/datasets/<name>/
        catalog = build_catalog(root_dir, cache)
        overlaps[name] = {
            "catalog_size": len(catalog),
            "intersection": len(derm_set & catalog),
            "intersection_md5s": sorted(derm_set & catalog)[:50],
        }
        print(f"  {name}: catalog={len(catalog)} overlap={overlaps[name]['intersection']}")

    # 3 · Report
    report = ["# Auditoría MD5 DermapixelAI 1.0 vs datasets de referencia",
              "",
              f"- Imágenes DermapixelAI filtradas: {len(derm_md5)}",
              f"- MD5 únicos: {len(derm_set)}",
              f"- Duplicados internos: {len(internal_dups)}",
              "",
              "| Dataset | Catálogo | Overlap | % DermapixelAI |",
              "|---------|---------:|--------:|---------------:|"]
    for name, info in overlaps.items():
        pct = 100.0 * info["intersection"] / max(1, len(derm_set))
        report.append(f"| {name} | {info['catalog_size']} | {info['intersection']} | {pct:.2f}% |")
    report.append("")

    # Detalle de overlap (si hay)
    for name, info in overlaps.items():
        if info["intersection"] == 0:
            continue
        report.append(f"## Detalle overlap con {name}")
        for m in info["intersection_md5s"]:
            matches = [fn for fn, md, _, _ in derm_md5 if md == m]
            report.append(f"- `{m}` → {matches}")
        report.append("")

    md_path = OUT_DIR / "report.md"
    md_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport: {md_path}")

    # CSV resumen
    csv_path = OUT_DIR / "overlap.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "catalog_size", "overlap_count", "overlap_pct_dermapixel"])
        for name, info in overlaps.items():
            pct = 100.0 * info["intersection"] / max(1, len(derm_set))
            w.writerow([name, info["catalog_size"], info["intersection"], f"{pct:.4f}"])

    # JSON completo
    with (OUT_DIR / "overlap.json").open("w", encoding="utf-8") as f:
        json.dump({
            "dermapixel": {
                "n": len(derm_md5),
                "n_unique": len(derm_set),
                "n_internal_dups": len(internal_dups),
            },
            "overlaps": overlaps,
        }, f, indent=2, ensure_ascii=False)

    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
