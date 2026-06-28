# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
build_m4bis_faiss.py · P2 · Construir índice FAISS de DermapixelAI 1.0
como módulo M4-bis (RAG castellano) del servicio dermapixel-server.

Output: $DERMAPIXEL_ROOT/output/m4bis_faiss_dermapixel/
  ├── faiss_index_train.bin       (IndexFlatIP, 874 train embeddings)
  ├── metadata_train.json          (874 dicts con info clínica + case_text)
  ├── faiss_index_test.bin         (36 test embeddings, para validación)
  └── metadata_test.json
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path

import numpy as np
import faiss

ROOT     = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
DATASET  = ROOT / "datasets" / "dermapixel_v1"
LP_DIR   = ROOT / "output" / "dermapixel_v1_lp"
OUT_DIR  = ROOT / "output" / "m4bis_faiss_dermapixel"
OUT_DIR.mkdir(parents=True, exist_ok=True)
L2_CANON = {"Trastornos queratinización": "Trastornos de la queratinización"}


def main():
    # Cargar embeddings
    X = np.load(LP_DIR / "panderm_large_embeddings.npy")
    with (LP_DIR / "panderm_large_filenames.json").open() as f:
        filenames = json.load(f)
    fn2idx = {fn: i for i, fn in enumerate(filenames)}
    print(f"Embeddings cacheados: {X.shape}")

    # L2-normalize
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Cargar dataset + cases
    rows = []
    with (DATASET / "dataset_filtered.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ontology_l2"] = L2_CANON.get(r["ontology_l2"], r["ontology_l2"])
            rows.append(r)
    case_text_map = {}
    case_title_map = {}
    with (DATASET / "metadata" / "cases.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            case_text_map[r["case_id"]] = r.get("case_text", "")
            case_title_map[r["case_id"]] = r.get("case_title", "")

    for split in ("train", "test"):
        idx_path = OUT_DIR / f"faiss_index_{split}.bin"
        meta_path = OUT_DIR / f"metadata_{split}.json"

        # Recolectar embeddings + metadata para split
        sp_rows = [r for r in rows if r["split"] == split and r["image_filename"] in fn2idx]
        X_sp = np.stack([X_norm[fn2idx[r["image_filename"]]] for r in sp_rows]).astype(np.float32)
        print(f"\n{split.upper()}: {len(sp_rows)} embeddings, dim={X_sp.shape[1]}")

        index = faiss.IndexFlatIP(X_sp.shape[1])
        index.add(X_sp)
        faiss.write_index(index, str(idx_path))
        print(f"  ✓ {idx_path} ({idx_path.stat().st_size/1024:.1f} KB)")

        metadata = []
        for r in sp_rows:
            case_id = r["case_id"]
            metadata.append({
                "image_filename": r["image_filename"],
                "image_path": r["image_path"],
                "ontology_l1": r["ontology_l1"],
                "ontology_l2": r["ontology_l2"],
                "ontology_l3": r["ontology_l3"],
                "case_id": case_id,
                "case_title": case_title_map.get(case_id, ""),
                "case_text_preview": case_text_map.get(case_id, "")[:500],
                "rosa_verified": r.get("rosa_verified", ""),
                "image_type": r.get("image_type", ""),
                "year": r.get("year", ""),
            })
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✓ {meta_path} ({meta_path.stat().st_size/1024:.1f} KB)")

    print(f"\nÍndice M4-bis listo en {OUT_DIR}")


if __name__ == "__main__":
    main()
