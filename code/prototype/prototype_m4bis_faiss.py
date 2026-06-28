# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
m4bis_dermapixel.py · Módulo M4-bis FAISS sobre DermapixelAI (RAG castellano)

Pega este archivo en:
  $DERMAPIXEL_ROOT/scripts/dermapixel_server/m4bis_dermapixel.py

Uso desde pipeline.py:
    from m4bis_dermapixel import load_m4bis, query_m4bis
    self.m4bis_index, self.m4bis_meta = load_m4bis()

    # Reutilizando embedding ya extraído por _get_embedding:
    result = query_m4bis(emb_np, self.m4bis_index, self.m4bis_meta, k=5)
"""
from __future__ import annotations
import json
import time
import os
from pathlib import Path

import numpy as np
import faiss

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
M4BIS_DIR = ROOT / "output" / "m4bis_faiss_dermapixel"


def load_m4bis():
    """Carga índice FAISS train + metadata.

    Returns:
        (faiss_index, metadata_list)
    """
    idx_path = M4BIS_DIR / "faiss_index_train.bin"
    meta_path = M4BIS_DIR / "metadata_train.json"
    index = faiss.read_index(str(idx_path))
    with meta_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def query_m4bis(image_embedding, index, metadata, k=5):
    """Para un embedding PanDerm Large (1024D), devuelve top-k casos similares
    de DermapixelAI con case_text en castellano.

    Args:
        image_embedding: np.ndarray (1024,) o (1, 1024). Se L2-normaliza
                         internamente.
        index: faiss.IndexFlatIP del corpus train.
        metadata: lista de 874 dicts con info clínica.
        k: número de vecinos.

    Returns:
        dict con: neighbors (list), latency_ms, model_id
    """
    t0 = time.time()
    emb = np.asarray(image_embedding, dtype=np.float32).reshape(1, -1)
    emb = emb / max(np.linalg.norm(emb), 1e-8)

    sims, idx = index.search(emb, k)
    sims = sims[0]; idx = idx[0]

    # Distribución L1/L2/L3 entre los k vecinos
    from collections import Counter
    l1_votes = Counter(metadata[i]["ontology_l1"] for i in idx)
    l2_votes = Counter(metadata[i]["ontology_l2"] for i in idx)
    l3_votes = Counter(metadata[i]["ontology_l3"] for i in idx)

    neighbors = []
    for rank, (sim, i) in enumerate(zip(sims, idx)):
        m = metadata[int(i)]
        neighbors.append({
            "rank": rank + 1,
            "similarity": float(sim),
            "image_filename": m["image_filename"],
            "image_path": m["image_path"],
            "ontology_l1": m["ontology_l1"],
            "ontology_l2": m["ontology_l2"],
            "ontology_l3": m["ontology_l3"],
            "case_id": m["case_id"],
            "case_title": m["case_title"],
            "case_text_preview": m["case_text_preview"],
            "rosa_verified": m.get("rosa_verified", ""),
            "year": m.get("year", ""),
        })

    return {
        "model": "M4bis_FAISS_DermapixelAI",
        "k": int(k),
        "neighbors": neighbors,
        "majority_vote": {
            "L1": l1_votes.most_common(1)[0][0] if l1_votes else None,
            "L2": l2_votes.most_common(1)[0][0] if l2_votes else None,
            "L3": l3_votes.most_common(1)[0][0] if l3_votes else None,
        },
        "votes_distribution": {
            "L1": dict(l1_votes),
            "L2": dict(l2_votes),
            "L3": dict(l3_votes),
        },
        "latency_ms": round((time.time()-t0) * 1000, 1),
    }


if __name__ == "__main__":
    print("Cargando M4-bis FAISS DermapixelAI...")
    index, metadata = load_m4bis()
    print(f"  ✓ {index.ntotal} vectores indexados, dim={index.d}")
    print(f"  ✓ {len(metadata)} entradas metadata")
    # Test con embedding aleatorio (sólo para verificar API)
    fake_emb = np.random.randn(1024).astype(np.float32)
    r = query_m4bis(fake_emb, index, metadata, k=5)
    print(f"  ✓ Query OK: top-1 sim={r['neighbors'][0]['similarity']:.3f}, "
          f"L1={r['majority_vote']['L1']}, latency={r['latency_ms']} ms")
