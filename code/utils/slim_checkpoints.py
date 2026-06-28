# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
slim_checkpoints.py · Reduce los checkpoints SpanDerm v0 + E4 al mínimo
necesario: solo LoRA adapters + cabezas FC (encoder PanDerm Large se carga
aparte desde weights/weights/panderm_large.pth).

Reduce tamaño de 1.2 GB → ~2-3 MB por checkpoint.
"""
from __future__ import annotations
import json
import os
import torch
from pathlib import Path

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
SPDERM = ROOT / "output" / "dermapixel_v1_spanderm_v0_multiseed"
E4     = ROOT / "output" / "derm7pt_sae_e4"


def slim_state_dict(state_dict):
    """Conserva solo claves de LoRA y cabezas FC."""
    kept = {}
    for k, v in state_dict.items():
        k_lower = k.lower()
        # LoRA: cualquier key con "lora" o "lora_a"/"lora_b"
        # Cabeza FC: keys que empiezan por "head" o "heads_concept" o "head_mel"
        if ("lora" in k_lower
            or k.startswith("head.")
            or k.startswith("heads_concept")
            or k.startswith("head_mel")):
            kept[k] = v
    return kept


def slim_spanderm():
    big = SPDERM / "best_seed42.pth"
    out = SPDERM / "best_seed42_slim.pth"
    ckpt = torch.load(big, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    slim_sd = slim_state_dict(sd)
    print(f"SpanDerm v0: {len(sd)} → {len(slim_sd)} keys")
    for k, v in list(slim_sd.items())[:5]:
        print(f"  {k}: {tuple(v.shape)}")

    new_ckpt = dict(ckpt)
    new_ckpt["state_dict"] = slim_sd
    torch.save(new_ckpt, out)
    sz_big = big.stat().st_size / 1e6
    sz_slim = out.stat().st_size / 1e6
    print(f"  {sz_big:.1f} MB → {sz_slim:.2f} MB ({sz_slim/sz_big*100:.2f}%)")
    return out


def slim_e4():
    big = E4 / "best_model.pth"
    out = E4 / "best_model_slim.pth"
    ckpt = torch.load(big, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    slim_sd = slim_state_dict(sd)
    print(f"\nE4 multitarea: {len(sd)} → {len(slim_sd)} keys")
    for k, v in list(slim_sd.items())[:5]:
        print(f"  {k}: {tuple(v.shape)}")

    new_ckpt = dict(ckpt)
    new_ckpt["state_dict"] = slim_sd
    torch.save(new_ckpt, out)
    sz_big = big.stat().st_size / 1e6
    sz_slim = out.stat().st_size / 1e6
    print(f"  {sz_big:.1f} MB → {sz_slim:.2f} MB ({sz_slim/sz_big*100:.2f}%)")
    return out


if __name__ == "__main__":
    sp_slim = slim_spanderm()
    e4_slim = slim_e4()
    print(f"\n✓ Slim checkpoints listos para el prototipo:")
    print(f"  {sp_slim}")
    print(f"  {e4_slim}")
    # Opcional: borrar los .pth originales si quieres ahorrar espacio
    # (mantengo por seguridad — son 2,4 GB)
