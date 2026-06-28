# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
m9_spanderm_v0.py · Módulo M9 SpanDerm v0 LoRA L2 castellano

Inserción al servicio dermapixel-server. Pega este archivo en:
  $DERMAPIXEL_ROOT/scripts/dermapixel_server/m9_spanderm_v0.py

Uso desde pipeline.py:
    from m9_spanderm_v0 import load_m9, classify_m9
    self.m9_model, self.m9_l2_mapping = load_m9(DEVICE)
    result = classify_m9(self.m9_model, self.m9_l2_mapping, image_pil)
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
SPDERM_DIR = ROOT / "output" / "dermapixel_v1_spanderm_v0_multiseed"
WEIGHTS_DIR = ROOT / "weights" / "weights"

if str(ROOT / "classification") not in sys.path:
    sys.path.insert(0, str(ROOT / "classification"))


NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)

T_EVAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class SpanDermV0(nn.Module):
    def __init__(self, encoder, n_classes_l2, dim=1024):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, n_classes_l2)
    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats)


def _build_spanderm(n_classes, ckpt_state_dict):
    """Construye SpanDerm v0 + LoRA y carga slim state_dict."""
    from models.modeling_finetune import panderm_large_patch16_224
    from peft import LoraConfig, get_peft_model

    # Encoder base
    sd = torch.load(WEIGHTS_DIR / "panderm_large.pth", map_location="cpu", weights_only=False)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    encoder = panderm_large_patch16_224()
    encoder.load_state_dict(sd, strict=False)
    encoder.head = nn.Identity()
    for p in encoder.parameters(): p.requires_grad = False

    n_blocks = len(encoder.blocks)
    target_modules = []
    for b in range(n_blocks - 2, n_blocks):
        for sub in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            target_modules.append(f"blocks.{b}.{sub}")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1,
                          target_modules=target_modules, bias="none")
    encoder = get_peft_model(encoder, lora_cfg)

    model = SpanDermV0(encoder, n_classes)
    msg = model.load_state_dict(ckpt_state_dict, strict=False)
    # Las missing keys son las del encoder PanDerm base (no LoRA) — esperado
    return model


def load_m9(device):
    """Carga M9 SpanDerm v0 LoRA L2.

    Returns:
        (model, l2_mapping_id_to_name, metrics_dict)
    """
    slim_ckpt = SPDERM_DIR / "best_seed42_slim.pth"
    mapping_path = SPDERM_DIR / "best_seed42_l2_mapping.json"
    metrics_path = SPDERM_DIR / "best_seed42_metrics.json"

    if not slim_ckpt.exists():
        raise FileNotFoundError(f"M9 slim checkpoint no encontrado: {slim_ckpt}")

    ckpt = torch.load(slim_ckpt, map_location="cpu", weights_only=False)
    with mapping_path.open() as f:
        mapping = json.load(f)
    with metrics_path.open() as f:
        metrics = json.load(f)

    n_classes = ckpt["n_classes_l2"]
    model = _build_spanderm(n_classes, ckpt["state_dict"])
    model = model.to(device).eval()
    id_to_l2 = {int(k): v for k, v in mapping["id_to_l2"].items()}
    return model, id_to_l2, metrics


@torch.no_grad()
def classify_m9(model, id_to_l2, image_pil, device, tta=True):
    """Clasifica L2 castellano con TTA opcional (5 augmentations).

    Returns:
        dict con: predictions (top-3), probabilities full, latency_ms, model_id
    """
    import torchvision.transforms.functional as TF
    t0 = time.time()
    image_pil = image_pil.convert("RGB")
    img_t = T_EVAL(image_pil).unsqueeze(0).to(device)

    if tta:
        tta_fns = [
            lambda x: x,
            lambda x: TF.hflip(x),
            lambda x: TF.vflip(x),
            lambda x: TF.rotate(x, 90),
            lambda x: TF.rotate(x, 270),
        ]
        prob_acc = None
        for fn in tta_fns:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(fn(img_t)).float()
            prob = F.softmax(logits, dim=-1)
            prob_acc = prob if prob_acc is None else prob_acc + prob
        prob_final = prob_acc / len(tta_fns)
    else:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(img_t).float()
        prob_final = F.softmax(logits, dim=-1)

    prob_np = prob_final[0].cpu().numpy()
    top3_idx = (-prob_np).argsort()[:3]
    top3 = [{"l2": id_to_l2[int(i)], "prob": float(prob_np[i])} for i in top3_idx]
    return {
        "model": "M9_SpanDerm_v0",
        "predictions_top3": top3,
        "probabilities": {id_to_l2[i]: float(prob_np[i]) for i in range(len(prob_np))},
        "latency_ms": round((time.time()-t0) * 1000, 1),
        "tta": tta,
    }


if __name__ == "__main__":
    # Test rápido de carga
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Cargando M9 SpanDerm v0...")
    model, id_to_l2, metrics = load_m9(device)
    print(f"  ✓ {len(id_to_l2)} clases L2 castellano")
    print(f"  ✓ Test metrics: BAcc={metrics['final_test']['BAcc']:.4f} "
          f"AUROC={metrics['final_test']['AUROC']:.4f}")
    # Test inference con imagen dummy
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = classify_m9(model, id_to_l2, img, device, tta=True)
    print(f"  ✓ Inferencia OK: top1={result['predictions_top3'][0]['l2']}, "
          f"latency={result['latency_ms']} ms")
