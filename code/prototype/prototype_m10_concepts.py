# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
m10_concepts_mel.py · Módulo M10 E4 multitarea: 7 conceptos + melanoma

Inserción al servicio dermapixel-server. Pega este archivo en:
  $DERMAPIXEL_ROOT/scripts/dermapixel_server/m10_concepts_mel.py

Uso desde pipeline.py:
    from m10_concepts_mel import load_m10, classify_m10
    self.m10_model, self.m10_meta = load_m10(DEVICE)
    result = classify_m10(self.m10_model, self.m10_meta, image_pil, device)
"""
from __future__ import annotations
import json
import sys
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(os.environ.get("DERMAPIXEL_ROOT", "./data"))
E4_DIR = ROOT / "output" / "derm7pt_sae_e4"
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


class MultitaskDerm7(nn.Module):
    def __init__(self, encoder, n_per_crit, dim=1024):
        super().__init__()
        self.encoder = encoder
        self.heads_concept = nn.ModuleList([nn.Linear(dim, n) for n in n_per_crit])
        self.head_mel = nn.Linear(dim, 2)
    def forward(self, x):
        feats = self.encoder(x)
        logits_c = [h(feats) for h in self.heads_concept]
        logits_m = self.head_mel(feats)
        return logits_c, logits_m


def _build_m10(n_per_crit, ckpt_state_dict):
    from models.modeling_finetune import panderm_large_patch16_224
    from peft import LoraConfig, get_peft_model
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
    model = MultitaskDerm7(encoder, n_per_crit)
    msg = model.load_state_dict(ckpt_state_dict, strict=False)
    return model


def load_m10(device):
    """Carga M10 E4 multitarea (7 conceptos + melanoma).

    Returns:
        (model, meta_dict con concept_names, CRIT_VALS, metrics)
    """
    slim_ckpt = E4_DIR / "best_model_slim.pth"
    mapping_path = E4_DIR / "best_concept_mapping.json"
    metrics_path = E4_DIR / "best_metrics.json"

    ckpt = torch.load(slim_ckpt, map_location="cpu", weights_only=False)
    with mapping_path.open() as f: mapping = json.load(f)
    with metrics_path.open() as f: metrics = json.load(f)

    n_per_crit = ckpt["n_per_crit"]
    concept_names = ckpt["concept_names"]
    model = _build_m10(n_per_crit, ckpt["state_dict"])
    model = model.to(device).eval()
    meta = {
        "concept_names": concept_names,
        "CRIT_VALS": mapping["CRIT_VALS"],
        "n_per_crit": n_per_crit,
        "metrics": metrics,
    }
    return model, meta


# Traducción castellano de los 7 conceptos para presentación clínica
CONCEPT_LABELS_ES = {
    "pigment_network":       "Retículo pigmentado",
    "streaks":               "Estrías radiales",
    "pigmentation":          "Pigmentación",
    "regression_structures": "Estructuras de regresión",
    "dots_and_globules":     "Puntos y glóbulos",
    "blue_whitish_veil":     "Velo azul-blanquecino",
    "vascular_structures":   "Estructuras vasculares",
}

VALUE_LABELS_ES = {
    "absent": "ausente",
    "typical": "típico",
    "atypical": "atípico",
    "regular": "regular",
    "irregular": "irregular",
    "present": "presente",
    "diffuse regular": "difusa regular",
    "diffuse irregular": "difusa irregular",
    "localized regular": "localizada regular",
    "localized irregular": "localizada irregular",
    "blue areas": "áreas azules",
    "combinations": "combinaciones",
    "white areas": "áreas blancas",
    "arborizing": "arborizantes",
    "comma": "tipo coma",
    "dotted": "puntiformes",
    "hairpin": "en horquilla",
    "linear irregular": "lineales irregulares",
    "within regression": "dentro de regresión",
    "wreath": "tipo corona",
}


@torch.no_grad()
def classify_m10(model, meta, image_pil, device, tta=True):
    """Inferencia M10: para cada uno de los 7 criterios devuelve probabilidad
    por valor + valor predicho. Más probabilidad melanoma binaria.

    Args:
        tta: si True, aplica 5 augmentations (orig + hflip + vflip + rot90 + rot270)
             y promedia softmax de las 8 heads. Consistente con M1/M7/M9.

    Returns:
        dict con: concepts (lista 7), melanoma {prob, pred}, latency_ms
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
        logits_c_acc = None
        logits_m_acc = None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for fn in tta_fns:
                aug = fn(img_t)
                lc, lm = model(aug)
                # Promediar probabilidades, no logits (más estable)
                probs_c = [F.softmax(li.float(), dim=-1) for li in lc]
                probs_m = F.softmax(lm.float(), dim=-1)
                if logits_c_acc is None:
                    logits_c_acc = probs_c
                    logits_m_acc = probs_m
                else:
                    logits_c_acc = [a + b for a, b in zip(logits_c_acc, probs_c)]
                    logits_m_acc = logits_m_acc + probs_m
        # Media sobre 5 augmentations
        probs_c_final = [a / len(tta_fns) for a in logits_c_acc]
        probs_m_final = logits_m_acc / len(tta_fns)
    else:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_c, logits_m = model(img_t)
        probs_c_final = [F.softmax(li.float(), dim=-1) for li in logits_c]
        probs_m_final = F.softmax(logits_m.float(), dim=-1)

    concepts = []
    for i, c_name in enumerate(meta["concept_names"]):
        probs = probs_c_final[i][0].cpu().numpy()
        values = meta["CRIT_VALS"][c_name]
        argmax = int(probs.argmax())
        pred_value = values[argmax]
        concepts.append({
            "concept_id": c_name,
            "concept_es": CONCEPT_LABELS_ES.get(c_name, c_name),
            "predicted_value": pred_value,
            "predicted_value_es": VALUE_LABELS_ES.get(pred_value, pred_value),
            "predicted_prob": float(probs[argmax]),
            "is_absent": pred_value == "absent",
            "probabilities": {v: float(probs[j]) for j, v in enumerate(values)},
        })

    prob_mel = probs_m_final[0].cpu().numpy()
    melanoma = {
        "probability": float(prob_mel[1]),
        "prediction": int(prob_mel.argmax()),
        "alert_level": "high" if prob_mel[1] > 0.7 else ("medium" if prob_mel[1] > 0.4 else "low"),
    }

    # Interpretación textual breve en castellano
    present_concepts = [c for c in concepts if not c["is_absent"]]
    if present_concepts:
        narrative = "Hallazgos dermatoscópicos: " + ", ".join(
            f"{c['concept_es'].lower()} {c['predicted_value_es']}"
            for c in present_concepts[:5]
        )
    else:
        narrative = "No se identifican hallazgos dermatoscópicos relevantes."

    return {
        "model": "M10_concepts_melanoma",
        "concepts": concepts,
        "melanoma": melanoma,
        "narrative_es": narrative,
        "tta": tta,
        "latency_ms": round((time.time()-t0) * 1000, 1),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Cargando M10 multitarea...")
    model, meta = load_m10(device)
    print(f"  ✓ {len(meta['concept_names'])} conceptos + melanoma")
    print(f"  ✓ Test mel AUROC: {meta['metrics']['test_mel_auroc']:.4f}")
    img = Image.new("RGB", (224, 224), color=(180, 130, 90))
    r = classify_m10(model, meta, img, device)
    print(f"  ✓ Inferencia OK: mel prob={r['melanoma']['probability']:.3f}, "
          f"latency={r['latency_ms']} ms")
    print(f"  Narrativa: {r['narrative_es']}")
