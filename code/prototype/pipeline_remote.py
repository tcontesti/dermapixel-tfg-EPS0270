#!/usr/bin/env python3
# =============================================================================
# Material reproducible del TFG EPS0270 — DermapixelAI.
# Pesos y datasets de terceros NO incluidos (ver licencias originales).
# Rutas configurables por entorno: DERMAPIXEL_ROOT (def. ./data).
# =============================================================================
"""
DermApIxel Pipeline -- core inference engine for the DermApIxel
teledermatology prototype.

Loads 8 AI modules onto a single GPU and exposes ``analyze(image_pil)``
plus specialised classifiers (``classify_unified``, ``classify_zero_shot``,
``classify_siglip``).  Each module is fault-tolerant: partial results are
returned even if an individual module fails.

Modules and estimated VRAM on NVIDIA DGX Spark (Grace-Blackwell, 128 GB):
    M1  Classification   -- PanDerm Large FT on HAM10000 (7 classes, ~1.3 GB)
    M1b Embedding        -- PanDerm Large base ViT-L/16 (CLS token, 1024-d)
    M2  Segmentation     -- SAM2.1-Large FT on ISIC2018 (~0.5 GB)
    M3  Interpretability -- SAE (16 384 features) -> SkinCon concepts (CPU)
    M4  Visual RAG       -- FAISS index, 421 K Derm1M vectors (CPU)
    M5  Reasoning        -- external LLM (not loaded here, called via API)
    M6  Zero-Shot        -- DermLIP v2 (PanDerm-base + PubMedBERT, ~0.78 GB)
    M7  Unified          -- merged43+TTA L3 classifier (43 classes, ~1.3 GB)
    --  SigLIP LP        -- melanoma safety screen (SigLIP ViT-SO400M, ~3.5 GB)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("dermapixel.pipeline")

# ─── Paths (same as dermapixel_demo.py) ──────────────────────────────────────
PATHS = {
    "cls_ckpt": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "output/ft_ham_large/checkpoint-best.pth"),
    "panderm_weights": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "weights/weights/panderm_large.pth"),
    "sam2_ckpt": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "output/sam2_isic2018/best_model.pth"),
    # S36-seg — cascada U-Net F6 → MedSAM2-tiny (M2-INTEGRAL-MAX). El worker
    # carga directo de las rutas Spark donde viven los checkpoints (verificado).
    "medsam2_tiny_ckpt": os.path.expanduser(
        "$DERMAPIXEL_ROOT/output/m2_integral_max/outputs/M2_v10_handoff/best.pt"),
    "unet_f6_ckpt": os.path.expanduser(
        "$DERMAPIXEL_ROOT/output/m2_integral_max/outputs/M2_v5_unet_r101/best.pt"),
    "sae_ckpt": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "output/sae_large/sae_large_best.pth"),
    "faiss_index": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "output/derm1m_rag/faiss_index.bin"),
    "faiss_fnames": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "output/derm1m_rag/filenames.json"),
    "concept_csv": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "datasets/Derm1M_meta/concept.csv"),
    "skincon_map": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "output/skincon_cbm/concept_feature_correlations.json"),
    "dermlip_v2_dir": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "weights/dermlip_v2/"),
    "unified_ckpt": os.path.expanduser(
        "$DERMAPIXEL_ROOT/output/ft_unified_l3_merged43/checkpoint-best.pth"),
    "unified_label_map": os.path.expanduser(
        "$DERMAPIXEL_ROOT/ontology/label_mapping_l3_merged43.json"),
    "unified_hierarchy": os.path.expanduser(
        "$DERMAPIXEL_ROOT/ontology/merged43_hierarchy.json"),
    "siglip_lp": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "output/siglip_lp/siglip_lp_ham7.joblib"),
    "siglip_scaler": os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "output/siglip_lp/siglip_scaler.joblib"),
}

HAM_CLASSES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "seborrheic keratosis",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevus",
    "vascular lesion",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (same as CLI demo)
_cls_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_sam_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── SAE model class ─────────────────────────────────────────────────────────

class _SparseAutoencoder(nn.Module):
    """Sparse Autoencoder (1024 → 16384 features)."""

    def __init__(self, n_input: int = 1024, n_learned: int = 16384):
        super().__init__()
        self.pre_bias = nn.Parameter(torch.zeros(n_input))
        self.post_bias = nn.Parameter(torch.zeros(n_input))
        self.encoder = nn.Linear(n_input, n_learned)
        self.decoder = nn.Linear(n_learned, n_input, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.pre_bias))


# ─── Pipeline class ──────────────────────────────────────────────────────────

class DermApIxelPipeline:
    """DermApIxel integrated pipeline -- loads 8 AI modules and runs inference.

    Modules loaded (total ~9.2 GB VRAM of 128 GB available on DGX Spark):
        M1  PanDerm Large FT       -- HAM10000 7-class classifier (~1.3 GB)
        M1b PanDerm Large base     -- embedding backbone (shared weights with M1)
        M2  SAM2.1-Large FT        -- lesion segmentation (~0.5 GB)
        M3  SAE                    -- 16 384-feature sparse autoencoder (CPU)
        M4  FAISS RAG              -- 421 K Derm1M vectors (CPU)
        M6  DermLIP v2             -- zero-shot via PubMedBERT text (~0.78 GB)
        M7  Unified merged43+TTA   -- 43-class hierarchical L1/L2/L3 (~1.3 GB)
        --  SigLIP LP              -- melanoma safety screen (~3.5 GB)

    Usage::

        pipe = DermApIxelPipeline()
        pipe.load_all()           # ~45 s cold start
        result = pipe.analyze(pil_image)
    """

    def __init__(self) -> None:
        self.models_loaded = False
        self.cls_model = None
        self.siglip_model = None
        self.siglip_lp = None
        self.siglip_scaler = None
        self.siglip_preprocess = None
        self.siglip_available = False
        self.emb_model = None
        self.sam2_model = None
        # S36-seg — cascada U-Net F6 (localizador) → MedSAM2-tiny (refinador).
        self._unet = None
        self._medsam2 = None
        self.sae_model = None
        self.concept_features: dict[str, list[tuple[int, float]]] = {}
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.faiss_index = None
        self.faiss_fnames: list[str] = []
        self.rag_meta: dict[str, dict[str, str]] = {}
        self.dermlip_model = None
        self.dermlip_tokenizer = None
        self.dermlip_preprocess = None
        self.dermlip_available: bool = False
        # M7 — Unified L3 merged43 (43 classes) + TTA, with L1/L2 derived
        self.unified_model = None
        self.unified_preprocess = None
        self.unified_l3_classes: list[str] = []
        self.unified_hierarchy: dict = {}
        self.unified_available: bool = False
        # M9 — SpanDerm v0 LoRA L2 castellano (DermapixelAI 1.0)
        self.m9_model = None
        self.m9_l2_mapping: dict = {}
        self.m9_metrics: dict = {}
        self.m9_available: bool = False
        # M10 — Multitarea 7 conceptos Seven-Point Checklist + melanoma (Derm7pt)
        self.m10_model = None
        self.m10_meta: dict = {}
        self.m10_available: bool = False
        # M4-bis — FAISS RAG sobre DermapixelAI castellano
        self.m4bis_index = None
        self.m4bis_metadata: list = []
        self.m4bis_available: bool = False
        self.load_time_s: float = 0.0

    # ─── Loading ──────────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Load all 8 models into GPU/CPU memory.  Idempotent.

        Total VRAM footprint is ~9.2 GB of the 128 GB available on the
        DGX Spark.  Cold-start takes ~45 s (dominated by SigLIP download
        on first run).
        """
        if self.models_loaded:
            logger.info("Models already loaded, skipping")
            return

        t0 = time.time()
        logger.info("Loading DermApIxel pipeline (7 models)...")

        logger.info("[1/8] Classification (PanDerm Large FT)...")
        self.cls_model = self._load_classifier()

        logger.info("[2/8] Embedding (PanDerm Large base)...")
        self.emb_model = self._load_embedding_model()

        logger.info("[3/8] Segmentation (SAM2.1-Large FT legacy)...")
        self.sam2_model = self._load_sam2()

        logger.info("[3b/8] Segmentation cascade (U-Net F6 + MedSAM2-tiny)...")
        self._unet = self._load_unet()
        self._medsam2 = self._load_medsam2()

        logger.info("[4/8] Interpretability (SAE Large)...")
        self.sae_model, self.concept_features = self._load_sae()

        logger.info("[5/8] Visual RAG (FAISS + Derm1M)...")
        self.faiss_index, self.faiss_fnames, self.rag_meta = self._load_rag()

        logger.info("[6/8] Zero-Shot (DermLIP v2)...")
        try:
            (
                self.dermlip_model,
                self.dermlip_tokenizer,
                self.dermlip_preprocess,
            ) = self._load_dermlip()
            self.dermlip_available = True
        except Exception as e:
            logger.error("DermLIP v2 load failed; zero-shot will be unavailable", exc_info=True)
            self.dermlip_available = False

        logger.info("[7/8] Unified L3 merged43+TTA (43 classes, hierarchical L1/L2/L3)...")
        try:
            (
                self.unified_model,
                self.unified_preprocess,
                self.unified_l3_classes,
                self.unified_hierarchy,
            ) = self._load_unified_merged43()
            self.unified_available = True
        except Exception as e:
            logger.error("Unified merged43 load failed; /analyze/unified will be unavailable", exc_info=True)
            self.unified_available = False

        logger.info("[8/11] SigLIP safety screen (LP melanoma detection)...")
        try:
            (
                self.siglip_model,
                self.siglip_preprocess,
                self.siglip_lp,
                self.siglip_scaler,
            ) = self._load_siglip()
            self.siglip_available = True
        except Exception as e:
            logger.error("SigLIP load failed; safety screen unavailable", exc_info=True)
            self.siglip_available = False

        logger.info("[9/11] M9 SpanDerm v0 LoRA L2 castellano (DermapixelAI)...")
        try:
            from m9_spanderm_v0 import load_m9 as _load_m9_fn
            self.m9_model, self.m9_l2_mapping, self.m9_metrics = _load_m9_fn(DEVICE)
            self.m9_available = True
            logger.info(f"  M9 OK: {len(self.m9_l2_mapping)} clases L2 ES, "
                        f"BAcc={self.m9_metrics['final_test']['BAcc']:.4f} "
                        f"AUROC={self.m9_metrics['final_test']['AUROC']:.4f}")
        except Exception as e:
            logger.error("M9 SpanDerm v0 load failed; L2 castellano unavailable", exc_info=True)
            self.m9_available = False

        logger.info("[10/11] M10 multitarea 7 conceptos + melanoma (Derm7pt)...")
        try:
            from m10_concepts_mel import load_m10 as _load_m10_fn
            self.m10_model, self.m10_meta = _load_m10_fn(DEVICE)
            self.m10_available = True
            logger.info(f"  M10 OK: {len(self.m10_meta['concept_names'])} conceptos, "
                        f"mel AUROC={self.m10_meta['metrics']['test_mel_auroc']:.4f}")
        except Exception as e:
            logger.error("M10 multitarea load failed; conceptos unavailable", exc_info=True)
            self.m10_available = False

        logger.info("[11/11] M4-bis FAISS DermapixelAI (RAG castellano)...")
        try:
            from m4bis_dermapixel import load_m4bis as _load_m4bis_fn
            self.m4bis_index, self.m4bis_metadata = _load_m4bis_fn()
            self.m4bis_available = True
            logger.info(f"  M4-bis OK: {self.m4bis_index.ntotal} vectores, "
                        f"{len(self.m4bis_metadata)} entradas metadata")
        except Exception as e:
            logger.error("M4-bis FAISS load failed; RAG castellano unavailable", exc_info=True)
            self.m4bis_available = False

        self.models_loaded = True
        self.load_time_s = time.time() - t0
        logger.info(
            f"Pipeline loaded in {self.load_time_s:.1f}s "
            f"({len(self.concept_features)} concepts, "
            f"{self.faiss_index.ntotal:,} RAG vectors, "
            f"zero_shot={'ok' if self.dermlip_available else 'disabled'}, "
            f"unified={'ok' if self.unified_available else 'disabled'}, "
            f"m9={'ok' if self.m9_available else 'disabled'}, "
            f"m10={'ok' if self.m10_available else 'disabled'}, "
            f"m4bis={'ok' if self.m4bis_available else 'disabled'})"
        )

    def _load_classifier(self) -> nn.Module:
        sys.path.insert(0, os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "classification"))
        from models.modeling_finetune import panderm_large_patch16_224_finetune
        ckpt = torch.load(PATHS["cls_ckpt"], map_location="cpu", weights_only=False)
        cargs = ckpt["args"]
        model = panderm_large_patch16_224_finetune(
            pretrained=False, num_classes=7,
            drop_rate=getattr(cargs, "drop", 0.0),
            drop_path_rate=getattr(cargs, "drop_path", 0.2),
            attn_drop_rate=getattr(cargs, "attn_drop_rate", 0.0),
            drop_block_rate=None,
            use_mean_pooling=getattr(cargs, "use_mean_pooling", True),
            init_scale=getattr(cargs, "init_scale", 0.001),
            use_rel_pos_bias=getattr(cargs, "rel_pos_bias", True),
            init_values=getattr(cargs, "layer_scale_init_value", 0.1),
            lin_probe=False,
        )
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval().to(DEVICE)
        return model

    def _load_embedding_model(self) -> nn.Module:
        """Load the PanDerm Large backbone used for RAG/SAE embeddings.

        IMPORTANT (2026-04-11): this loader previously used
        `timm.create_model("vit_large_patch16_224")`, which does NOT include
        LayerScale (gamma_1/gamma_2) nor split q_bias/v_bias biases. Loading
        the PanDerm Large checkpoint into that stock ViT with strict=False
        silently left 24 qkv.bias missing and 96 PanDerm-specific keys
        ignored, so the embedding model in production was a partially
        randomly-initialised network. Symptoms: M3 SAE/Concepts returned
        concepts incoherent with the diagnosis (e.g. Warty/Exophytic/Friable
        for a melanoma), and M4 RAG returned unrelated nearest neighbours.
        Fixed by loading through `panderm_large_patch16_224` from
        models.modeling_finetune, plus an assert guarding against silent
        regression.
        """
        sys.path.insert(0, os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "classification"))
        from models.modeling_finetune import panderm_large_patch16_224

        model = panderm_large_patch16_224()
        ckpt = torch.load(PATHS["panderm_weights"], weights_only=False, map_location="cpu")
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in ckpt.items()
            if k.startswith("encoder.")
        }
        msg = model.load_state_dict(state_dict, strict=False)
        non_head_missing = [k for k in msg.missing_keys if not k.startswith("head.")]
        assert len(non_head_missing) == 0, (
            f"PanDerm Large loader regression: {len(non_head_missing)} "
            f"unexpected missing non-head keys: {non_head_missing[:5]}"
        )
        assert len(msg.unexpected_keys) == 0, (
            f"PanDerm Large loader regression: {len(msg.unexpected_keys)} "
            f"unexpected keys: {msg.unexpected_keys[:5]}"
        )
        logger.info(
            f"Embedding model loaded (panderm_large_patch16_224): "
            f"missing(non-head)=0, unexpected=0"
        )
        model.eval().to(DEVICE)
        return model

    def _load_sam2(self) -> Any:
        from sam2.build_sam import build_sam2_hf
        sam2 = build_sam2_hf("facebook/sam2.1-hiera-large")
        ft_ckpt = torch.load(PATHS["sam2_ckpt"], map_location=DEVICE, weights_only=False)
        sam2.load_state_dict(ft_ckpt["model_state_dict"], strict=False)
        sam2.eval().to(DEVICE)
        return sam2

    def _load_unet(self) -> Any:
        """U-Net ResNet101 F6 — localizador automático (bbox-friendly, recall 0.942)."""
        import segmentation_models_pytorch as smp
        unet = smp.Unet(
            encoder_name="resnet101", encoder_weights=None, classes=1, activation=None,
        ).to(DEVICE)
        state = torch.load(PATHS["unet_f6_ckpt"], map_location=DEVICE, weights_only=False)
        unet.load_state_dict(
            state["state"] if (isinstance(state, dict) and "state" in state) else state
        )
        unet.eval()
        return unet

    def _load_medsam2(self) -> Any:
        """MedSAM2-tiny — refinador box-prompted (Dice 0.9562, 25.6ms). Build idéntico
        al reference handoff: SAM2.1-hiera-tiny + encoder médico MedSAM2_latest.pt +
        decoder/prompt slim (best['state']). El image_encoder va congelado."""
        from sam2.build_sam import build_sam2_hf
        from huggingface_hub import hf_hub_download
        m = build_sam2_hf("facebook/sam2.1-hiera-tiny").to(DEVICE)
        init = hf_hub_download("wanglab/MedSAM2", "MedSAM2_latest.pt")
        ck = torch.load(init, map_location=DEVICE, weights_only=False)
        m.load_state_dict(
            ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck, strict=False
        )
        best = torch.load(PATHS["medsam2_tiny_ckpt"], map_location=DEVICE, weights_only=False)
        m.load_state_dict(best["state"], strict=False)
        m.eval()
        return m

    def _load_sae(self) -> tuple[_SparseAutoencoder, dict[str, list[tuple[int, float]]]]:
        sae = _SparseAutoencoder(1024, 16384)
        ckpt = torch.load(PATHS["sae_ckpt"], map_location="cpu", weights_only=False)
        sae.load_state_dict(ckpt["model_state_dict"])
        sae.eval().to(DEVICE)

        with open(PATHS["skincon_map"]) as f:
            skincon = json.load(f)

        concept_features: dict[str, list[tuple[int, float]]] = {}
        for concept, data in skincon.items():
            feats: list[tuple[int, float]] = []
            for feat in data["top5_features"]:
                if feat["direction"] == "positive" and feat["auroc"] >= 0.65:
                    feats.append((feat["feature_id"], feat["auroc"]))
                    break  # top-1 only: reduces dilution from noisy lower-ranked features
            if feats:
                concept_features[concept] = feats

        # Load per-feature mean/std for z-score normalization in _interpret
        stats_path = os.path.expanduser(
            "$DERMAPIXEL_ROOT/output/sae_large/feature_stats.npz")
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.feat_mean = stats["mean"]   # [16384]
            self.feat_std = stats["std"]     # [16384]
            logger.info(
                f"SAE feature stats loaded: mean [{self.feat_mean.min():.3f}, "
                f"{self.feat_mean.max():.3f}], std [{self.feat_std.min():.3f}, "
                f"{self.feat_std.max():.3f}]")
        else:
            logger.warning(
                "feature_stats.npz not found — SAE scoring without z-score "
                "normalization (promiscuous concepts may appear)")
            self.feat_mean = None
            self.feat_std = None

        return sae, concept_features

    def _load_rag(self) -> tuple[Any, list[str], dict[str, dict[str, str]]]:
        import faiss
        import pandas as pd

        index = faiss.read_index(PATHS["faiss_index"])
        with open(PATHS["faiss_fnames"]) as f:
            filenames = json.load(f)

        concept_df = pd.read_csv(PATHS["concept_csv"])
        meta: dict[str, dict[str, str]] = {}
        for _, row in concept_df.iterrows():
            bn = os.path.basename(str(row["filename"]))
            meta[bn] = {
                "disease": str(row["disease_label"]),
                "concept": str(row["skin_concept"]),
            }

        return index, filenames, meta

    def _load_dermlip(self) -> tuple[Any, Any, Any]:
        """Load DermLIP v2 (PanDerm-base vision + PubMedBERT text) for zero-shot.

        IMPORTANT: must use the custom open_clip fork from Derm1M repo, not the
        standard pip-installed open_clip. The DermLIP v2 visual encoder is
        PanDerm Base (CAEVisionTransformer with LayerScale + separated q/v
        biases), not the standard ViT instantiated by stock open_clip. The
        previous loader silently loaded a randomly-initialised visual encoder
        because of this mismatch (152 missing keys, 188 unexpected keys
        ignored by strict=False).

        The Derm1M fork at $DERMAPIXEL_ROOT/dermfm_zero/src/open_clip understands the
        'pretrain_path' field of vision_cfg and replaces model.visual with the
        correct PanDerm Base architecture before loading the checkpoint.
        """
        import sys
        from transformers import AutoTokenizer

        derm1m_src = os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "dermfm_zero/src")
        if derm1m_src not in sys.path:
            sys.path.insert(0, derm1m_src)

        # Force re-import: if the standard open_clip was already imported
        # (e.g. by another module), Python's module cache would shadow the
        # custom Derm1M fork. Purge any cached open_clip modules first.
        cached = [m for m in list(sys.modules.keys()) if m.startswith("open_clip")]
        for m in cached:
            del sys.modules[m]

        import open_clip  # now imports from Derm1M fork

        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        )
        model.eval().to(DEVICE)
        logger.info(
            f"DermLIP v2 loaded via Derm1M fork: visual={type(model.visual).__name__}, "
            f"embed_dim=512"
        )

        tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
        return model, tokenizer, preprocess

    # ─── Inference ────────────────────────────────────────────────────────

    def analyze(self, image_pil: Image.Image, m2_model: str = "medsam2_cascade") -> dict[str, Any]:
        """Run the full M1-M4 + SigLIP pipeline on a single PIL image.

        S36-seg: ``m2_model`` selecciona la estrategia de segmentación M2
        (medsam2_cascade default / unet_only / sam2_large_legacy).

        Args:
            image_pil: RGB PIL image of any size (resized internally per module).

        Returns:
            dict with keys: status, inference_time_ms, timings_ms,
            image_size, classifications, segmentation, concepts,
            similar_cases, annotated_image_base64, siglip_safety.
            Individual module errors are caught so partial results are
            still returned.

        Latency: ~1.5-2.5 s/image (GPU, includes TTA x5 for M1).
        """
        if not self.models_loaded:
            raise RuntimeError("Pipeline not loaded — call load_all() first")

        image_pil = image_pil.convert("RGB")
        orig_size = image_pil.size

        timings_ms: dict[str, float] = {}
        t_total = time.time()

        # ── Precompute classifier tensor (shared with embedding model) ──
        img_t = _cls_transform(image_pil).unsqueeze(0).to(DEVICE)

        # ── M1: Classification ──
        classifications: list[dict[str, Any]] = []
        try:
            t0 = time.time()
            probs = self._classify(img_t)
            timings_ms["classification"] = round((time.time() - t0) * 1000, 2)
            for rank, idx in enumerate(np.argsort(probs)[::-1][:3]):
                classifications.append({
                    "class_name": HAM_CLASSES[idx],
                    "probability": float(probs[idx]),
                    "rank": int(rank),
                })
        except Exception as e:
            logger.error("M1 classification failed", exc_info=True)
            timings_ms["classification"] = 0.0
            return self._error_result(f"M1 classification failed: {e}", timings_ms, t_total)

        # ── M1b: Embedding (needed for M3 + M4) ──
        try:
            t0 = time.time()
            embedding = self._get_embedding(img_t)
            timings_ms["embedding"] = round((time.time() - t0) * 1000, 2)
        except Exception as e:
            logger.error("M1b embedding failed", exc_info=True)
            timings_ms["embedding"] = 0.0
            return self._error_result(f"M1b embedding failed: {e}", timings_ms, t_total)

        # ── M2: Segmentation ──
        segmentation: dict[str, Any] = {
            "mask_base64": "", "lesion_percent": 0.0, "m2_model": m2_model,
        }
        mask_orig: Optional[np.ndarray] = None
        try:
            t0 = time.time()
            seg = self._segment(image_pil, m2_model)
            mask_orig = seg["mask"]
            timings_ms["segmentation"] = round((time.time() - t0) * 1000, 2)
            lesion_pct = float((mask_orig > 127).sum()) / mask_orig.size * 100.0
            segmentation = {
                "mask_base64": self._encode_png(mask_orig),
                "lesion_percent": round(lesion_pct, 2),
                "bbox": seg.get("bbox"),
                "m2_model": m2_model,
                "bbox_strategy": seg.get("bbox_strategy"),
                "latency_ms": seg.get("latency_ms"),
            }
        except Exception as e:
            logger.error("M2 segmentation failed", exc_info=True)
            timings_ms["segmentation"] = 0.0
            segmentation["error"] = str(e)
            segmentation["m2_model"] = m2_model

        # ── M3: Interpretability ──
        concepts: list[dict[str, Any]] = []
        try:
            t0 = time.time()
            interp = self._interpret(embedding)
            timings_ms["interpretability"] = round((time.time() - t0) * 1000, 2)
            for name, score in interp["concepts"]:
                concepts.append({
                    "concept_name": name,
                    "score": float(score),
                    "present": bool(score > 0),
                })
        except Exception as e:
            logger.error("M3 interpretability failed", exc_info=True)
            timings_ms["interpretability"] = 0.0

        # ── M4: Visual RAG ──
        similar_cases: list[dict[str, Any]] = []
        try:
            t0 = time.time()
            similar_cases = self._search_similar(embedding, k=20)
            timings_ms["rag_search"] = round((time.time() - t0) * 1000, 2)
        except Exception as e:
            logger.error("M4 RAG search failed", exc_info=True)
            timings_ms["rag_search"] = 0.0

        # ── SigLIP safety screen ──
        siglip_result = None
        if self.siglip_available:
            try:
                siglip_result = self.classify_siglip(image_pil)
                timings_ms["siglip"] = siglip_result.get("timing_ms", 0)
            except Exception as e:
                logger.error("SigLIP safety screen failed", exc_info=True)
                timings_ms["siglip"] = 0

        # ── M9 SpanDerm v0 LoRA L2 castellano ──
        m9_result = None
        if self.m9_available:
            try:
                from m9_spanderm_v0 import classify_m9
                m9_result = classify_m9(self.m9_model, self.m9_l2_mapping,
                                         image_pil, DEVICE, tta=True)
                timings_ms["m9_spanderm"] = m9_result.get("latency_ms", 0)
            except Exception:
                logger.error("M9 SpanDerm v0 failed", exc_info=True)
                timings_ms["m9_spanderm"] = 0

        # ── M10 Multitarea: 7 conceptos Seven-Point Checklist + melanoma ──
        m10_result = None
        if self.m10_available:
            try:
                from m10_concepts_mel import classify_m10
                m10_result = classify_m10(self.m10_model, self.m10_meta,
                                            image_pil, DEVICE)
                timings_ms["m10_concepts"] = m10_result.get("latency_ms", 0)
            except Exception:
                logger.error("M10 multitarea failed", exc_info=True)
                timings_ms["m10_concepts"] = 0

        # ── M4-bis FAISS DermapixelAI (RAG castellano) ──
        m4bis_result = None
        if self.m4bis_available and embedding is not None:
            try:
                from m4bis_dermapixel import query_m4bis
                m4bis_result = query_m4bis(embedding[0], self.m4bis_index,
                                             self.m4bis_metadata, k=5)
                timings_ms["m4bis_rag_es"] = m4bis_result.get("latency_ms", 0)
            except Exception:
                logger.error("M4-bis FAISS failed", exc_info=True)
                timings_ms["m4bis_rag_es"] = 0

        # ── Annotated image ──
        # (Ensemble M11 se calcula al final, una vez recogidos todos los outputs)
        annotated_b64 = ""
        if mask_orig is not None:
            try:
                annotated_b64 = self._make_annotated_image(image_pil, mask_orig)
            except Exception:
                logger.error("Annotated image generation failed", exc_info=True)

        # ── M7 Unified (necesario para M11 ensemble) ──
        # Lo calculamos aquí dentro de analyze() para que el ensemble tenga
        # acceso completo. ~120-180 ms con TTA 5 augmentations.
        unified_full = None
        if self.unified_available:
            try:
                t0 = time.time()
                unified_full = self.classify_unified(image_pil)
                timings_ms["unified_for_ensemble"] = round((time.time() - t0) * 1000, 1)
            except Exception:
                logger.error("M7 unified for ensemble failed", exc_info=True)

        # ── M11 Ensemble ponderado (M1+M7+M9+M4-bis) ──
        ensemble_result = None
        try:
            from m11_ensemble import ensemble as compute_ensemble
            partial = {
                "classifications": classifications,
                "m9_spanderm": m9_result,
                "m4bis_rag_es": m4bis_result,
            }
            ensemble_result = compute_ensemble(partial, unified_result=unified_full)
            timings_ms["m11_ensemble"] = ensemble_result.get("latency_ms", 0)
        except Exception:
            logger.error("M11 ensemble failed", exc_info=True)
            timings_ms["m11_ensemble"] = 0

        total_ms = round((time.time() - t_total) * 1000, 2)
        return {
            "status": "completed",
            "inference_time_ms": total_ms,
            "timings_ms": timings_ms,
            "image_size": list(orig_size),
            "classifications": classifications,
            "segmentation": segmentation,
            "concepts": concepts,
            "similar_cases": similar_cases,
            "annotated_image_base64": annotated_b64,
            "siglip_safety": siglip_result,
            "m9_spanderm": m9_result,
            "m10_concepts": m10_result,
            "m4bis_rag_es": m4bis_result,
            "m11_ensemble": ensemble_result,
        }

    def _error_result(
        self, msg: str, timings_ms: dict[str, float], t_total: float
    ) -> dict[str, Any]:
        return {
            "status": "error",
            "error": msg,
            "inference_time_ms": round((time.time() - t_total) * 1000, 2),
            "timings_ms": timings_ms,
            "classifications": [],
            "segmentation": {"mask_base64": "", "lesion_percent": 0.0},
            "concepts": [],
            "similar_cases": [],
            "annotated_image_base64": "",
        }

    def _load_unified_merged43(self) -> tuple[nn.Module, Any, list[str], dict]:
        """Load the L3 merged43 finetune classifier (43 classes) for the
        unified hierarchical view of /analyze/unified.

        The model is the PanDerm Large finetune trained on the unified
        ontology (see $DERMAPIXEL_ROOT/output/ft_unified_l3_merged43/). It is
        instantiated with the EXACT args from the training checkpoint
        (use_mean_pooling=True, rel_pos_bias=True, init_values=0.1) so
        the strict load succeeds.

        Returns the model, its preprocess transform, the ordered list of
        43 L3 class names (according to label_mapping_l3_merged43.json),
        and the precomputed L3→L2/L1 hierarchy from the Rosa ontology.
        """
        sys.path.insert(0, os.path.join(os.environ.get("DERMAPIXEL_ROOT", "./data"), "classification"))
        from models.modeling_finetune import panderm_large_patch16_224_finetune

        ckpt = torch.load(PATHS["unified_ckpt"], map_location="cpu",
                          weights_only=False)
        cargs = ckpt["args"]
        model = panderm_large_patch16_224_finetune(
            pretrained=False, num_classes=43,
            drop_rate=getattr(cargs, "drop", 0.0),
            drop_path_rate=getattr(cargs, "drop_path", 0.2),
            attn_drop_rate=getattr(cargs, "attn_drop_rate", 0.0),
            drop_block_rate=None,
            use_mean_pooling=getattr(cargs, "use_mean_pooling", True),
            init_scale=getattr(cargs, "init_scale", 0.001),
            use_rel_pos_bias=getattr(cargs, "rel_pos_bias", True),
            init_values=getattr(cargs, "layer_scale_init_value", 0.1),
            lin_probe=False,
        )
        msg = model.load_state_dict(ckpt["model"], strict=False)
        non_head_missing = [k for k in msg.missing_keys
                            if not k.startswith("head.")]
        assert len(non_head_missing) == 0, (
            f"merged43 loader regression: non-head missing="
            f"{non_head_missing[:5]}"
        )
        assert len(msg.unexpected_keys) == 0, (
            f"merged43 loader regression: unexpected="
            f"{msg.unexpected_keys[:5]}"
        )
        model.eval().to(DEVICE)

        with open(PATHS["unified_label_map"]) as f:
            label_map = json.load(f)
        l3_classes = [label_map[str(i)] for i in range(43)]

        with open(PATHS["unified_hierarchy"]) as f:
            hierarchy = json.load(f)

        # Sanity-check that label_mapping order matches hierarchy.L3_classes
        assert l3_classes == hierarchy["L3_classes"], (
            "label_mapping_l3_merged43.json order != merged43_hierarchy.L3_classes"
        )

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        logger.info(
            f"Unified L3 merged43 loaded: num_classes=43, "
            f"non_head_missing=0, unexpected=0, "
            f"L1={len(hierarchy['L1_classes'])}, "
            f"L2={len(hierarchy['L2_classes'])}, "
            f"L3={len(hierarchy['L3_classes'])}"
        )
        return model, preprocess, l3_classes, hierarchy

    # ─── Module implementations ───────────────────────────────────────────

    # ─── SigLIP safety screen ─────────────────────────────────────────────

    def _load_siglip(self):
        """Load SigLIP ViT-SO400M + serialized LP for melanoma safety."""
        import joblib
        from transformers import SiglipModel, AutoProcessor

        logger.info("  Loading SigLIP ViT-SO400M-14-384...")
        model = SiglipModel.from_pretrained(
            "google/siglip-so400m-patch14-384"
        ).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

        lp = joblib.load(PATHS["siglip_lp"])
        scaler = joblib.load(PATHS["siglip_scaler"])
        logger.info(f"  SigLIP LP loaded: {lp.classes_.shape[0]} classes, "
                     f"embedding_dim={scaler.mean_.shape[0]}")
        return model, processor, lp, scaler

    def classify_siglip(self, image_pil: Image.Image) -> dict[str, Any]:
        """Run SigLIP LP classification as melanoma safety screen.

        Returns dict with classifications and melanoma-specific flags.
        NOT the primary classifier -- only a safety net that catches
        melanomas that M1 and M7 miss (uncorrelated errors due to different
        backbones). In ensemble eval, M1+SigLIP OR top-3 = 100% recall.
        """
        if not self.siglip_available:
            return {"error": "SigLIP not available", "melanoma_flag": False}

        t0 = time.time()
        image_pil = image_pil.convert("RGB")

        # Extract embedding
        inputs = self.siglip_preprocess(
            images=[image_pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.siglip_model.vision_model(
                **{k: v for k, v in inputs.items()
                   if k in ["pixel_values", "attention_mask"]}
            )
            emb = (outputs.pooler_output
                   if hasattr(outputs, "pooler_output")
                   and outputs.pooler_output is not None
                   else outputs.last_hidden_state[:, 0])
        emb_np = emb.cpu().float().numpy()

        # Scale + predict
        emb_scaled = self.siglip_scaler.transform(emb_np)
        probs = self.siglip_lp.predict_proba(emb_scaled)[0]

        elapsed_ms = (time.time() - t0) * 1000

        # Build response
        sorted_idx = np.argsort(probs)[::-1]
        classifications = []
        for rank, idx in enumerate(sorted_idx):
            classifications.append({
                "class_name": HAM_CLASSES[idx],
                "probability": float(probs[idx]),
                "rank": int(rank),
            })

        mel_idx = HAM_CLASSES.index("melanoma")
        mel_prob = float(probs[mel_idx])
        mel_rank = int(np.where(sorted_idx == mel_idx)[0][0])
        mel_flag = mel_rank < 3  # melanoma in top-3

        return {
            "classifications": classifications,
            "melanoma_flag": mel_flag,
            "melanoma_prob": mel_prob,
            "melanoma_rank": mel_rank,
            "timing_ms": round(elapsed_ms, 2),
        }

    def _classify(self, img_t: torch.Tensor) -> np.ndarray:
        """M1 HAM10000 7-class classification with 5-view TTA.

        Args:
            img_t: preprocessed tensor [1, 3, 224, 224] on DEVICE.

        Returns:
            numpy array of shape (7,) with softmax probabilities.

        TTA: identity + hflip + vflip + rot90 + rot270.  Logits averaged
        before softmax (same scheme as M7 classify_unified).
        Improves melanoma recall +4.3pp at cost of -2.6pp BAcc overall.
        Latency: ~120 ms (5 forward passes with AMP).
        """
        import torchvision.transforms.functional as TF
        tta_fns = [
            lambda x: x,
            lambda x: TF.hflip(x),
            lambda x: TF.vflip(x),
            lambda x: TF.rotate(x, 90),
            lambda x: TF.rotate(x, 270),
        ]
        logits_acc = None
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            for fn in tta_fns:
                aug = fn(img_t)
                logits = self.cls_model(aug).float()
                logits_acc = logits if logits_acc is None else logits_acc + logits
        return F.softmax(logits_acc / len(tta_fns), dim=-1)[0].cpu().numpy()

    def _get_embedding(self, img_t: torch.Tensor) -> np.ndarray:
        """Extract CLS-token embedding from PanDerm Large base (1024-d).

        Args:
            img_t: preprocessed tensor [1, 3, 224, 224] on DEVICE.

        Returns:
            numpy array of shape [1, 1024].  Used downstream by M3 (SAE)
            and M4 (FAISS RAG).  Latency: ~15 ms.
        """
        # PanDerm's forward_features(is_train=False) already returns the CLS
        # token (or mean-pooled patch features, depending on config) as a
        # [B, 1024] tensor. The previous version of this function indexed
        # `features[:, 0]` assuming features was `[B, seq, 1024]`, which was
        # correct for timm's ViT but not for PanDerm Large custom.
        with torch.no_grad():
            features = self.emb_model.forward_features(img_t, is_train=False)
        return features.float().cpu().numpy()  # [1, 1024]

    def _segment(self, image_pil: Image.Image, m2_model: str = "medsam2_cascade") -> dict[str, Any]:
        """Router de segmentación M2 (S36-seg). Devuelve dict con la máscara a
        resolución original (uint8 0/255), el bbox usado (coords 1024², o None),
        la estrategia de bbox y la latencia ms.

            medsam2_cascade   — U-Net F6 → bbox componente mayor +10% → MedSAM2-tiny
            unet_only         — solo U-Net F6 (baseline research)
            sam2_large_legacy — SAM2.1-Large con caja central 80% (M2_v0)
        """
        if m2_model == "medsam2_cascade":
            return self._segment_cascade(image_pil)
        if m2_model == "unet_only":
            return self._segment_unet_only(image_pil)
        if m2_model == "sam2_large_legacy":
            return self._segment_sam2_legacy(image_pil)
        raise ValueError(f"Unknown m2_model={m2_model}")

    def _box_forward(self, model: Any, img_t: "torch.Tensor", bbox_t: "torch.Tensor") -> np.ndarray:
        """Forward box-prompted de un modelo SAM2/MedSAM2 → máscara binaria 1024²
        (np float {0,1}). Idéntico al reference handoff: solo cambia el modelo."""
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            backbone_out = model.forward_image(img_t)
            _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
            if model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(1, -1, *fs)
                for feat, fs in zip(vision_feats[::-1], feat_sizes[::-1])
            ][::-1]
            sparse_emb, dense_emb = model.sam_prompt_encoder(
                points=None, boxes=bbox_t, masks=None,
            )
            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=feats[-1],
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=feats[:-1],
            )
            mask_1024 = F.interpolate(
                low_res_masks, (1024, 1024), mode="bilinear", align_corners=False
            )
            return (torch.sigmoid(mask_1024) > 0.5).float()[0, 0].cpu().numpy()

    @staticmethod
    def _resize_to_orig(mask_1024_bin: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        """Máscara binaria 1024² {0,1} → uint8 0/255 a resolución original."""
        return np.array(
            Image.fromarray((mask_1024_bin * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.NEAREST
            )
        )

    def _segment_cascade(self, image_pil: Image.Image) -> dict[str, Any]:
        """U-Net F6 localiza → bbox componente conexo mayor +10% pad → MedSAM2-tiny.
        Fallback monótono a caja central 80% si U-Net no detecta nada (≥0.5% píxeles)."""
        t0 = time.time()
        orig_w, orig_h = image_pil.size
        img_t = _sam_transform(image_pil).unsqueeze(0).to(DEVICE)

        # Step 1 — U-Net → máscara cruda 1024²
        with torch.no_grad():
            logits = self._unet(img_t)
            mask_unet = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()

        # Step 2 — componente conexo mayor → bbox +10% (regla F8: nunca apretada)
        from scipy.ndimage import label as scipy_label
        labeled, n = scipy_label(mask_unet)
        if n == 0 or mask_unet.sum() < 0.005 * mask_unet.size:
            bbox_1024 = np.array([102.4, 102.4, 921.6, 921.6], dtype=np.float32)
            bbox_strategy = "fallback_center80"
        else:
            sizes = np.bincount(labeled.ravel())[1:]
            largest = int(np.argmax(sizes)) + 1
            ys, xs = np.where(labeled == largest)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            side = max(x_max - x_min, y_max - y_min)
            pad = int(side * 0.10)
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(1024, x_max + pad)
            y_max = min(1024, y_max + pad)
            bbox_1024 = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            bbox_strategy = "unet_largest_component_pad10"

        # Step 3 — MedSAM2-tiny con bbox prompt
        bbox_t = torch.tensor(bbox_1024, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mask_bin = self._box_forward(self._medsam2, img_t, bbox_t)
        return {
            "mask": self._resize_to_orig(mask_bin, orig_w, orig_h),
            "bbox": bbox_1024.tolist(),
            "bbox_strategy": bbox_strategy,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    def _segment_unet_only(self, image_pil: Image.Image) -> dict[str, Any]:
        """Solo U-Net F6 (baseline research F6). Máscara directa + bbox derivado."""
        t0 = time.time()
        orig_w, orig_h = image_pil.size
        img_t = _sam_transform(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self._unet(img_t)
            mask_unet = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()
        ys, xs = np.where(mask_unet)
        bbox_1024 = (
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            if len(xs) else None
        )
        return {
            "mask": self._resize_to_orig(mask_unet.astype(np.float32), orig_w, orig_h),
            "bbox": bbox_1024,
            "bbox_strategy": "unet_only_derived",
            "latency_ms": int((time.time() - t0) * 1000),
        }

    def _segment_sam2_legacy(self, image_pil: Image.Image) -> dict[str, Any]:
        """SAM2.1-Large con caja central 80% (M2_v0, comportamiento histórico)."""
        t0 = time.time()
        orig_w, orig_h = image_pil.size
        img_t = _sam_transform(image_pil).unsqueeze(0).to(DEVICE)
        margin = 0.1
        bbox_1024 = np.array(
            [1024 * margin, 1024 * margin, 1024 * (1 - margin), 1024 * (1 - margin)],
            dtype=np.float32,
        )
        bbox_t = torch.tensor(bbox_1024, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        mask_bin = self._box_forward(self.sam2_model, img_t, bbox_t)
        return {
            "mask": self._resize_to_orig(mask_bin, orig_w, orig_h),
            "bbox": bbox_1024.tolist(),
            "bbox_strategy": "center80",
            "latency_ms": int((time.time() - t0) * 1000),
        }

    def _interpret(self, embedding: np.ndarray) -> dict[str, Any]:
        """M3: Score each SkinCon concept via its mapped SAE features.

        Args:
            embedding: numpy array [1, 1024] from ``_get_embedding()``.

        Returns:
            dict with n_active_features, n_total_features (16384),
            and concepts (list of (name, score) tuples sorted descending).

        After fix R1+R2 (2026-04-12): each concept uses its top-1
        feature only (R2, threshold AUROC >= 0.65).  The raw activation
        is z-score normalised per feature (R1) using precomputed
        mean/std from $DERMAPIXEL_ROOT/output/sae_large/feature_stats.npz, so
        features with high mean activation no longer dominate the score.
        The final score is clipped to [0, 1].  Latency: ~5 ms (CPU).
        """
        emb_t = torch.tensor(embedding, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            z = self.sae_model.encode(emb_t)
        activations = z[0].cpu().numpy()

        n_active = int((activations > 0).sum())

        concept_scores: list[tuple[str, float]] = []
        for concept, feats in self.concept_features.items():
            score = 0.0
            n = 0
            for fid, auroc in feats:
                if self.feat_mean is not None:
                    zscore = (float(activations[fid]) - self.feat_mean[fid]) / max(self.feat_std[fid], 1e-8)
                else:
                    zscore = float(activations[fid])
                if zscore > 1.0:  # feature must be > 1 std above its mean
                    score += (zscore - 1.0) * (auroc - 0.5) * 2
                    n += 1
            score = score / max(n, 1)
            score = float(max(0.0, min(1.0, score / 4.0)))
            concept_scores.append((concept, score))

        concept_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "n_active_features": n_active,
            "n_total_features": 16384,
            "concepts": concept_scores,
        }

    def _search_similar(self, embedding: np.ndarray, k: int = 20) -> list[dict[str, Any]]:
        """M4: Top-k FAISS cosine search with metadata filter.

        Args:
            embedding: numpy array [1, 1024] (L2-normalised internally).
            k: number of raw candidates to retrieve (filtered down to 5).

        Returns:
            list of dicts with filename, similarity, disease, concept.
            Latency: ~2 ms (CPU, IVF index).
        """
        import faiss
        query = embedding.astype("float32").copy()
        faiss.normalize_L2(query)
        D, I = self.faiss_index.search(query, k)
        results: list[dict[str, Any]] = []
        for dist, idx in zip(D[0], I[0]):
            fname = self.faiss_fnames[idx]
            m = self.rag_meta.get(fname)
            if m is None:
                continue
            results.append({
                "filename": fname,
                "similarity": float(dist),
                "disease": m["disease"],
                "concept": m["concept"],
            })
            if len(results) >= 5:
                break
        return results

    # ─── M6: Zero-shot classification ─────────────────────────────────────

    def classify_zero_shot(
        self, image_pil: Image.Image, class_descriptions: list[str]
    ) -> dict[str, Any]:
        """Classify a PIL image against an arbitrary list of text descriptions.

        Uses DermLIP v2 (PanDerm-base vision encoder + PubMedBERT text encoder,
        trained on Derm1M, 403,563 skin image-text pairs).

        Args:
            image_pil: PIL image
            class_descriptions: list of candidate class strings (e.g.
                ["melanoma", "eczema with scaling", "healthy skin"]).

        Returns:
            {
                "predictions": [{"class_name", "similarity", "rank"}, ...],
                "model": "DermLIP v2",
                "timing_ms": float,
            }
        """
        if not self.dermlip_available or self.dermlip_model is None:
            raise RuntimeError("DermLIP v2 is not loaded; zero-shot unavailable")
        if not class_descriptions:
            raise ValueError("class_descriptions must be a non-empty list")

        t0 = time.time()
        image_pil = image_pil.convert("RGB")

        # Preprocess image
        img_tensor = self.dermlip_preprocess(image_pil).unsqueeze(0).to(DEVICE)

        # Tokenize (PubMedBERT via HF tokenizer)
        tokens = self.dermlip_tokenizer(
            list(class_descriptions),
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(DEVICE)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            img_features = self.dermlip_model.encode_image(img_tensor)
            txt_features = self.dermlip_model.encode_text(input_ids)

            img_features = F.normalize(img_features.float(), dim=-1)
            txt_features = F.normalize(txt_features.float(), dim=-1)

            # Cosine similarity + CLIP standard temperature
            logits = (20.0 * img_features @ txt_features.T).softmax(dim=-1)
            similarities = logits[0].cpu().numpy()

        indices = similarities.argsort()[::-1]
        predictions = [
            {
                "class_name": class_descriptions[int(i)],
                "similarity": float(similarities[int(i)]),
                "rank": int(rank),
            }
            for rank, i in enumerate(indices)
        ]

        return {
            "predictions": predictions,
            "model": "DermLIP v2",
            "timing_ms": round((time.time() - t0) * 1000, 2),
        }

    def classify_unified(self, image_pil: Image.Image) -> dict[str, Any]:
        """Classify a PIL image with the L3 merged43 finetune (43 classes)
        using TTA (5 augmentations on the normalized tensor, logits averaged
        before softmax). Derives the L2 (26-class) and L1 (4-class) marginal
        probabilities by summing softmax probs over the L3→L2/L1 partitions
        of the Rosa ontology.

        The TTA implementation reproduces the configuration evaluated in
        the TFG (Cuadro 8.12, Acc 0.810, BAcc 0.818, Top-3 0.954). Sampled
        verification on 500 random test images of the merged43 split gives
        Top-3 = 0.954, matching the TFG figure.

        Returns:
            {
              "level_3": {"top_class", "top_prob", "top3", "all_probs"},
              "level_2": {"top_class", "top_prob", "top3", "all_probs"},
              "level_1": {"top_class", "top_prob", "top3", "all_probs"},
              "model": "PanDerm Large FT unified merged43 + TTA",
              "tta_augmentations": 5,
              "timing_ms": float,
            }
        """
        if not self.unified_available or self.unified_model is None:
            raise RuntimeError("Unified merged43 not loaded")

        import torchvision.transforms.functional as TF

        t0 = time.time()
        image_pil = image_pil.convert("RGB")
        img_tensor = self.unified_preprocess(image_pil).unsqueeze(0).to(DEVICE)

        # 5-view TTA on the already-normalized tensor (must match the TFG eval)
        tta_fns = [
            lambda x: x,
            lambda x: TF.hflip(x),
            lambda x: TF.vflip(x),
            lambda x: TF.rotate(x, 90),
            lambda x: TF.rotate(x, 270),
        ]

        logits_acc = None
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for fn in tta_fns:
                aug = fn(img_tensor)
                logits = self.unified_model(aug).float()
                logits_acc = logits if logits_acc is None else logits_acc + logits
        logits_mean = logits_acc / len(tta_fns)
        probs_l3 = F.softmax(logits_mean, dim=-1)[0].cpu().numpy()

        # ── L3 view (43 classes) ──
        l3_probs_dict = {self.unified_l3_classes[i]: float(probs_l3[i])
                         for i in range(43)}
        l3_sorted = sorted(l3_probs_dict.items(), key=lambda x: x[1], reverse=True)

        def _build_view(sorted_items, all_probs_dict):
            return {
                "top_class": sorted_items[0][0],
                "top_prob": float(sorted_items[0][1]),
                "top3": [{"class": c, "prob": float(p)}
                         for c, p in sorted_items[:3]],
                "all_probs": all_probs_dict,
            }

        # ── L2 view (26 classes, derived by summing L3 probs over the partition) ──
        l3_to_l2 = self.unified_hierarchy["L3_to_L2"]
        l2_probs: dict[str, float] = {}
        for l3_name, p in l3_probs_dict.items():
            l2 = l3_to_l2[l3_name]
            l2_probs[l2] = l2_probs.get(l2, 0.0) + p
        l2_sorted = sorted(l2_probs.items(), key=lambda x: x[1], reverse=True)

        # ── L1 view (4 classes, same procedure) ──
        l3_to_l1 = self.unified_hierarchy["L3_to_L1"]
        l1_probs: dict[str, float] = {}
        for l3_name, p in l3_probs_dict.items():
            l1 = l3_to_l1[l3_name]
            l1_probs[l1] = l1_probs.get(l1, 0.0) + p
        l1_sorted = sorted(l1_probs.items(), key=lambda x: x[1], reverse=True)

        return {
            "level_3": _build_view(l3_sorted, l3_probs_dict),
            "level_2": _build_view(l2_sorted, l2_probs),
            "level_1": _build_view(l1_sorted, l1_probs),
            "model": "PanDerm Large FT unified merged43 + TTA",
            "tta_augmentations": 5,
            "timing_ms": round((time.time() - t0) * 1000, 2),
        }

    # ─── Image helpers ────────────────────────────────────────────────────

    @staticmethod
    def _encode_png(arr: np.ndarray) -> str:
        """Encode a numpy array as a base64 PNG string."""
        buf = io.BytesIO()
        Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _make_annotated_image(original: Image.Image, mask: np.ndarray) -> str:
        """Blend mask as magenta overlay (alpha 0.4) over the original.

        Returns base64-encoded PNG.
        """
        base = original.convert("RGBA")
        # Magenta overlay where mask > 127, transparent elsewhere.
        mask_bool = mask > 127
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask_bool] = (255, 0, 255, int(0.4 * 255))  # magenta + alpha 0.4
        overlay = Image.fromarray(rgba, mode="RGBA")
        # Make sure sizes match
        if overlay.size != base.size:
            overlay = overlay.resize(base.size, Image.NEAREST)
        composed = Image.alpha_composite(base, overlay)
        buf = io.BytesIO()
        composed.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
