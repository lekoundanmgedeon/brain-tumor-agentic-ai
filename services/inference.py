"""
Service d'inférence — Détection de tumeurs cérébrales.

Deux modes :
──────────────────────────────────────────────────────────
DEMO            : simulation déterministe (aucun modèle requis)
PRODUCTION_READY: EfficientNet-B5 réel téléchargé depuis HuggingFace
                  Modèle : DunnBC22/efficientnet-b5-Brain_Tumors_Image_Classification
                  Entraîné sur : Kaggle Brain Tumor MRI Dataset (7023 images)
                  Accuracy     : ~80% sur le jeu de test Kaggle
──────────────────────────────────────────────────────────
"""

import os
import logging
import numpy as np
from PIL import Image

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]

# Mapping entre les labels HuggingFace et notre format interne
HF_LABEL_MAP = {
    "Glioma":          "glioma",
    "Meningioma":      "meningioma",
    "No Tumor":        "no_tumor",
    "Pituitary Tumor": "pituitary_tumor",
    "notumor":         "no_tumor",
    "pituitary":       "pituitary_tumor",
    "glioma":          "glioma",
    "meningioma":      "meningioma",
}

TUMOR_LOCATIONS = [
    "région frontale gauche",
    "région pariétale droite",
    "région temporale gauche",
    "région occipitale",
    "région frontale droite",
    "région centrale",
    "lobe temporal droit",
    "cervelet",
]

# ── Singleton modèle ──────────────────────────────────────────
_model_cache     = None
_processor_cache = None
_device_cache    = None
_model_type      = None   # "hf_transformers" | "efficientnet_b0_custom"


# ══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def run_inference(preprocessed_image: np.ndarray, image_path: str) -> dict:
    """
    Lance l'inférence. Dispatch DEMO / PRODUCTION selon APP_MODE.

    Args:
        preprocessed_image: np.ndarray (1, H, W) normalisé [0,1]
        image_path: chemin de l'image originale

    Returns:
        dict structuré avec tumor_detected, confidence, etc.
    """
    if settings.APP_MODE == "PRODUCTION_READY":
        return _run_production_inference(preprocessed_image, image_path)
    return _run_demo_inference(preprocessed_image, image_path)


# ══════════════════════════════════════════════════════════════
# MODE PRODUCTION — EfficientNet-B5 HuggingFace
# ══════════════════════════════════════════════════════════════

def _load_hf_model():
    """
    Charge le modèle HuggingFace depuis models/hf_brain_tumor/ (cache local)
    ou directement depuis HuggingFace Hub si le cache est absent.

    Returns:
        (model, processor, device)
    """
    global _model_cache, _processor_cache, _device_cache, _model_type

    if _model_cache is not None:
        return _model_cache, _processor_cache, _device_cache

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chemin cache local
    local_dir = os.path.join("models", "hf_brain_tumor")
    hf_model_id = "DunnBC22/efficientnet-b5-Brain_Tumors_Image_Classification"

    if os.path.exists(os.path.join(local_dir, "config.json")):
        source = local_dir
        logger.info(f"[Inference] Chargement modèle HuggingFace depuis cache : {local_dir}")
    else:
        source = hf_model_id
        logger.info(f"[Inference] Téléchargement modèle HuggingFace : {hf_model_id}")

    processor = AutoImageProcessor.from_pretrained(source)
    model     = AutoModelForImageClassification.from_pretrained(source)
    model     = model.to(device).eval()

    _model_cache     = model
    _processor_cache = processor
    _device_cache    = device
    _model_type      = "hf_transformers"

    logger.info(f"[Inference] ✅ Modèle chargé sur {device} — {len(model.config.id2label)} classes")
    return model, processor, device


def _run_production_inference(preprocessed_image: np.ndarray, image_path: str) -> dict:
    """
    Inférence réelle avec EfficientNet-B5 HuggingFace.

    Pipeline :
      1. Charger le modèle (singleton)
      2. Préprocesser l'image avec AutoImageProcessor
      3. Forward pass → softmax → classe + confiance
      4. Grad-CAM pour la localisation
      5. Construire le résultat structuré
    """
    import torch
    import torch.nn.functional as F

    # ── Chargement modèle ──────────────────────────────────────
    try:
        model, processor, device = _load_hf_model()
    except Exception as e:
        logger.error(f"[Inference PROD] Modèle indisponible : {e}")
        logger.warning("[Inference PROD] → Fallback DEMO")
        return _run_demo_inference(preprocessed_image, image_path)

    # ── Préprocessing HuggingFace ──────────────────────────────
    try:
        img_pil = Image.open(image_path).convert("RGB")
        inputs  = processor(images=img_pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)   # (1, 3, H, W)
    except Exception as e:
        logger.error(f"[Inference PROD] Erreur preprocessing : {e}")
        return _run_demo_inference(preprocessed_image, image_path)

    # ── Forward pass ───────────────────────────────────────────
    try:
        with torch.no_grad():
            outputs = model(pixel_values)
            logits  = outputs.logits                        # (1, 4)
            probs   = F.softmax(logits, dim=1)[0]          # (4,)
            probs_np = probs.cpu().numpy()

        pred_idx     = int(np.argmax(probs_np))
        hf_label     = model.config.id2label[pred_idx]      # "Glioma" etc.
        pred_class   = HF_LABEL_MAP.get(hf_label, hf_label.lower().replace(" ", "_"))
        confidence   = float(probs_np[pred_idx])
        tumor_detected = pred_class != "no_tumor"

    except Exception as e:
        logger.error(f"[Inference PROD] Erreur forward pass : {e}")
        return _run_demo_inference(preprocessed_image, image_path)

    # ── Grad-CAM (avec gradient) ───────────────────────────────
    gradcam_map  = _compute_gradcam_hf(model, processor, img_pil, pred_idx, device)
    tumor_location = _estimate_location_from_cam(gradcam_map) if tumor_detected else None

    # ── Surface tumorale estimée ───────────────────────────────
    tumor_area_mm2 = None
    if tumor_detected and gradcam_map is not None:
        active_ratio   = float(np.sum(gradcam_map > 0.5) / gradcam_map.size)
        tumor_area_mm2 = round(active_ratio * 220 * 220, 1)

    # ── Distribution des probabilités ─────────────────────────
    class_probs = {}
    for i, hf_lbl in model.config.id2label.items():
        internal = HF_LABEL_MAP.get(hf_lbl, hf_lbl.lower().replace(" ", "_"))
        class_probs[internal] = float(probs_np[int(i)])

    # ── Findings ───────────────────────────────────────────────
    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    prob_str = ", ".join(f"{k}: {v:.1%}" for k, v in sorted_probs)

    findings = [
        f"Modèle : EfficientNet-B5 (DunnBC22/HuggingFace — Kaggle 80.2%)",
        f"Classe prédite : {pred_class.replace('_',' ')} ({confidence:.1%})",
        f"Distribution : {prob_str}",
    ]
    if tumor_detected:
        findings.append("Anomalie détectée — confirmation professionnelle requise")
    else:
        findings.append("Aucune anomalie détectée par le modèle")

    logger.info(
        f"[Inference PROD] {pred_class} — {confidence:.2%} "
        f"({'tumeur' if tumor_detected else 'sain'})"
    )

    return {
        "tumor_detected":       tumor_detected,
        "suspected_tumor_type": pred_class if tumor_detected else "no_tumor",
        "confidence":           round(confidence, 3),
        "tumor_location":       tumor_location,
        "tumor_area_mm2":       tumor_area_mm2,
        "tumor_volume_mm3":     None,
        "technical_findings":   findings,
        "gradcam_map":          gradcam_map,
        "class_probabilities":  class_probs,
        "mode":                 "PRODUCTION_READY",
    }


def _compute_gradcam_hf(model, processor, img_pil, pred_class_idx: int, device) -> np.ndarray:
    """
    Grad-CAM sur la dernière couche convolutive du modèle HuggingFace.
    Fonctionne avec EfficientNet (et la plupart des CNN Transformers).

    Returns:
        Carte d'activation (H, W) normalisée [0,1], ou carte uniforme en cas d'erreur
    """
    import torch
    import torch.nn.functional as F

    try:
        # Recréer un tensor avec gradient activé
        inputs  = processor(images=img_pil, return_tensors="pt")
        pv      = inputs["pixel_values"].to(device).requires_grad_(True)

        features_out = {}
        grads_out    = {}

        # Identifier la dernière couche convolutive
        # Pour EfficientNet HuggingFace : model.efficientnet.top_conv
        # Fallback : parcourir les modules pour trouver le dernier Conv2d
        target_layer = _find_last_conv(model)

        if target_layer is None:
            logger.warning("[GradCAM] Couche cible introuvable — carte uniforme")
            return np.ones((7, 7), dtype=np.float32) * 0.5

        def fwd_hook(m, inp, out):
            features_out["feat"] = out

        def bwd_hook(m, gin, gout):
            grads_out["grad"] = gout[0]

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        try:
            out = model(pv).logits
            model.zero_grad()
            out[0, pred_class_idx].backward()
        finally:
            h1.remove()
            h2.remove()

        feat = features_out.get("feat")
        grad = grads_out.get("grad")

        if feat is None or grad is None:
            return np.ones((7, 7), dtype=np.float32) * 0.5

        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * feat).sum(dim=1)).squeeze()
        cam     = cam.detach().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.astype(np.float32)

    except Exception as e:
        logger.warning(f"[GradCAM] Erreur : {e}")
        return np.ones((7, 7), dtype=np.float32) * 0.5


def _find_last_conv(model) -> "torch.nn.Module | None":
    """Trouve le dernier module Conv2d du modèle (pour Grad-CAM)."""
    import torch.nn as nn
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def _estimate_location_from_cam(cam: np.ndarray) -> str:
    """Estime la localisation tumorale depuis le centroïde de la CAM."""
    if cam is None:
        return "région indéterminée"
    h, w = cam.shape
    y_c, x_c = np.mgrid[:h, :w]
    total = cam.sum() + 1e-8
    rx = float((x_c * cam).sum() / total) / w
    ry = float((y_c * cam).sum() / total) / h

    h_label = "gauche" if rx < 0.45 else ("droite" if rx > 0.55 else "centrale")
    v_label = "frontale" if ry < 0.35 else ("occipitale" if ry > 0.65 else "pariétale")
    return f"région {v_label} {h_label} (estimation Grad-CAM)"


# ══════════════════════════════════════════════════════════════
# MODE DEMO — simulation statistique
# ══════════════════════════════════════════════════════════════

def _run_demo_inference(preprocessed_image: np.ndarray, image_path: str) -> dict:
    """Simulation déterministe basée sur les statistiques de l'image."""
    logger.info("[Inference DEMO] Simulation.")
    arr = preprocessed_image.squeeze()

    mean_i  = float(np.mean(arr))
    var     = float(np.var(arr))
    max_i   = float(np.max(arr))

    seed = int((mean_i * 1000 + var * 100) % 2**31)
    rng  = np.random.default_rng(seed)

    tumor_prob     = min(0.95, 0.3 + var * 3.0 + max_i * 0.4)
    tumor_detected = tumor_prob > 0.5

    if tumor_detected:
        tumor_idx      = int(rng.choice(3, p=[0.50, 0.25, 0.25]))
        tumor_type     = ["glioma", "meningioma", "pituitary_tumor"][tumor_idx]
        confidence     = round(float(rng.uniform(0.65, 0.95)), 3)
        tumor_location = TUMOR_LOCATIONS[int(rng.integers(0, len(TUMOR_LOCATIONS)))]
        tumor_area_mm2 = round(float(rng.uniform(80.0, 450.0)), 1)
        findings       = [
            "Mode DEMO — résultats simulés (non médicaux)",
            "zone hyperintense détectée",
            f"confiance simulée : {confidence:.0%}",
        ]
    else:
        tumor_type     = "no_tumor"
        confidence     = round(float(rng.uniform(0.70, 0.92)), 3)
        tumor_location = None
        tumor_area_mm2 = None
        findings       = [
            "Mode DEMO — résultats simulés (non médicaux)",
            "aucune anomalie détectée",
        ]

    return {
        "tumor_detected":       tumor_detected,
        "suspected_tumor_type": tumor_type,
        "confidence":           confidence,
        "tumor_location":       tumor_location,
        "tumor_area_mm2":       tumor_area_mm2,
        "tumor_volume_mm3":     None,
        "technical_findings":   findings,
        "gradcam_map":          None,
        "class_probabilities":  {},
        "mode":                 "DEMO",
    }
