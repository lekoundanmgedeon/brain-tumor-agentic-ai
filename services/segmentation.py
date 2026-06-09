"""
Service de segmentation tumorale.

Génère un masque de segmentation binaire et une heatmap de visualisation
à partir des résultats d'inférence.

Deux modes :
- DEMO : segmentation simulée avec formes géométriques réalistes
- PRODUCTION_READY : prêt pour MONAI / nnU-Net / MedSAM
"""

import os
import uuid
import logging
import numpy as np
from PIL import Image, ImageFilter

from config.settings import settings

logger = logging.getLogger(__name__)


def generate_segmentation(
    preprocessed_image: np.ndarray,
    inference_result: dict,
    image_path: str,
) -> dict:
    """
    Génère le masque de segmentation et la heatmap.

    Args:
        preprocessed_image: Tableau NumPy (1, H, W) normalisé
        inference_result: Résultat de l'inférence
        image_path: Chemin de l'image originale

    Returns:
        Dictionnaire avec segmentation_mask_path et heatmap_path
    """
    if not inference_result.get("tumor_detected", False):
        logger.info("[Segmentation] Pas de tumeur détectée — pas de segmentation.")
        return {
            "segmentation_mask_path": None,
            "heatmap_path": None,
        }

    if settings.APP_MODE == "PRODUCTION_READY":
        return _production_segmentation(preprocessed_image, inference_result, image_path)
    else:
        return _demo_segmentation(preprocessed_image, inference_result, image_path)


def _demo_segmentation(
    preprocessed_image: np.ndarray,
    inference_result: dict,
    image_path: str,
) -> dict:
    """
    Segmentation simulée en mode DEMO.

    Génère une forme elliptique réaliste positionnée de façon
    semi-déterministe basée sur la localisation rapportée.
    """
    logger.info("[Segmentation DEMO] Génération du masque simulé.")
    arr = preprocessed_image.squeeze()  # (H, W)
    H, W = arr.shape

    # Générer un seed reproductible
    seed = int(np.sum(arr[:10, :10]) * 1000) % 2**31
    rng = np.random.default_rng(seed)

    # --- Masque de segmentation ---
    mask = np.zeros((H, W), dtype=np.uint8)

    # Centroïde de la tumeur simulée (dans la partie centrale de l'image)
    cx = int(rng.integers(W // 4, 3 * W // 4))
    cy = int(rng.integers(H // 4, 3 * H // 4))

    # Rayon semi-aléatoire (taille réaliste)
    rx = int(rng.integers(W // 10, W // 5))
    ry = int(rng.integers(H // 10, H // 5))

    # Dessiner une ellipse dans le masque
    y_grid, x_grid = np.ogrid[:H, :W]
    ellipse = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2
    mask[ellipse <= 1.0] = 255

    # Ajouter du flou pour des contours plus naturels
    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=3))
    mask_arr = np.array(mask_img)
    mask_binary = (mask_arr > 128).astype(np.uint8) * 255

    # --- Heatmap de visualisation ---
    # Charger l'image originale en RGB
    orig = Image.open(image_path).convert("RGB").resize((W, H), Image.LANCZOS)
    orig_arr = np.array(orig, dtype=np.float32)

    # Créer une heatmap : overlay rouge sur la zone suspecte
    heatmap = orig_arr.copy()
    tumor_region = mask_binary > 128

    # Augmenter le canal rouge, réduire bleu/vert dans la région
    heatmap[tumor_region, 0] = np.clip(heatmap[tumor_region, 0] * 1.5 + 80, 0, 255)
    heatmap[tumor_region, 1] = np.clip(heatmap[tumor_region, 1] * 0.4, 0, 255)
    heatmap[tumor_region, 2] = np.clip(heatmap[tumor_region, 2] * 0.3, 0, 255)

    # Ajouter un contour vert sur le masque
    from PIL import ImageDraw
    heatmap_img = Image.fromarray(heatmap.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(heatmap_img)
    # Contour simplifié autour de l'ellipse
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, outline=(0, 255, 100), width=3)

    # --- Sauvegarde ---
    uid = str(uuid.uuid4())[:8]
    mask_path = os.path.join(settings.MASKS_DIR, f"mask_{uid}.png")
    heatmap_path = os.path.join(settings.HEATMAPS_DIR, f"heatmap_{uid}.png")

    Image.fromarray(mask_binary, mode="L").save(mask_path)
    heatmap_img.save(heatmap_path)

    logger.info(f"[Segmentation DEMO] Masque sauvegardé : {mask_path}")
    logger.info(f"[Segmentation DEMO] Heatmap sauvegardée : {heatmap_path}")

    return {
        "segmentation_mask_path": mask_path,
        "heatmap_path": heatmap_path,
    }


def _production_segmentation(
    preprocessed_image: np.ndarray,
    inference_result: dict,
    image_path: str,
) -> dict:
    """
    Segmentation en mode PRODUCTION_READY.

    Utilise la carte Grad-CAM produite par EfficientNet pour générer
    un masque de segmentation et une heatmap réels.

    Si aucune CAM n'est disponible (erreur modèle), fallback sur DEMO.

    Args:
        preprocessed_image: Tableau NumPy (1, H, W)
        inference_result: Résultat d'inférence (contient gradcam_map)
        image_path: Chemin de l'image originale

    Returns:
        Dictionnaire avec segmentation_mask_path et heatmap_path
    """
    gradcam_map = inference_result.get("gradcam_map")

    if gradcam_map is None:
        logger.warning("[Segmentation PROD] Pas de Grad-CAM — fallback DEMO.")
        return _demo_segmentation(preprocessed_image, inference_result, image_path)

    logger.info("[Segmentation PROD] Génération du masque depuis Grad-CAM.")

    import cv2
    import uuid

    # --- 1. Charger l'image originale ---
    orig = Image.open(image_path).convert("RGB")
    W, H = orig.size
    orig_arr = np.array(orig, dtype=np.uint8)

    # --- 2. Upscale de la CAM vers la taille de l'image ---
    cam_upscaled = cv2.resize(
        gradcam_map.astype(np.float32),
        (W, H),
        interpolation=cv2.INTER_CUBIC,
    )
    cam_upscaled = np.clip(cam_upscaled, 0, 1)

    # --- 3. Masque binaire (seuil Otsu sur la CAM) ---
    cam_uint8 = (cam_upscaled * 255).astype(np.uint8)
    _, mask_binary = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphologie pour nettoyer le masque
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

    # --- 4. Heatmap colorée (JET colormap) ---
    heatmap_jet = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

    # Overlay sur l'image originale (60% original + 40% heatmap)
    overlay = (0.6 * orig_arr + 0.4 * heatmap_jet).astype(np.uint8)

    # Dessiner le contour du masque en vert
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 100), 2)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # --- 5. Sauvegarde ---
    uid = str(uuid.uuid4())[:8]
    mask_path = os.path.join(settings.MASKS_DIR, f"mask_{uid}.png")
    heatmap_path = os.path.join(settings.HEATMAPS_DIR, f"heatmap_{uid}.png")

    Image.fromarray(mask_binary, mode="L").save(mask_path)
    Image.fromarray(overlay).save(heatmap_path)

    logger.info(f"[Segmentation PROD] Masque CAM : {mask_path}")
    logger.info(f"[Segmentation PROD] Heatmap CAM : {heatmap_path}")

    return {
        "segmentation_mask_path": mask_path,
        "heatmap_path": heatmap_path,
    }

    # =========================================================
    # POINT D'INTÉGRATION MedSAM (si vous avez les poids) :
    #
    # from segment_anything import sam_model_registry, SamPredictor
    # sam = sam_model_registry["vit_b"](checkpoint="models/medsam_vit_b.pth")
    # predictor = SamPredictor(sam)
    # predictor.set_image(orig_arr)
    #
    # # Utiliser le bounding box de la CAM comme prompt
    # ys, xs = np.where(mask_binary > 0)
    # if len(xs) > 0:
    #     bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    #     masks, _, _ = predictor.predict(box=bbox[None])
    #     mask_binary = (masks[0] * 255).astype(np.uint8)
    # =========================================================
