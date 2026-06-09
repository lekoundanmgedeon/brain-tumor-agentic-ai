"""
Agent Radiologue — Vision Agent
================================
Reçoit l'image IRM, la prétraite, détecte la tumeur,
génère la segmentation et retourne un résultat structuré JSON.
"""

import logging
from workflow.state import BrainTumorState
from services.preprocessing import preprocess_image
from services.inference import run_inference
from services.segmentation import generate_segmentation

logger = logging.getLogger(__name__)


def radiology_agent(state: BrainTumorState) -> BrainTumorState:
    """
    Agent Radiologue (Vision Agent).

    Pipeline :
    1. Prétraitement de l'image
    2. Inférence (détection + classification)
    3. Segmentation (masque + heatmap)
    4. Mise à jour de l'état global

    Args:
        state: État LangGraph courant

    Returns:
        État mis à jour avec les résultats d'analyse
    """
    logger.info("[Radiologue] Démarrage de l'analyse IRM.")
    image_path = state.get("image_path", "")

    try:
        # --- 1. Prétraitement ---
        logger.info("[Radiologue] Prétraitement de l'image...")
        preprocessed = preprocess_image(image_path)

        # --- 2. Inférence ---
        logger.info("[Radiologue] Lancement de l'inférence...")
        inference_result = run_inference(preprocessed, image_path)

        # --- 3. Segmentation ---
        logger.info("[Radiologue] Génération de la segmentation...")
        seg_result = generate_segmentation(preprocessed, inference_result, image_path)

        # --- 4. Agrégation ---
        updated_state = {
            **state,
            "tumor_detected": inference_result.get("tumor_detected", False),
            "suspected_tumor_type": inference_result.get("suspected_tumor_type", "unknown"),
            "confidence": inference_result.get("confidence", 0.0),
            "segmentation_mask_path": seg_result.get("segmentation_mask_path"),
            "heatmap_path": seg_result.get("heatmap_path"),
            "tumor_location": inference_result.get("tumor_location"),
            "tumor_area_mm2": inference_result.get("tumor_area_mm2"),
            "tumor_volume_mm3": inference_result.get("tumor_volume_mm3"),
            "technical_findings": inference_result.get("technical_findings", []),
        }

        logger.info(
            f"[Radiologue] Analyse terminée. "
            f"Tumeur détectée : {inference_result.get('tumor_detected')}. "
            f"Type suspecté : {inference_result.get('suspected_tumor_type')}. "
            f"Confiance : {inference_result.get('confidence'):.2f}."
        )

        return updated_state

    except Exception as e:
        logger.error(f"[Radiologue] Erreur lors de l'analyse : {e}", exc_info=True)
        return {
            **state,
            "tumor_detected": None,
            "suspected_tumor_type": "unknown",
            "confidence": 0.0,
            "segmentation_mask_path": None,
            "heatmap_path": None,
            "tumor_location": None,
            "tumor_area_mm2": None,
            "tumor_volume_mm3": None,
            "technical_findings": [f"Erreur lors de l'analyse : {str(e)}"],
        }
