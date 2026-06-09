"""
Agent Validation Résultats
===========================
Vérifie la cohérence et la fiabilité des résultats produits
par l'Agent Radiologue avant de déclencher le RAG et le rapport.
"""

import os
import logging

from workflow.state import BrainTumorState
from config.settings import settings

logger = logging.getLogger(__name__)


def validate_results_agent(state: BrainTumorState) -> BrainTumorState:
    """
    Agent de validation des résultats d'inférence.

    Vérifie :
    - Score de confiance (seuil minimal)
    - Masque de segmentation non vide
    - Taille tumorale cohérente
    - Type de tumeur dans les classes valides
    - Cohérence globale du résultat

    Détermine :
    - Si le RAG peut être appelé
    - Si un rapport affirmatif peut être généré
    - Le niveau de confiance : "high", "medium", "low"

    Args:
        state: État LangGraph courant

    Returns:
        État mis à jour avec les champs de validation
    """
    logger.info("[ValidationRésultats] Validation des résultats d'inférence.")
    warnings = []

    confidence = state.get("confidence", 0.0) or 0.0
    tumor_detected = state.get("tumor_detected", None)
    tumor_type = state.get("suspected_tumor_type", "unknown")
    mask_path = state.get("segmentation_mask_path")
    area = state.get("tumor_area_mm2")

    # --- 1. Vérification du type de tumeur ---
    if tumor_type not in settings.VALID_TUMOR_TYPES:
        warnings.append(
            f"Type de tumeur non reconnu : '{tumor_type}'. "
            f"Valeurs acceptées : {settings.VALID_TUMOR_TYPES}."
        )

    # --- 2. Vérification de la confiance ---
    if confidence < settings.MIN_CONFIDENCE_THRESHOLD:
        warnings.append(
            f"Score de confiance trop faible : {confidence:.2f} "
            f"(seuil minimum : {settings.MIN_CONFIDENCE_THRESHOLD:.2f}). "
            "Le résultat est incertain."
        )

    # --- 3. Vérification du masque de segmentation ---
    if tumor_detected and mask_path:
        if not os.path.exists(mask_path):
            warnings.append(f"Masque de segmentation introuvable : '{mask_path}'.")
        else:
            try:
                from PIL import Image
                import numpy as np
                mask_img = Image.open(mask_path).convert("L")
                mask_arr = np.array(mask_img)
                non_zero_ratio = float(np.sum(mask_arr > 0)) / mask_arr.size
                if non_zero_ratio < 0.001:
                    warnings.append(
                        f"Masque de segmentation vide ou quasi-vide "
                        f"(ratio non-zero : {non_zero_ratio:.4f})."
                    )
                elif non_zero_ratio > 0.8:
                    warnings.append(
                        f"Masque de segmentation trop large "
                        f"(ratio non-zero : {non_zero_ratio:.2%}). "
                        "Possible faux positif."
                    )
            except Exception as e:
                warnings.append(f"Impossible de vérifier le masque : {e}")
    elif tumor_detected and not mask_path:
        warnings.append("Tumeur détectée mais aucun masque de segmentation disponible.")

    # --- 4. Vérification de la surface tumorale ---
    if tumor_detected and area is not None:
        if area < 1.0:
            warnings.append(
                f"Surface tumorale anormalement petite : {area:.1f} mm². "
                "Possible artefact."
            )
        elif area > 10000.0:
            warnings.append(
                f"Surface tumorale anormalement grande : {area:.1f} mm². "
                "Vérification recommandée."
            )

    # --- 5. Cas : tumor_detected est None (erreur d'inférence) ---
    if tumor_detected is None:
        warnings.append(
            "L'inférence n'a pas produit de résultat valide (tumor_detected=None). "
            "Erreur possible dans le pipeline de détection."
        )

    # --- 6. Calcul du niveau de confiance ---
    confidence_level = _compute_confidence_level(confidence, warnings)

    # --- 7. Décision : peut-on appeler RAG et générer un rapport ? ---
    can_call_rag = (
        tumor_detected is not None
        and tumor_type != "unknown"
        and len(warnings) < 3  # Trop d'avertissements → prudence
    )

    can_generate_report = tumor_detected is not None  # Toujours générer un rapport

    result_valid = len(warnings) == 0 and tumor_detected is not None

    if result_valid:
        logger.info(f"[ValidationRésultats] Résultat valide. Confiance : {confidence_level}.")
    else:
        logger.warning(
            f"[ValidationRésultats] Résultat avec avertissements : {warnings}. "
            f"Confiance : {confidence_level}."
        )

    return {
        **state,
        "result_valid": result_valid,
        "confidence_level": confidence_level,
        "validation_warnings": warnings,
        "can_call_rag": can_call_rag,
        "can_generate_report": can_generate_report,
    }


def _compute_confidence_level(confidence: float, warnings: list) -> str:
    """
    Détermine le niveau de confiance textuel.

    Args:
        confidence: Score numérique [0, 1]
        warnings: Liste d'avertissements

    Returns:
        "high", "medium" ou "low"
    """
    if warnings:
        # Les avertissements dégradent le niveau
        penalty = min(len(warnings) * 0.1, 0.3)
        effective_confidence = confidence - penalty
    else:
        effective_confidence = confidence

    if effective_confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif effective_confidence >= settings.MIN_CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"
