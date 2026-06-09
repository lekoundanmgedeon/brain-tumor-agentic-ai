"""
Agent Orchestrateur — LangGraph
================================
Initialise l'état global et contrôle le flux du workflow.
Les transitions conditionnelles sont définies dans workflow/graph.py.
"""

import logging
from datetime import datetime

from workflow.state import BrainTumorState

logger = logging.getLogger(__name__)


def orchestrator_agent(image_path: str, patient_context: dict = None) -> BrainTumorState:
    """
    Initialise l'état global du workflow LangGraph.

    C'est le point d'entrée de tout le pipeline. Il crée
    l'état initial avec toutes les valeurs par défaut.

    Args:
        image_path: Chemin vers l'image IRM uploadée
        patient_context: Contexte patient optionnel (âge, sexe, symptômes...)

    Returns:
        État initial BrainTumorState
    """
    logger.info(f"[Orchestrateur] Initialisation du workflow. Image : {image_path}")

    initial_state: BrainTumorState = {
        # --- Entrée ---
        "image_path": image_path,
        "patient_context": patient_context or {},

        # --- Validation image (valeurs initiales) ---
        "image_valid": False,
        "image_quality_score": None,
        "image_errors": [],

        # --- Radiologue (valeurs initiales) ---
        "tumor_detected": None,
        "suspected_tumor_type": None,
        "confidence": None,
        "segmentation_mask_path": None,
        "heatmap_path": None,
        "tumor_location": None,
        "tumor_area_mm2": None,
        "tumor_volume_mm3": None,
        "technical_findings": [],

        # --- Validation résultats (valeurs initiales) ---
        "result_valid": None,
        "confidence_level": None,
        "validation_warnings": [],
        "can_call_rag": True,
        "can_generate_report": True,

        # --- RAG (valeurs initiales) ---
        "rag_query": None,
        "retrieved_documents": [],
        "medical_context_summary": None,
        "sources": [],

        # --- Rapport (valeurs initiales) ---
        "final_report": None,
        "final_status": "EN_COURS",
        "llm_provider_used": None,
    }

    logger.info(f"[Orchestrateur] État initial créé. Timestamp : {datetime.now().isoformat()}")
    return initial_state


def should_continue_after_image_validation(state: BrainTumorState) -> str:
    """
    Fonction de routage conditionnel après la validation de l'image.

    Args:
        state: État courant

    Returns:
        "radiologist" si l'image est valide
        "report" si l'image est invalide (rapport d'erreur direct)
    """
    if state.get("image_valid", False):
        logger.info("[Orchestrateur] Image valide → route vers Radiologue.")
        return "radiologist"
    else:
        logger.warning("[Orchestrateur] Image invalide → route vers Rapport (erreur).")
        return "report"


def should_continue_after_validation_results(state: BrainTumorState) -> str:
    """
    Fonction de routage conditionnel après la validation des résultats.

    Args:
        state: État courant

    Returns:
        "rag" si les résultats sont suffisamment fiables
        "report" si les résultats sont trop incertains (skip RAG)
    """
    confidence = state.get("confidence", 0.0) or 0.0
    can_call_rag = state.get("can_call_rag", True)
    tumor_detected = state.get("tumor_detected", None)

    # Si tumeur non détectée, on peut quand même appeler le RAG (pour dire "pas de tumeur")
    # Si résultat totalement invalide, on skip le RAG
    if can_call_rag and tumor_detected is not None:
        logger.info("[Orchestrateur] Résultat valide → route vers RAG Médical.")
        return "rag"
    else:
        logger.warning("[Orchestrateur] Résultat incertain → route directe vers Rapport.")
        return "report"
