"""
Workflow LangGraph — Brain Tumor Agentic AI
============================================
Définit le graphe d'agents avec les transitions conditionnelles.

Flux principal :
  START
    → validate_image
      → [si valide]   → radiology_agent
                          → validate_results
                            → [si fiable]  → medical_rag → report_generator → END
                            → [si incertain]              → report_generator → END
      → [si invalide]                                     → report_generator → END
"""

import logging
from langgraph.graph import StateGraph, END

from workflow.state import BrainTumorState
from agents.validation_image_agent import validate_image_agent
from agents.radiology_agent import radiology_agent
from agents.validation_results_agent import validate_results_agent
from agents.medical_rag_agent import medical_rag_agent
from agents.report_agent import report_agent
from agents.orchestrator_agent import (
    should_continue_after_image_validation,
    should_continue_after_validation_results,
)

logger = logging.getLogger(__name__)


def build_brain_tumor_graph() -> StateGraph:
    """
    Construit et compile le graphe LangGraph pour l'analyse de tumeurs cérébrales.

    Returns:
        Graphe LangGraph compilé et prêt à être exécuté
    """
    logger.info("[Graph] Construction du graphe LangGraph.")

    # --- Initialisation du graphe ---
    builder = StateGraph(BrainTumorState)

    # --- Ajout des nœuds (agents) ---
    builder.add_node("validate_image", validate_image_agent)
    builder.add_node("radiologist", radiology_agent)
    builder.add_node("validate_results", validate_results_agent)
    builder.add_node("medical_rag", medical_rag_agent)
    builder.add_node("report_generator", report_agent)

    # --- Point d'entrée ---
    builder.set_entry_point("validate_image")

    # --- Transition conditionnelle après validation image ---
    builder.add_conditional_edges(
        "validate_image",
        should_continue_after_image_validation,
        {
            "radiologist": "radiologist",   # Image valide → Radiologue
            "report": "report_generator",    # Image invalide → Rapport direct
        },
    )

    # --- Transition simple : Radiologue → Validation Résultats ---
    builder.add_edge("radiologist", "validate_results")

    # --- Transition conditionnelle après validation des résultats ---
    builder.add_conditional_edges(
        "validate_results",
        should_continue_after_validation_results,
        {
            "rag": "medical_rag",            # Résultat fiable → RAG
            "report": "report_generator",    # Résultat incertain → Rapport direct
        },
    )

    # --- Transition simple : RAG → Rapport ---
    builder.add_edge("medical_rag", "report_generator")

    # --- Fin du workflow ---
    builder.add_edge("report_generator", END)

    # --- Compilation ---
    graph = builder.compile()
    logger.info("[Graph] Graphe LangGraph compilé avec succès.")
    return graph


def run_brain_tumor_analysis(image_path: str, patient_context: dict = None) -> BrainTumorState:
    """
    Point d'entrée principal pour lancer l'analyse complète.

    Initialise l'état, compile le graphe et l'exécute.

    Args:
        image_path: Chemin vers l'image IRM à analyser
        patient_context: Contexte patient optionnel

    Returns:
        État final avec tous les résultats de l'analyse
    """
    from agents.orchestrator_agent import orchestrator_agent

    logger.info(f"[Graph] Lancement de l'analyse pour : {image_path}")

    # Initialiser l'état via l'orchestrateur
    initial_state = orchestrator_agent(image_path, patient_context)

    # Construire et exécuter le graphe
    graph = build_brain_tumor_graph()

    # Exécution synchrone du workflow
    final_state = graph.invoke(initial_state)

    logger.info(
        f"[Graph] Analyse terminée. Statut : {final_state.get('final_status', 'INCONNU')}. "
        f"Provider LLM : {final_state.get('llm_provider_used', 'N/A')}."
    )

    return final_state
