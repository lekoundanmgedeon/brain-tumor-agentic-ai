"""
État global du workflow LangGraph pour l'analyse de tumeurs cérébrales.
Ce module définit la structure de données partagée entre tous les agents.
"""

from typing import TypedDict, Optional


class BrainTumorState(TypedDict):
    """
    État partagé entre tous les agents du workflow LangGraph.
    Chaque agent lit et/ou écrit dans cet état.
    """

    # --- Entrée ---
    image_path: str                        # Chemin vers l'image IRM uploadée
    patient_context: Optional[dict]        # Contexte patient facultatif

    # --- Agent Validation Image ---
    image_valid: bool                      # L'image est-elle valide ?
    image_quality_score: Optional[float]   # Score de qualité [0.0 - 1.0]
    image_errors: list                     # Liste des erreurs de validation

    # --- Agent Radiologue (Vision) ---
    tumor_detected: Optional[bool]         # Tumeur détectée ?
    suspected_tumor_type: Optional[str]    # Type de tumeur suspecté
    confidence: Optional[float]            # Score de confiance [0.0 - 1.0]
    segmentation_mask_path: Optional[str]  # Chemin du masque de segmentation
    heatmap_path: Optional[str]            # Chemin de la heatmap
    tumor_location: Optional[str]          # Localisation approximative
    tumor_area_mm2: Optional[float]        # Surface tumorale estimée (mm²)
    tumor_volume_mm3: Optional[float]      # Volume tumoral estimé (mm³, si 3D)
    technical_findings: list               # Observations techniques

    # --- Agent Validation Résultats ---
    result_valid: Optional[bool]           # Résultat valide ?
    confidence_level: Optional[str]        # "high", "medium", "low"
    validation_warnings: list              # Avertissements de validation
    can_call_rag: Optional[bool]           # Autorisation d'appeler le RAG
    can_generate_report: Optional[bool]    # Autorisation de générer le rapport

    # --- Agent RAG Médical ---
    rag_query: Optional[str]               # Requête construite pour le RAG
    retrieved_documents: list              # Documents récupérés
    medical_context_summary: Optional[str] # Résumé du contexte médical
    sources: list                          # Sources utilisées

    # --- Agent Rapport ---
    final_report: Optional[str]            # Rapport final généré
    final_status: str                      # Statut final du workflow
    llm_provider_used: Optional[str]       # LLM utilisé pour le rapport
