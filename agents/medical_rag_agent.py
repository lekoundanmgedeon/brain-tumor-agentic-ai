"""
Agent RAG Médical
==================
Construit une requête médicale, interroge PubMed et/ou la base locale,
synthétise le contexte médical et retourne les sources utilisées.
"""

import logging
from workflow.state import BrainTumorState
from services.rag_pubmed import search_pubmed
from services.rag_nci import search_nci_and_local

logger = logging.getLogger(__name__)


def medical_rag_agent(state: BrainTumorState) -> BrainTumorState:
    """
    Agent RAG Médical.

    Pipeline :
    1. Construction de la requête médicale
    2. Interrogation PubMed (si activé)
    3. Interrogation NCI + fichiers locaux
    4. Déduplication et sélection des passages pertinents
    5. Synthèse du contexte médical
    6. Mise à jour de l'état

    Args:
        state: État LangGraph courant

    Returns:
        État mis à jour avec les documents récupérés et le résumé
    """
    logger.info("[RAG Médical] Démarrage de la recherche documentaire.")

    tumor_type = state.get("suspected_tumor_type", "unknown") or "unknown"
    can_call_rag = state.get("can_call_rag", True)

    if not can_call_rag:
        logger.warning(
            "[RAG Médical] RAG désactivé par la validation (résultat trop incertain). "
            "Utilisation du contexte générique."
        )

    # --- 1. Construction de la requête ---
    rag_query = _build_rag_query(tumor_type)
    logger.info(f"[RAG Médical] Requête construite : '{rag_query}'")

    all_documents = []
    all_sources = []

    # --- 2. Recherche PubMed ---
    try:
        pubmed_docs = search_pubmed(tumor_type, max_results=3)
        all_documents.extend(pubmed_docs)
        for doc in pubmed_docs:
            source_entry = doc.get("url") or doc.get("source", "PubMed")
            if source_entry not in all_sources:
                all_sources.append(source_entry)
        logger.info(f"[RAG Médical] PubMed : {len(pubmed_docs)} document(s).")
    except Exception as e:
        logger.warning(f"[RAG Médical] Erreur PubMed : {e}")

    # --- 3. Recherche NCI + Local ---
    try:
        nci_docs = search_nci_and_local(tumor_type)
        all_documents.extend(nci_docs)
        for doc in nci_docs:
            source_entry = doc.get("url") or doc.get("source", "Base locale")
            if source_entry and source_entry not in all_sources:
                all_sources.append(source_entry)
        logger.info(f"[RAG Médical] NCI/Local : {len(nci_docs)} document(s).")
    except Exception as e:
        logger.warning(f"[RAG Médical] Erreur NCI/Local : {e}")

    # --- 4. Synthèse du contexte ---
    medical_summary = _synthesize_context(all_documents, tumor_type)

    logger.info(
        f"[RAG Médical] Terminé. {len(all_documents)} document(s) total, "
        f"{len(all_sources)} source(s)."
    )

    return {
        **state,
        "rag_query": rag_query,
        "retrieved_documents": all_documents,
        "medical_context_summary": medical_summary,
        "sources": all_sources,
    }


def _build_rag_query(tumor_type: str) -> str:
    """
    Construit une requête médicale structurée pour le RAG.

    Args:
        tumor_type: Type de tumeur suspecté

    Returns:
        Requête en langage naturel
    """
    type_labels = {
        "glioma": "gliome cérébral (glioma)",
        "meningioma": "méningiome (meningioma)",
        "pituitary_tumor": "adénome hypophysaire (pituitary adenoma)",
        "no_tumor": "absence de tumeur cérébrale",
        "unknown": "tumeur cérébrale de type inconnu",
    }
    label = type_labels.get(tumor_type, tumor_type)
    return (
        f"Informations médicales générales sur {label} : "
        "protocoles de traitement actuels, signes radiologiques, pronostic, "
        "recommandations cliniques, limites diagnostiques IRM."
    )


def _synthesize_context(documents: list, tumor_type: str) -> str:
    """
    Synthétise le contexte médical depuis les documents récupérés.

    Prend les N premiers passages pertinents et les concatène
    en un résumé structuré pour le LLM de rapport.

    Args:
        documents: Liste de documents récupérés
        tumor_type: Type de tumeur pour contextualiser

    Returns:
        Texte de synthèse médicale
    """
    if not documents:
        return (
            f"Aucune information médicale spécifique récupérée pour '{tumor_type}'. "
            "Les informations générales sur les tumeurs cérébrales indiquent que "
            "tout diagnostic doit être confirmé par un professionnel de santé qualifié."
        )

    parts = []
    seen_content = set()

    for doc in documents[:5]:  # Limiter à 5 documents
        content = doc.get("content") or doc.get("abstract") or ""
        title = doc.get("title", "")

        # Déduplication basique
        content_key = content[:50]
        if content_key in seen_content or not content:
            continue
        seen_content.add(content_key)

        parts.append(f"**{title}**\n{content}")

    if not parts:
        return "Contexte médical non disponible."

    summary = "\n\n".join(parts)

    # Limiter la longueur totale pour le LLM
    if len(summary) > 3000:
        summary = summary[:3000] + "\n[...contexte tronqué pour la génération du rapport]"

    return summary
