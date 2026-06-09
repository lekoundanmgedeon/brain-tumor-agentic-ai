"""
Agent Rapport — Gemini / Mistral / Fallback
============================================
Génère un rapport médical complet, prudent et structuré
en combinant les résultats d'analyse et le contexte RAG.
"""

import logging
from datetime import datetime

from workflow.state import BrainTumorState
from services.llm_client import generate_report

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Tu es un assistant médical IA spécialisé en neuroradiologie, conçu pour un usage académique.

RÈGLES ABSOLUES :
1. Tu ne poses JAMAIS de diagnostic médical définitif.
2. Tu utilises toujours des formulations prudentes : "suggère", "compatible avec", "possible", "à confirmer".
3. Tu rappelles TOUJOURS que l'analyse IA ne remplace pas un professionnel de santé.
4. Tu cites les sources utilisées.
5. Tu structures le rapport de façon claire et lisible.
6. Tu n'inventes JAMAIS de données médicales.
7. Si les données sont insuffisantes, tu le mentionnes explicitement.
8. Tu ne dis JAMAIS "Le patient a [tumeur]" mais "L'analyse suggère une anomalie compatible avec [tumeur]".
""".strip()


def report_agent(state: BrainTumorState) -> BrainTumorState:
    """
    Agent de génération du rapport final.

    Construit un prompt complet à partir de tous les résultats,
    l'envoie au LLM configuré, et retourne le rapport dans l'état.

    Args:
        state: État LangGraph courant

    Returns:
        État mis à jour avec final_report et llm_provider_used
    """
    logger.info("[Rapport] Génération du rapport final.")

    # --- 1. Vérifier si l'image était invalide ---
    if not state.get("image_valid", True):
        report = _build_invalid_image_report(state)
        return {
            **state,
            "final_report": report,
            "final_status": "IMAGE_INVALIDE",
            "llm_provider_used": "FALLBACK",
        }

    # --- 2. Construire le prompt médical ---
    prompt = _build_medical_prompt(state)

    # --- 3. Appeler le LLM ---
    try:
        report_text, provider = generate_report(prompt, system_prompt=SYSTEM_PROMPT)
        logger.info(f"[Rapport] Rapport généré via {provider}.")
    except Exception as e:
        logger.error(f"[Rapport] Erreur lors de la génération : {e}", exc_info=True)
        report_text = _build_error_report(state, str(e))
        provider = "FALLBACK"

    # --- 4. Ajouter l'en-tête et le pied de page standard ---
    report_text = _add_standard_disclaimer(report_text, state)

    return {
        **state,
        "final_report": report_text,
        "final_status": "ANALYSE_COMPLETE",
        "llm_provider_used": provider,
    }


def _build_medical_prompt(state: BrainTumorState) -> str:
    """
    Construit le prompt médical complet pour le LLM.

    Args:
        state: État LangGraph avec tous les résultats

    Returns:
        Prompt formaté
    """
    tumor_detected = state.get("tumor_detected", False)
    tumor_type = state.get("suspected_tumor_type", "unknown") or "unknown"
    confidence = state.get("confidence", 0.0) or 0.0
    confidence_level = state.get("confidence_level", "low") or "low"
    location = state.get("tumor_location") or "non déterminée"
    area = state.get("tumor_area_mm2")
    findings = state.get("technical_findings", [])
    warnings = state.get("validation_warnings", [])
    medical_context = state.get("medical_context_summary") or "Aucun contexte médical disponible."
    sources = state.get("sources", [])
    patient_context = state.get("patient_context") or {}
    result_valid = state.get("result_valid", False)
    quality_score = state.get("image_quality_score", 0.0)

    # Formatage des sources
    sources_text = (
        "\n".join(f"- {s}" for s in sources)
        if sources
        else "- Base de connaissances interne uniquement"
    )

    # Formatage du contexte patient
    patient_text = ""
    if patient_context:
        patient_text = "\n".join(f"- {k}: {v}" for k, v in patient_context.items())
    else:
        patient_text = "- Non fourni (analyse anonyme)"

    # Formatage des avertissements
    warnings_text = (
        "\n".join(f"- ⚠️ {w}" for w in warnings)
        if warnings
        else "- Aucun avertissement"
    )

    area_text = f"{area:.1f} mm²" if area else "Non calculée"
    confidence_pct = f"{confidence:.1%}"

    prompt = f"""
Tu dois générer un rapport médical académique complet, prudent et structuré en français,
basé sur les données suivantes issues d'une analyse automatisée par IA d'une image IRM cérébrale.

=== RÉSULTATS DE L'ANALYSE IA ===

Date d'analyse : {datetime.now().strftime("%d/%m/%Y à %H:%M")}
Qualité de l'image : {quality_score:.2f}/1.00
Résultat validé : {"OUI" if result_valid else "NON — résultat incertain"}

Tumeur détectée (IA) : {"OUI" if tumor_detected else "NON"}
Type suspecté : {tumor_type.replace('_', ' ').upper() if tumor_detected else "N/A"}
Score de confiance : {confidence_pct} ({confidence_level})
Localisation approximative : {location}
Surface estimée : {area_text}

Observations techniques :
{chr(10).join(f"- {f}" for f in findings) if findings else "- Aucune"}

Avertissements de validation :
{warnings_text}

=== CONTEXTE MÉDICAL (RAG) ===

{medical_context}

=== CONTEXTE PATIENT ===

{patient_text}

=== SOURCES UTILISÉES ===

{sources_text}

=== INSTRUCTIONS POUR LE RAPPORT ===

Génère un rapport structuré avec les sections suivantes :
1. AVERTISSEMENT MÉDICAL (toujours en premier et en évidence)
2. Résumé de l'Analyse
3. Observations Radiologiques (prudentes, jamais affirmatives)
4. Contexte Médical Général (basé sur les données RAG)
5. Recommandations Générales (consultation médicale obligatoire)
6. Limites de l'Analyse IA
7. Sources Utilisées

RAPPEL : 
- Ne jamais dire "Le patient a [tumeur]"
- Toujours dire "L'analyse IA suggère..." ou "Compatible avec..."
- Toujours recommander une consultation médicale
- Si confiance basse ou résultat non valide, être particulièrement prudent
""".strip()

    return prompt


def _build_invalid_image_report(state: BrainTumorState) -> str:
    """
    Rapport spécial pour une image invalide.

    Args:
        state: État avec les erreurs d'image

    Returns:
        Rapport d'image invalide
    """
    errors = state.get("image_errors", ["Erreur inconnue"])
    errors_text = "\n".join(f"- {e}" for e in errors)

    return f"""
# Rapport d'Analyse IRM — Image Non Valide

---

## ⚕️ AVERTISSEMENT MÉDICAL

Ce système est un outil académique d'assistance à l'analyse. Il ne remplace pas un professionnel de santé.

---

## Résultat de la Validation

**L'image soumise n'a pas pu être analysée** car elle ne satisfait pas les critères
de qualité ou de format requis.

### Erreurs détectées :

{errors_text}

---

## Actions Recommandées

1. Vérifiez que le fichier est une image IRM valide (format PNG, JPG ou JPEG).
2. Assurez-vous que l'image n'est pas corrompue.
3. Vérifiez que l'image a une résolution suffisante (minimum 64×64 pixels).
4. Soumettez une nouvelle image de meilleure qualité.

---

*Aucune analyse médicale n'a été effectuée sur cette image.*
""".strip()


def _build_error_report(state: BrainTumorState, error_msg: str) -> str:
    """
    Rapport de secours en cas d'erreur LLM.

    Args:
        state: État courant
        error_msg: Message d'erreur

    Returns:
        Rapport d'erreur basique
    """
    tumor_detected = state.get("tumor_detected", False)
    tumor_type = state.get("suspected_tumor_type", "unknown") or "unknown"
    confidence = state.get("confidence", 0.0) or 0.0

    return f"""
# Rapport d'Analyse IRM — Mode Dégradé

---

## ⚕️ AVERTISSEMENT MÉDICAL

Ce rapport est généré par un système d'IA académique. Il ne remplace pas un professionnel de santé.

---

## Résumé

Une erreur est survenue lors de la génération du rapport complet ({error_msg}).
Voici les données brutes de l'analyse :

- **Anomalie détectée (IA)** : {"Oui" if tumor_detected else "Non"}
- **Type suspecté** : {tumor_type.replace('_', ' ')} (à confirmer par un professionnel)
- **Confiance** : {confidence:.1%}

---

## IMPORTANT

Ces données brutes ne constituent pas un diagnostic. Consultez un médecin spécialiste.

*Rapport généré en mode dégradé suite à une erreur technique.*
""".strip()


def _add_standard_disclaimer(report: str, state: BrainTumorState) -> str:
    """
    Ajoute un pied de page standard au rapport.

    Args:
        report: Rapport généré
        state: État pour les métadonnées

    Returns:
        Rapport avec pied de page
    """
    provider = state.get("llm_provider_used", "FALLBACK")
    app_mode = state.get("final_status", "")

    footer = f"""

---

---

*Ce rapport a été généré automatiquement par le système Brain Tumor Agentic AI (version académique).*
*LLM utilisé : {provider} | Date : {datetime.now().strftime("%d/%m/%Y %H:%M")}*
*⚠️ CE RAPPORT N'EST PAS UN DOCUMENT MÉDICAL CERTIFIÉ. CONSULTEZ UN PROFESSIONNEL DE SANTÉ.*
"""
    return report + footer
