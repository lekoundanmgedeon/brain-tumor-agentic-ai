"""
Service LLM Client
==================
Interface unifiée pour Gemini (SDK google-genai v2), Mistral et le mode mock/fallback.
Détermine automatiquement le provider disponible.
"""

import logging
from config.settings import settings

logger = logging.getLogger(__name__)


def generate_report(prompt: str, system_prompt: str = "") -> tuple[str, str]:
    """
    Génère un rapport médical via le LLM configuré.

    Sélection automatique :
    1. Gemini si GEMINI_API_KEY disponible
    2. Mistral si MISTRAL_API_KEY disponible
    3. Mode mock/fallback sinon

    Args:
        prompt: Prompt utilisateur avec le contexte médical
        system_prompt: Instruction système pour le LLM

    Returns:
        Tuple (rapport_généré, provider_utilisé)
    """
    provider = settings.get_llm_provider()
    logger.info(f"[LLM Client] Provider sélectionné : {provider}")

    if provider == "gemini":
        return _call_gemini(prompt, system_prompt), "GEMINI"
    elif provider == "mistral":
        return _call_mistral(prompt, system_prompt), "MISTRAL"
    else:
        return _mock_report(prompt), "FALLBACK"


def _call_gemini(prompt: str, system_prompt: str = "") -> str:
    """
    Appelle l'API Google Gemini via le SDK google-genai (v2+).
    Essaie automatiquement plusieurs modèles en cas de quota épuisé (429).

    Ordre de tentative :
      1. GEMINI_MODEL configuré dans .env
      2. gemini-1.5-flash
      3. gemini-2.0-flash
      4. gemini-1.5-pro
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("[Gemini] 'google-genai' non installé → pip install google-genai")
        return _mock_report(prompt)

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # Liste de modèles à essayer dans l'ordre
    candidates = [settings.GEMINI_MODEL]
    for m in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
        if m not in candidates:
            candidates.append(m)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        temperature=0.3,
        max_output_tokens=2048,
    )

    for model in candidates:
        try:
            logger.info(f"[Gemini] Tentative avec {model}...")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = response.text
            logger.info(f"[Gemini] ✅ Rapport généré ({len(text)} car.) via {model}.")
            return text

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                logger.warning(f"[Gemini] Quota épuisé sur {model} — essai suivant...")
                continue
            elif "404" in err or "not found" in err.lower():
                logger.warning(f"[Gemini] Modèle {model} non disponible — essai suivant...")
                continue
            else:
                logger.error(f"[Gemini] Erreur sur {model} : {err[:200]}")
                break

    # Tous les modèles ont échoué → fallback legacy SDK puis mock
    logger.warning("[Gemini] Tous les modèles ont échoué → tentative SDK legacy...")
    return _call_gemini_legacy(prompt, system_prompt)


def _call_gemini_legacy(prompt: str, system_prompt: str = "") -> str:
    """
    Fallback vers l'ancien SDK google-generativeai (si google-genai indisponible).

    Args:
        prompt: Prompt utilisateur
        system_prompt: Instruction système

    Returns:
        Texte généré ou rapport mock
    """
    try:
        import google.generativeai as genai
        import warnings
        warnings.filterwarnings("ignore")   # Masquer le FutureWarning du SDK déprécié

        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            system_instruction=system_prompt if system_prompt else None,
        )
        response = model.generate_content(prompt)
        text = response.text
        logger.info(f"[Gemini Legacy] Rapport généré ({len(text)} car.).")
        return text

    except Exception as e:
        logger.error(f"[Gemini Legacy] Erreur : {e}")
        return _mock_report(prompt)


def _call_mistral(prompt: str, system_prompt: str = "") -> str:
    """
    Appelle l'API Mistral AI pour générer le rapport.
    Compatible avec le SDK mistralai v2+ (import depuis mistralai.client).

    Args:
        prompt: Prompt utilisateur
        system_prompt: Instruction système

    Returns:
        Texte du rapport généré
    """
    try:
        # SDK mistralai v2+ (mistralai >= 1.0)
        try:
            from mistralai.client import Mistral
        except ImportError:
            from mistralai import Mistral  # fallback ancien SDK

        client = Mistral(api_key=settings.MISTRAL_API_KEY)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.complete(
            model=settings.MISTRAL_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
        text = response.choices[0].message.content
        logger.info(f"[Mistral] Rapport généré ({len(text)} caractères) via {settings.MISTRAL_MODEL}.")
        return text

    except ImportError:
        logger.warning("[Mistral] SDK 'mistralai' non installé. pip install mistralai")
        return _mock_report(prompt)
    except Exception as e:
        logger.error(f"[Mistral] Erreur API : {e}", exc_info=True)
        return _mock_report(prompt)


def _mock_report(prompt: str) -> str:
    """
    Génère un rapport de démonstration sans appel LLM.
    Utilisé quand aucune clé API n'est disponible.
    """
    logger.info("[Mock LLM] Génération du rapport en mode FALLBACK.")

    tumor_detected = "oui" in prompt.lower() and "tumeur détectée" not in prompt.lower()
    tumor_detected = "tumor_detected: true" in prompt.lower() or "anomalie suspecte" in prompt.lower() or "OUI" in prompt

    has_glioma     = "glioma" in prompt.lower()
    has_meningioma = "meningioma" in prompt.lower()
    has_pituitary  = "pituitary" in prompt.lower()

    if has_glioma:
        tumor_label = "gliome"
    elif has_meningioma:
        tumor_label = "méningiome"
    elif has_pituitary:
        tumor_label = "adénome hypophysaire"
    else:
        tumor_label = "anomalie cérébrale"

    if tumor_detected:
        return f"""# Rapport d'Analyse IRM — Mode Fallback

## ⚕️ AVERTISSEMENT MÉDICAL IMPORTANT

Ce rapport est généré par un système d'IA à des fins **académiques uniquement**.
Il ne constitue **en aucun cas** un diagnostic médical définitif et ne remplace pas
l'avis d'un radiologue, neurologue ou professionnel de santé qualifié.

---

## Résumé

L'analyse automatisée suggère la présence d'une **anomalie compatible avec un {tumor_label}**.

> ⚠️ Ce résultat doit impérativement être confirmé par un professionnel de santé.

---

## Observations Radiologiques (IA)

- **Type d'anomalie suspecté** : compatible avec un {tumor_label} (analyse IA — non certifiée)
- **Localisation** : voir métriques techniques
- **Niveau de confiance** : voir score affiché ci-dessus

---

## Recommandations Générales

1. **Consultation spécialisée** : Consultez un neurologue ou neurochirurgien.
2. **Imagerie complémentaire** : IRM multiséquences avec injection de gadolinium recommandée.
3. **Bilan complet** : bilan neurologique et biologique indiqué.
4. **Ne pas différer** : si des symptômes sont présents, consultez rapidement.

---

## Limites

- Prototype académique, **non certifié médicalement**
- Analyse 2D — une IRM 3D complète est nécessaire cliniquement
- Faux positifs et faux négatifs possibles

*Rapport généré en mode FALLBACK. Configurez GEMINI_API_KEY ou MISTRAL_API_KEY pour un rapport LLM.*""".strip()

    return """# Rapport d'Analyse IRM — Mode Fallback

## ⚕️ AVERTISSEMENT MÉDICAL IMPORTANT

Ce rapport est généré par un système d'IA à des fins **académiques uniquement**.

---

## Résumé

L'analyse automatisée **n'a détecté aucune anomalie focale suspecte** sur cette image IRM.

> ℹ️ Un résultat négatif ne garantit pas l'absence de pathologie.
> Seul un radiologue qualifié peut interpréter correctement une IRM cérébrale.

---

## Recommandations

1. Faites interpréter vos IRM par un radiologue qualifié.
2. En cas de symptômes persistants, consultez votre médecin.
3. Ce résultat ne constitue pas un bilan de santé.

*Rapport généré en mode FALLBACK. Configurez GEMINI_API_KEY pour un rapport plus détaillé.*""".strip()