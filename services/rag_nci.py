"""
Service RAG — NCI (National Cancer Institute) & documents locaux.
Fournit des informations médicales fiables depuis la base de connaissances locale
ou depuis le NCI quand Internet est disponible.
"""

import os
import logging
from typing import List, Dict

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

NCI_BASE_URL = "https://www.cancer.gov/types/brain"
REQUEST_TIMEOUT = 8


def search_nci_and_local(tumor_type: str) -> List[Dict]:
    """
    Récupère des informations médicales depuis :
    1. Les fichiers locaux dans knowledge_base/
    2. Le NCI (si disponible et activé)

    Args:
        tumor_type: Type de tumeur suspecté

    Returns:
        Liste de documents {"title", "content", "source", "url"}
    """
    documents = []

    # --- 1. Toujours lire les fichiers locaux (ne nécessite pas Internet) ---
    local_docs = _load_local_knowledge(tumor_type)
    documents.extend(local_docs)

    # --- 2. Tenter le NCI si fallback activé ---
    if settings.ENABLE_NCI_FALLBACK and not documents:
        nci_docs = _fetch_nci_summary(tumor_type)
        documents.extend(nci_docs)

    if not documents:
        logger.warning(
            f"[NCI/Local] Aucune source disponible pour '{tumor_type}'. "
            "Utilisation du contexte générique."
        )
        documents.append(_get_generic_context(tumor_type))

    logger.info(f"[NCI/Local] {len(documents)} document(s) récupéré(s) pour '{tumor_type}'.")
    return documents


def _load_local_knowledge(tumor_type: str) -> List[Dict]:
    """
    Charge les fichiers Markdown de la knowledge_base locale.

    Args:
        tumor_type: Type de tumeur (doit correspondre à un fichier .md)

    Returns:
        Liste de documents locaux
    """
    kb_dir = settings.KNOWLEDGE_BASE_DIR
    documents = []

    # Mapper le type de tumeur vers le fichier correspondant
    file_map = {
        "glioma": "glioma.md",
        "meningioma": "meningioma.md",
        "pituitary_tumor": "pituitary_tumor.md",
        "no_tumor": None,
        "unknown": None,
    }

    filename = file_map.get(tumor_type)
    if filename is None:
        return documents

    filepath = os.path.join(kb_dir, filename)
    if not os.path.exists(filepath):
        logger.warning(f"[NCI/Local] Fichier local introuvable : {filepath}")
        return documents

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extraire les sections principales
        sections = _parse_markdown_sections(content)
        for section_title, section_content in sections.items():
            if section_content.strip():
                documents.append({
                    "title": f"{tumor_type.replace('_', ' ').title()} — {section_title}",
                    "content": section_content.strip()[:800],
                    "source": "Knowledge Base Locale",
                    "url": f"file://{os.path.abspath(filepath)}",
                })

    except Exception as e:
        logger.error(f"[NCI/Local] Erreur lecture fichier local : {e}")

    return documents


def _parse_markdown_sections(content: str) -> dict:
    """
    Découpe un fichier Markdown en sections basées sur les titres ## .

    Args:
        content: Contenu Markdown brut

    Returns:
        Dictionnaire {titre_section: contenu}
    """
    sections = {}
    current_title = "Général"
    current_content = []

    for line in content.split("\n"):
        if line.startswith("## "):
            if current_content:
                sections[current_title] = "\n".join(current_content)
            current_title = line[3:].strip()
            current_content = []
        elif not line.startswith("# "):  # Ignorer le titre principal
            current_content.append(line)

    if current_content:
        sections[current_title] = "\n".join(current_content)

    return sections


def _fetch_nci_summary(tumor_type: str) -> List[Dict]:
    """
    Tente de récupérer une page du NCI (National Cancer Institute).

    Note : Le NCI ne fournit pas d'API directe, donc nous utilisons
    une requête HTTP simple vers les pages informatives.

    Args:
        tumor_type: Type de tumeur

    Returns:
        Liste de documents NCI
    """
    nci_urls = {
        "glioma": "https://www.cancer.gov/types/brain/patient/adult-brain-treatment-pdq",
        "meningioma": "https://www.cancer.gov/types/brain/patient/meningioma-treatment-pdq",
        "pituitary_tumor": "https://www.cancer.gov/types/pituitary/patient/pituitary-treatment-pdq",
    }

    url = nci_urls.get(tumor_type)
    if not url:
        return []

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            "User-Agent": "BrainTumorAcademicProject/1.0 (educational use)"
        })
        if response.status_code == 200:
            # Extraction basique du texte (sans BeautifulSoup pour minimiser les dépendances)
            text = response.text
            # Extraire un extrait du contenu brut
            start_idx = text.find("<main")
            end_idx = text.find("</main>", start_idx)
            if start_idx > 0 and end_idx > 0:
                raw_content = text[start_idx:end_idx]
                # Nettoyage basique des balises HTML
                import re
                clean_content = re.sub(r"<[^>]+>", " ", raw_content)
                clean_content = re.sub(r"\s+", " ", clean_content).strip()[:600]
                return [{
                    "title": f"NCI — {tumor_type.replace('_', ' ').title()}",
                    "content": clean_content,
                    "source": "National Cancer Institute",
                    "url": url,
                }]
    except Exception as e:
        logger.warning(f"[NCI] Impossible d'accéder au NCI : {e}")

    return []


def _get_generic_context(tumor_type: str) -> Dict:
    """
    Retourne un contexte médical générique en cas d'absence de sources.

    Args:
        tumor_type: Type de tumeur

    Returns:
        Document générique
    """
    generic_content = {
        "glioma": (
            "Les gliomes sont des tumeurs cérébrales primitives provenant des cellules gliales. "
            "Ils représentent environ 30% des tumeurs cérébrales. Le traitement dépend du grade "
            "(I à IV), et inclut généralement la chirurgie, la radiothérapie et la chimiothérapie. "
            "Le glioblastome (grade IV) est la forme la plus agressive. "
            "Le diagnostic définitif nécessite une biopsie et une analyse histopathologique."
        ),
        "meningioma": (
            "Les méningiomes sont des tumeurs des méninges (enveloppes du cerveau). "
            "La majorité (90%) sont bénignes (grade I). Ils sont souvent découverts fortuitement. "
            "Le traitement peut être la surveillance, la chirurgie ou la radiochirurgie stéréotaxique, "
            "selon la taille, la localisation et les symptômes. Meilleur pronostic général que les gliomes."
        ),
        "pituitary_tumor": (
            "Les adénomes hypophysaires sont des tumeurs bénignes de l'hypophyse. "
            "Ils peuvent être fonctionnels (sécrétants) ou non fonctionnels. "
            "Les symptômes incluent troubles visuels, troubles hormonaux, céphalées. "
            "Le traitement inclut médicaments (agonistes dopaminergiques), chirurgie transsphénoïdale, "
            "ou radiothérapie. Pronostic généralement favorable."
        ),
        "no_tumor": (
            "Aucune anomalie suspecte n'a été détectée sur cette image IRM cérébrale. "
            "Les structures cérébrales semblent dans les limites de la normale. "
            "Ce résultat doit être interprété par un radiologue ou neurologue qualifié."
        ),
        "unknown": (
            "Type de tumeur non identifié. Une évaluation par un spécialiste est indispensable "
            "pour établir un diagnostic précis et proposer une prise en charge adaptée."
        ),
    }

    return {
        "title": f"Contexte général — {tumor_type.replace('_', ' ').title()}",
        "content": generic_content.get(tumor_type, generic_content["unknown"]),
        "source": "Base de connaissances interne",
        "url": "",
    }
