"""
Service RAG — PubMed
=====================
Interroge l'API NCBI/PubMed pour récupérer des résumés d'articles
médicaux pertinents sur le type de tumeur suspecté.
"""

import logging
import time
from typing import List, Dict, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RESULTS = 5
REQUEST_TIMEOUT = 10  # secondes


def search_pubmed(tumor_type: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """
    Recherche des articles PubMed sur un type de tumeur.

    Args:
        tumor_type: Type de tumeur (ex: "glioma", "meningioma")
        max_results: Nombre maximum d'articles à récupérer

    Returns:
        Liste de dictionnaires {"title", "abstract", "url", "source"}
    """
    if not settings.ENABLE_PUBMED:
        logger.info("[PubMed] PubMed désactivé — utilisation des documents locaux.")
        return []

    query = _build_pubmed_query(tumor_type)
    logger.info(f"[PubMed] Recherche : '{query}'")

    try:
        # --- 1. Recherche des IDs d'articles ---
        pmids = _search_pmids(query, max_results)
        if not pmids:
            logger.warning("[PubMed] Aucun article trouvé.")
            return []

        # --- 2. Récupération des résumés ---
        documents = _fetch_abstracts(pmids)
        logger.info(f"[PubMed] {len(documents)} article(s) récupéré(s).")
        return documents

    except requests.exceptions.ConnectionError:
        logger.warning("[PubMed] Pas de connexion Internet — PubMed non disponible.")
        return []
    except requests.exceptions.Timeout:
        logger.warning("[PubMed] Timeout lors de la requête PubMed.")
        return []
    except Exception as e:
        logger.error(f"[PubMed] Erreur inattendue : {e}", exc_info=True)
        return []


def _build_pubmed_query(tumor_type: str) -> str:
    """
    Construit une requête PubMed médicale ciblée.

    Args:
        tumor_type: Type de tumeur brut (ex: "pituitary_tumor")

    Returns:
        Requête formatée pour PubMed
    """
    # Normalisation du type de tumeur
    type_map = {
        "glioma": "glioma brain tumor MRI diagnosis treatment",
        "meningioma": "meningioma brain MRI diagnosis prognosis",
        "pituitary_tumor": "pituitary adenoma MRI diagnosis treatment",
        "no_tumor": "brain MRI normal findings",
        "unknown": "brain tumor MRI detection",
    }
    return type_map.get(tumor_type, f"{tumor_type} brain tumor MRI")


def _search_pmids(query: str, max_results: int) -> List[str]:
    """
    Recherche les PMIDs via l'API esearch de NCBI.

    Args:
        query: Requête de recherche
        max_results: Nombre max de résultats

    Returns:
        Liste de PMIDs (identifiants PubMed)
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
        "email": settings.PUBMED_EMAIL,
    }

    response = requests.get(
        f"{PUBMED_BASE_URL}/esearch.fcgi",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """
    Récupère les titres et résumés via l'API efetch de NCBI.

    Args:
        pmids: Liste d'identifiants PubMed

    Returns:
        Liste de documents structurés
    """
    if not pmids:
        return []

    # Petite pause pour respecter les rate limits NCBI (3 req/s sans clé)
    time.sleep(0.4)

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
        "email": settings.PUBMED_EMAIL,
    }

    response = requests.get(
        f"{PUBMED_BASE_URL}/efetch.fcgi",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    return _parse_pubmed_xml(response.text, pmids)


def _parse_pubmed_xml(xml_text: str, pmids: List[str]) -> List[Dict]:
    """
    Parse le XML retourné par efetch pour extraire titres et résumés.

    Args:
        xml_text: Contenu XML brut
        pmids: PMIDs correspondants

    Returns:
        Liste de documents structurés
    """
    import xml.etree.ElementTree as ET

    documents = []
    try:
        root = ET.fromstring(xml_text)
        articles = root.findall(".//PubmedArticle")

        for i, article in enumerate(articles):
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            pmid_el = article.find(".//PMID")

            title = title_el.text if title_el is not None else "Titre non disponible"
            abstract = abstract_el.text if abstract_el is not None else "Résumé non disponible"
            pmid = pmid_el.text if pmid_el is not None else (pmids[i] if i < len(pmids) else "?")

            documents.append({
                "title": title,
                "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source": "PubMed",
                "pmid": pmid,
            })

    except ET.ParseError as e:
        logger.warning(f"[PubMed] Erreur de parsing XML : {e}")

    return documents
