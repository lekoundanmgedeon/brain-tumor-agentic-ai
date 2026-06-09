"""
Configuration centrale de l'application.
Charge les variables d'environnement et expose les paramètres globaux.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Paramètres globaux de l'application chargés depuis .env."""

    # --- Mode applicatif ---
    APP_MODE: str = os.getenv("APP_MODE", "DEMO")  # DEMO | PRODUCTION_READY

    # --- Clés API LLM ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")  # gemini | mistral | mock

    # --- RAG / PubMed ---
    PUBMED_EMAIL: str = os.getenv("PUBMED_EMAIL", "demo@example.com")
    ENABLE_PUBMED: bool = os.getenv("ENABLE_PUBMED", "false").lower() == "true"
    ENABLE_NCI_FALLBACK: bool = os.getenv("ENABLE_NCI_FALLBACK", "true").lower() == "true"

    # --- Chemins ---
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    MASKS_DIR: str = "outputs/masks"
    HEATMAPS_DIR: str = "outputs/heatmaps"
    REPORTS_DIR: str = "outputs/reports"
    KNOWLEDGE_BASE_DIR: str = "knowledge_base"

    # --- Seuils de validation ---
    MIN_IMAGE_SIZE: int = 64            # Taille minimale en pixels (largeur et hauteur)
    MAX_IMAGE_SIZE: int = 4096          # Taille maximale en pixels
    MIN_CONFIDENCE_THRESHOLD: float = 0.4  # En dessous → résultat incertain
    HIGH_CONFIDENCE_THRESHOLD: float = 0.75  # Au dessus → confiance haute

    # --- Modèles ---
    # Utiliser gemini-2.0-flash (SDK google-genai v2) ou gemini-1.5-flash
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    # --- Tumeurs valides ---
    VALID_TUMOR_TYPES: list = [
        "glioma",
        "meningioma",
        "pituitary_tumor",
        "no_tumor",
        "unknown",
    ]

    @classmethod
    def get_llm_provider(cls) -> str:
        """
        Détermine le provider LLM effectif selon les clés disponibles.
        Priorité : gemini > mistral > mock
        """
        if cls.GEMINI_API_KEY and cls.LLM_PROVIDER in ("gemini", "auto"):
            return "gemini"
        if cls.GEMINI_API_KEY and cls.LLM_PROVIDER == "mock":
            # Si la clé est là mais LLM_PROVIDER n'est pas configuré, on utilise gemini
            return "gemini"
        if cls.MISTRAL_API_KEY and cls.LLM_PROVIDER in ("mistral", "auto"):
            return "mistral"
        if cls.MISTRAL_API_KEY and cls.LLM_PROVIDER == "mock":
            return "mistral"
        return "mock"

    @classmethod
    def ensure_directories(cls):
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        for d in [
            cls.UPLOAD_DIR,
            cls.OUTPUT_DIR,
            cls.MASKS_DIR,
            cls.HEATMAPS_DIR,
            cls.REPORTS_DIR,
        ]:
            os.makedirs(d, exist_ok=True)


# Instance globale
settings = Settings()
settings.ensure_directories()
