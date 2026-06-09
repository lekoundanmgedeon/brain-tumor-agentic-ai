"""
Agent Validation Image
======================
Vérifie que l'image IRM uploadée est valide et exploitable
avant de lancer le reste du pipeline d'analyse.
"""

import os
import logging
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from workflow.state import BrainTumorState
from config.settings import settings

logger = logging.getLogger(__name__)

ALLOWED_FORMATS = {"PNG", "JPEG", "JPG"}


def validate_image_agent(state: BrainTumorState) -> BrainTumorState:
    """
    Agent de validation de l'image IRM.

    Vérifie :
    - Existence du fichier
    - Format supporté (PNG, JPG, JPEG)
    - Taille minimale et maximale
    - Qualité de l'image (contraste, bruit)
    - Lisibilité du fichier

    Met à jour l'état avec :
    - image_valid (bool)
    - image_quality_score (float 0-1)
    - image_errors (list[str])
    """
    logger.info("[ValidationImage] Démarrage de la validation de l'image.")
    errors = []
    quality_score = 0.0

    image_path = state.get("image_path", "")

    # --- 1. Vérification existence du fichier ---
    if not image_path or not os.path.exists(image_path):
        errors.append(f"Fichier introuvable : '{image_path}'")
        return {
            **state,
            "image_valid": False,
            "image_quality_score": 0.0,
            "image_errors": errors,
        }

    # --- 2. Vérification du format ---
    try:
        img = Image.open(image_path)
        img_format = img.format  # 'PNG', 'JPEG', etc.
        if img_format not in ALLOWED_FORMATS and img_format != "JPEG":
            # PIL retourne JPEG pour les .jpg
            errors.append(
                f"Format non supporté : '{img_format}'. Formats acceptés : PNG, JPG, JPEG."
            )
    except UnidentifiedImageError:
        errors.append("Impossible d'ouvrir le fichier : format non reconnu ou fichier corrompu.")
        return {
            **state,
            "image_valid": False,
            "image_quality_score": 0.0,
            "image_errors": errors,
        }
    except Exception as e:
        errors.append(f"Erreur lors de l'ouverture de l'image : {str(e)}")
        return {
            **state,
            "image_valid": False,
            "image_quality_score": 0.0,
            "image_errors": errors,
        }

    # --- 3. Vérification des dimensions ---
    width, height = img.size
    if width < settings.MIN_IMAGE_SIZE or height < settings.MIN_IMAGE_SIZE:
        errors.append(
            f"Image trop petite : {width}x{height}px. "
            f"Minimum requis : {settings.MIN_IMAGE_SIZE}x{settings.MIN_IMAGE_SIZE}px."
        )
    if width > settings.MAX_IMAGE_SIZE or height > settings.MAX_IMAGE_SIZE:
        errors.append(
            f"Image trop grande : {width}x{height}px. "
            f"Maximum accepté : {settings.MAX_IMAGE_SIZE}x{settings.MAX_IMAGE_SIZE}px."
        )

    # --- 4. Calcul du score de qualité ---
    if not errors:
        quality_score = _compute_quality_score(img)
        if quality_score < 0.15:
            errors.append(
                f"Qualité d'image insuffisante (score={quality_score:.2f}). "
                "L'image semble trop sombre, trop uniforme ou corrompue."
            )

    # --- 5. Résultat final ---
    image_valid = len(errors) == 0
    if image_valid:
        logger.info(
            f"[ValidationImage] Image valide. Score qualité : {quality_score:.2f}. "
            f"Dimensions : {width}x{height}px."
        )
    else:
        logger.warning(f"[ValidationImage] Image invalide. Erreurs : {errors}")

    return {
        **state,
        "image_valid": image_valid,
        "image_quality_score": round(quality_score, 3),
        "image_errors": errors,
    }


def _compute_quality_score(img: Image.Image) -> float:
    """
    Calcule un score de qualité [0.0 - 1.0] basé sur :
    - La variance des niveaux de gris (contraste)
    - La proportion de pixels non-noirs (contenu)

    Args:
        img: Image PIL

    Returns:
        Score de qualité entre 0.0 et 1.0
    """
    try:
        # Conversion en niveaux de gris
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float32)

        # Score 1 : variance normalisée (contraste)
        variance = float(np.var(arr))
        max_variance = (255.0 ** 2) / 4  # Variance maximale théorique
        variance_score = min(variance / max_variance, 1.0)

        # Score 2 : proportion de pixels non-noirs (>10/255)
        non_black = float(np.sum(arr > 10)) / arr.size
        content_score = min(non_black * 2.0, 1.0)  # Pénalise les images très sombres

        # Score composite
        quality = 0.6 * variance_score + 0.4 * content_score
        return float(quality)

    except Exception as e:
        logger.warning(f"[ValidationImage] Erreur calcul qualité : {e}")
        return 0.5  # Score neutre en cas d'erreur
