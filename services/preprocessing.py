"""
Service de prétraitement des images IRM.
Prépare l'image pour l'inférence du modèle de détection/segmentation.
"""

import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

TARGET_SIZE = (256, 256)  # Taille cible pour les modèles


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Prétraite une image IRM pour l'inférence.

    Étapes :
    1. Chargement de l'image
    2. Conversion en niveaux de gris
    3. Redimensionnement
    4. Normalisation [0, 1]
    5. Ajout de la dimension batch

    Args:
        image_path: Chemin vers l'image IRM

    Returns:
        Tableau NumPy de forme (1, H, W) normalisé [0, 1]
    """
    try:
        img = Image.open(image_path).convert("L")  # Niveaux de gris
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)

        # Normalisation min-max
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 0:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = arr / 255.0

        # Shape : (1, H, W) — batch de 1
        arr = arr[np.newaxis, ...]
        logger.debug(f"[Preprocessing] Image prétraitée. Shape : {arr.shape}")
        return arr

    except Exception as e:
        logger.error(f"[Preprocessing] Erreur lors du prétraitement : {e}")
        raise


def load_image_rgb(image_path: str, size: tuple = (512, 512)) -> np.ndarray:
    """
    Charge une image IRM en RGB pour la visualisation.

    Args:
        image_path: Chemin vers l'image
        size: Taille cible (largeur, hauteur)

    Returns:
        Tableau NumPy (H, W, 3) uint8
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """
    Normalise un tableau NumPy pour l'affichage (plage 0-255 uint8).

    Args:
        arr: Tableau de forme quelconque

    Returns:
        Tableau uint8 [0, 255]
    """
    arr = arr.squeeze()
    if arr.max() - arr.min() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = (arr * 255).astype(np.uint8)
    return arr
