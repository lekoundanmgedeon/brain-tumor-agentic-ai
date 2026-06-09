"""
Définition de l'architecture du classificateur de tumeurs cérébrales.
EfficientNet-B0 adapté à la classification multi-classes sur IRM.

Classes supportées :
  0 → glioma
  1 → meningioma
  2 → no_tumor
  3 → pituitary_tumor
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224


def build_classifier() -> nn.Module:
    """
    Construit le modèle EfficientNet-B0 pour la classification tumorale.
    Tête de classification remplacée pour 4 classes.

    Returns:
        Modèle PyTorch non chargé (poids aléatoires)
    """
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def load_classifier(weights_path: str, device: str = "cpu") -> nn.Module:
    """
    Charge le classificateur avec ses poids entraînés.

    Args:
        weights_path: Chemin vers le fichier .pth
        device: 'cpu' ou 'cuda'

    Returns:
        Modèle en mode eval(), prêt pour l'inférence
    """
    model = build_classifier()
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)
