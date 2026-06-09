"""
Script de téléchargement des poids du modèle de classification.

Ce script télécharge et entraîne un modèle EfficientNet-B0 léger
sur un jeu de données synthétique si aucun vrai dataset n'est disponible,
ou sur le dataset Kaggle Brain Tumor MRI si disponible.

Usage :
    python models/download_weights.py
    python models/download_weights.py --kaggle   # avec vraies données Kaggle
"""
import os
import sys
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Classes du modèle (ordre identique à l'entraînement)
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier_weights.pth")


def create_synthetic_dataset(n_samples=200, img_size=224):
    """Crée un petit dataset synthétique pour valider le pipeline."""
    import torch
    X, y = [], []
    for i in range(n_samples):
        # Image aléatoire simulant une IRM (niveaux de gris → 3 canaux)
        img = torch.randn(3, img_size, img_size) * 0.3 + 0.5
        label = i % NUM_CLASSES
        X.append(img)
        y.append(label)
    return torch.stack(X), torch.tensor(y, dtype=torch.long)


def build_model():
    """Construit un modèle EfficientNet-B0 adapté à la classification tumorale."""
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # Remplacer la tête de classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def train_on_synthetic(epochs=5):
    """
    Entraîne rapidement le modèle sur des données synthétiques.
    Permet de valider le pipeline sans données réelles.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    logger.info("Construction du modèle EfficientNet-B0...")
    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    model = model.to(device)

    logger.info("Génération du dataset synthétique (200 samples)...")
    X, y = create_synthetic_dataset(200)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    logger.info(f"Entraînement sur {epochs} époques (données synthétiques)...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        logger.info(f"  Époque {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    # Sauvegarde
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "architecture": "efficientnet_b0",
        "trained_on": "synthetic_data",
        "note": "Modèle entraîné sur données synthétiques — performances non médicales",
    }, MODEL_PATH)

    logger.info(f"✅ Poids sauvegardés : {MODEL_PATH}")
    logger.warning(
        "⚠️  Ce modèle est entraîné sur des données SYNTHÉTIQUES.\n"
        "    Les prédictions ne sont PAS médicalement valides.\n"
        "    Pour un usage sérieux, entraînez sur le dataset Kaggle Brain Tumor MRI."
    )
    return model


def download_and_train_kaggle():
    """
    Instructions pour entraîner sur le vrai dataset Kaggle.
    Nécessite : kaggle CLI + compte Kaggle avec API key.
    """
    logger.info("""
=== ENTRAÎNEMENT SUR DATASET KAGGLE ===

Dataset recommandé : Brain Tumor MRI Dataset
URL : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Étapes :
1. Créez un compte Kaggle : https://www.kaggle.com
2. Téléchargez votre API key (kaggle.json) depuis Account Settings
3. Placez kaggle.json dans ~/.kaggle/
4. Exécutez :
   pip install kaggle
   kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset
   unzip brain-tumor-mri-dataset.zip -d data/brain_tumor/

5. Relancez avec :
   python models/download_weights.py --kaggle --data_dir data/brain_tumor/

Le dataset contient ~7000 images IRM dans 4 classes :
  - glioma_tumor/
  - meningioma_tumor/
  - no_tumor/
  - pituitary_tumor/
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Téléchargement et entraînement du modèle")
    parser.add_argument("--kaggle", action="store_true", help="Afficher les instructions Kaggle")
    parser.add_argument("--epochs", type=int, default=5, help="Nombre d'époques (synthétique)")
    args = parser.parse_args()

    if args.kaggle:
        download_and_train_kaggle()
    else:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Poids déjà présents : {MODEL_PATH}")
            logger.info("Supprimez le fichier et relancez pour ré-entraîner.")
        else:
            train_on_synthetic(epochs=args.epochs)
