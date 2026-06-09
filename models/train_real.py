"""
Entraînement du modèle EfficientNet-B0 sur le dataset Kaggle réel.

Dataset requis : Brain Tumor MRI Dataset
URL : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Usage :
    python models/train_real.py --data_dir data/brain_tumor/ --epochs 20
    python models/train_real.py --data_dir data/brain_tumor/ --epochs 30 --batch_size 32
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASS_MAP = {
    "glioma_tumor": "glioma",
    "meningioma_tumor": "meningioma",
    "no_tumor": "no_tumor",
    "pituitary_tumor": "pituitary_tumor",
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary_tumor",
}
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
NUM_CLASSES = 4
MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier_weights.pth")


def build_datasets(data_dir: str, img_size: int = 224):
    """
    Construit les datasets d'entraînement et de validation.

    Args:
        data_dir: Répertoire racine du dataset
        img_size: Taille cible des images

    Returns:
        (train_loader, val_loader, class_counts)
    """
    import torch
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader, random_split

    # Transformations robustes pour IRM
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # IRM → 3 canaux
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Chercher le dossier Training/
    train_dir = os.path.join(data_dir, "Training")
    test_dir  = os.path.join(data_dir, "Testing")

    if not os.path.exists(train_dir):
        # Peut-être que les dossiers sont directement dans data_dir
        train_dir = data_dir
        logger.warning(f"Pas de sous-dossier Training/ trouvé — utilisation de {data_dir} directement.")

    logger.info(f"Chargement des données depuis : {train_dir}")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

    # Remapper les classes si nécessaire
    logger.info(f"Classes trouvées : {train_dataset.classes}")

    if os.path.exists(test_dir):
        val_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    else:
        # Créer une validation 80/20
        n = len(train_dataset)
        n_train = int(0.8 * n)
        train_dataset, val_dataset = random_split(
            train_dataset, [n_train, n - n_train],
            generator=torch.Generator().manual_seed(42)
        )
        val_dataset.dataset.transform = val_transform

    logger.info(f"Train : {len(train_dataset)} images | Val : {len(val_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

    return train_loader, val_loader


def train(data_dir: str, epochs: int = 20, lr: float = 1e-3):
    """
    Boucle d'entraînement complète avec early stopping et sauvegarde du meilleur modèle.

    Args:
        data_dir: Répertoire du dataset
        epochs: Nombre d'époques
        lr: Learning rate initial
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision.models import efficientnet_b0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU : {torch.cuda.get_device_name(0)}")

    # Modèle
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model = model.to(device)
    logger.info(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    # Données
    train_loader, val_loader = build_datasets(data_dir)

    # Optimisation
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ── Entraînement ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        avg_loss = train_loss / len(train_loader)

        # ── Validation ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_acc = correct / total

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Époque {epoch+1:02d}/{epochs} — "
            f"Loss: {avg_loss:.4f} — Val Acc: {val_acc:.2%}"
            f"{' ← MEILLEUR' if val_acc > best_acc else ''}"
        )

        # Sauvegarde du meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "architecture": "efficientnet_b0",
                "trained_on": "kaggle_brain_tumor_mri_dataset",
                "val_accuracy": val_acc,
                "epoch": epoch + 1,
                "note": (
                    "Modèle entraîné sur le dataset Kaggle Brain Tumor MRI. "
                    "USAGE ACADEMIQUE UNIQUEMENT — non certifié médicalement."
                ),
            }, MODEL_PATH)

    logger.info(f"\n✅ Entraînement terminé. Meilleure accuracy : {best_acc:.2%}")
    logger.info(f"   Poids sauvegardés : {MODEL_PATH}")

    # Sauvegarder l'historique
    with open("models/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("   Historique : models/training_history.json")

    logger.warning(
        "\n⚠️  RAPPEL MÉDICAL : Ce modèle est entraîné sur un dataset public.\n"
        "    Les performances en conditions cliniques réelles sont inconnues.\n"
        "    Usage académique uniquement — consultez un professionnel de santé."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement EfficientNet-B0 sur IRM cérébrales")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Répertoire du dataset (doit contenir Training/ et Testing/)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Nombre d'époques (défaut: 20)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate initial (défaut: 0.001)")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        logger.error(f"Répertoire introuvable : {args.data_dir}")
        print("\n📥 Téléchargez le dataset Kaggle :")
        print("   kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset")
        print("   unzip brain-tumor-mri-dataset.zip -d data/brain_tumor/")
        sys.exit(1)

    train(args.data_dir, epochs=args.epochs, lr=args.lr)
