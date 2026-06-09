"""
Entraînement EfficientNet-B0 avec Transfer Learning.

Stratégie en 2 phases :
  Phase 1 — Feature extraction  : geler le backbone, entraîner seulement la tête
  Phase 2 — Fine-tuning         : dégeler les dernières couches, lr très faible

Compatible avec :
  A) Dataset synthétique réaliste (mode autonome, sans Internet)
  B) Dataset Kaggle réel         (recommandé pour la production)

Usage :
  # Mode synthétique (par défaut)
  python models/train_efficientnet.py

  # Avec le dataset Kaggle décompressé
  python models/train_efficientnet.py --data_dir data/brain_tumor_kaggle/ --real
"""

import os
import sys
import json
import argparse
import logging
import time
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES   = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
NUM_CLASSES   = len(CLASS_NAMES)
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "classifier_weights.pth")
IMG_SIZE      = 224


# ─── Modèle ──────────────────────────────────────────────────

def build_model(freeze_backbone: bool = True):
    """
    Construit EfficientNet-B0 avec tête de classification personnalisée.
    Utilise les poids ImageNet pré-entraînés pour le transfer learning.
    """
    import torch.nn as nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Geler le backbone si Phase 1
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info("Backbone gelé (Phase 1 — feature extraction)")
    else:
        # Dégeler seulement les 3 derniers blocs
        for i, block in enumerate(model.features):
            for param in block.parameters():
                param.requires_grad = (i >= 6)
        logger.info("3 derniers blocs dégelés (Phase 2 — fine-tuning)")

    # Tête de classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


# ─── Données ─────────────────────────────────────────────────

def get_transforms(augment: bool = True):
    """Transformations avec augmentation robuste pour IRM."""
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


def load_datasets(data_dir: str, batch_size: int = 32):
    """
    Charge les datasets depuis la structure ImageFolder.
    Gère à la fois les datasets Kaggle et synthétiques.
    """
    import torch
    from torchvision import datasets
    from torch.utils.data import DataLoader, random_split

    train_dir = os.path.join(data_dir, "Training")
    test_dir  = os.path.join(data_dir, "Testing")

    if not os.path.exists(train_dir):
        train_dir = data_dir
        logger.warning(f"Pas de Training/ trouvé — utilisation directe de {data_dir}")

    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms(augment=True))
    logger.info(f"Classes détectées : {train_ds.classes}")
    logger.info(f"Distribution train : { {c: train_ds.targets.count(i) for i, c in enumerate(train_ds.classes)} }")

    if os.path.exists(test_dir):
        val_ds = datasets.ImageFolder(test_dir, transform=get_transforms(augment=False))
    else:
        n = len(train_ds)
        n_val = max(int(0.2 * n), 4 * NUM_CLASSES)
        train_ds, val_ds = random_split(
            train_ds, [n - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        val_ds.dataset.transform = get_transforms(augment=False)
        logger.info(f"Split 80/20 : {n - n_val} train / {n_val} val")

    # Calcul des poids de classe pour gérer le déséquilibre
    targets = [train_ds[i][1] for i in range(len(train_ds))]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    logger.info(f"Poids de classe : {dict(zip(CLASS_NAMES, weights.round(3)))}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    return train_loader, val_loader, weights


# ─── Métriques ───────────────────────────────────────────────

def compute_metrics(model, loader, device, class_names):
    """Calcule accuracy, per-class accuracy et matrice de confusion."""
    import torch

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(device)).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.tolist())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc    = (preds == labels).mean()

    per_class = {}
    for i, cls in enumerate(class_names):
        mask = labels == i
        per_class[cls] = float((preds[mask] == labels[mask]).mean()) if mask.sum() > 0 else 0.0

    return acc, per_class


# ─── Boucle d'entraînement ───────────────────────────────────

def train_phase(model, loader, val_loader, optimizer, scheduler,
                criterion, device, n_epochs, phase_name, best_acc):
    """Boucle d'entraînement pour une phase donnée."""
    import torch

    for epoch in range(n_epochs):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        t0 = time.time()

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(yb)
            n_correct  += (out.argmax(1) == yb).sum().item()
            n_total    += len(yb)

        scheduler.step()
        train_acc  = n_correct / n_total
        avg_loss   = total_loss / n_total
        val_acc, per_class = compute_metrics(model, val_loader, device, CLASS_NAMES)
        elapsed    = time.time() - t0

        marker = " ← MEILLEUR ✓" if val_acc > best_acc else ""
        logger.info(
            f"[{phase_name}] Époque {epoch+1:02d}/{n_epochs}"
            f"  loss={avg_loss:.4f}  train={train_acc:.2%}"
            f"  val={val_acc:.2%}{marker}"
            f"  ({elapsed:.1f}s)"
        )
        for cls, acc in per_class.items():
            logger.info(f"    {cls:20s}: {acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "num_classes": NUM_CLASSES,
                "architecture": "efficientnet_b0",
                "val_accuracy": float(val_acc),
                "per_class_accuracy": per_class,
                "epoch": epoch + 1,
                "phase": phase_name,
                "note": (
                    "EfficientNet-B0 entraîné avec transfer learning ImageNet. "
                    "USAGE ACADÉMIQUE UNIQUEMENT — non certifié médicalement."
                ),
            }, MODEL_PATH)
            logger.info(f"    💾 Sauvegardé (val_acc={val_acc:.2%})")

    return best_acc


# ─── Pipeline principal ───────────────────────────────────────

def train(data_dir: str, epochs_phase1: int = 10, epochs_phase2: int = 15,
          batch_size: int = 32, is_real: bool = False):
    """
    Entraînement en 2 phases avec transfer learning.

    Phase 1 : backbone gelé → entraîner uniquement la tête (10 époques)
    Phase 2 : fine-tuning des 3 derniers blocs (15 époques, lr×0.1)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU : {torch.cuda.get_device_name(0)}")

    # ── Données ──────────────────────────────────────────────
    logger.info(f"Chargement des données depuis : {data_dir}")
    train_loader, val_loader, class_weights = load_datasets(data_dir, batch_size)
    logger.info(f"Train batches : {len(train_loader)} | Val batches : {len(val_loader)}")

    # Critère avec pondération des classes
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

    # ── Phase 1 : Feature Extraction ─────────────────────────
    logger.info("\n" + "="*50)
    logger.info("PHASE 1 — Feature Extraction (backbone gelé)")
    logger.info("="*50)

    model = build_model(freeze_backbone=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Paramètres entraînables : {n_params:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=epochs_phase1,
        pct_start=0.3,
    )

    best_acc = 0.0
    best_acc = train_phase(model, train_loader, val_loader, optimizer, scheduler,
                            criterion, device, epochs_phase1, "P1", best_acc)

    # ── Phase 2 : Fine-tuning ─────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("PHASE 2 — Fine-tuning (3 derniers blocs dégelés)")
    logger.info("="*50)

    # Recharger le meilleur modèle de la Phase 1
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Meilleur modèle Phase 1 rechargé (val={ckpt.get('val_accuracy',0):.2%})")

    # Dégeler les 3 derniers blocs
    for i, block in enumerate(model.features):
        for param in block.parameters():
            param.requires_grad = (i >= 6)
    # Tête toujours entraînable
    for param in model.classifier.parameters():
        param.requires_grad = True

    n_params2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Paramètres entraînables : {n_params2:,}")

    # LR 10× plus faible pour le fine-tuning
    optimizer2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs_phase2)

    best_acc = train_phase(model, train_loader, val_loader, optimizer2, scheduler2,
                            criterion, device, epochs_phase2, "P2", best_acc)

    # ── Résumé ────────────────────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info(f"✅ Entraînement terminé")
    logger.info(f"   Meilleure val_accuracy : {best_acc:.2%}")
    logger.info(f"   Poids sauvegardés : {MODEL_PATH}")
    if not is_real:
        logger.warning(
            "\n⚠️  Entraîné sur données SYNTHÉTIQUES.\n"
            "   Les performances sont indicatives du pipeline, pas de la valeur médicale.\n"
            "   Pour la production : utilisez le dataset Kaggle avec --real"
        )
    logger.info("="*50)

    return best_acc


# ─── Point d'entrée ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement EfficientNet-B0 (Transfer Learning)")
    parser.add_argument("--data_dir",      type=str, default=None,
                        help="Répertoire du dataset (défaut : génère le dataset synthétique)")
    parser.add_argument("--real",          action="store_true",
                        help="Dataset réel (Kaggle) — désactive l'avertissement synthétique")
    parser.add_argument("--epochs_p1",     type=int, default=10)
    parser.add_argument("--epochs_p2",     type=int, default=15)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--n_train",       type=int, default=300,
                        help="Images train par classe (mode synthétique uniquement)")
    parser.add_argument("--n_val",         type=int, default=75,
                        help="Images val par classe (mode synthétique uniquement)")
    args = parser.parse_args()

    # Générer le dataset synthétique si aucun dossier fourni
    if args.data_dir is None:
        synthetic_dir = "data/synthetic_mri"
        if not os.path.exists(os.path.join(synthetic_dir, "Training", "glioma")):
            logger.info("Dataset synthétique absent — génération en cours...")
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from models.generate_realistic_dataset import generate_dataset
            generate_dataset(synthetic_dir, n_train=args.n_train, n_val=args.n_val)
        else:
            n_existing = sum(
                len(os.listdir(os.path.join(synthetic_dir, "Training", cls)))
                for cls in CLASS_NAMES if os.path.exists(
                    os.path.join(synthetic_dir, "Training", cls))
            )
            logger.info(f"Dataset synthétique existant ({n_existing} images train) — réutilisé")
        args.data_dir = synthetic_dir

    elif not os.path.exists(args.data_dir):
        logger.error(f"Dossier introuvable : {args.data_dir}")
        logger.error("Pour le dataset Kaggle :")
        logger.error("  kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset")
        logger.error("  unzip brain-tumor-mri-dataset.zip -d data/brain_tumor_kaggle/")
        sys.exit(1)

    train(args.data_dir,
          epochs_phase1=args.epochs_p1,
          epochs_phase2=args.epochs_p2,
          batch_size=args.batch_size,
          is_real=args.real)
