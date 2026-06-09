"""
Téléchargement du modèle EfficientNet-B5 pré-entraîné sur le dataset
Brain Tumor MRI (Kaggle) depuis HuggingFace.

Modèle : DunnBC22/efficientnet-b5-Brain_Tumors_Image_Classification
  - Architecture : EfficientNet-B5 (fine-tuné depuis google/efficientnet-b5)
  - Dataset      : Brain Tumor MRI Dataset (Kaggle) — 7023 images, 4 classes
  - Accuracy     : 80.2% sur le jeu de test
  - Licence      : Apache 2.0
  - Format       : HuggingFace Transformers (PyTorch)

Usage :
    python models/download_hf_model.py
"""

import os
import sys

HF_MODEL_ID  = "DunnBC22/efficientnet-b5-Brain_Tumors_Image_Classification"
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "hf_brain_tumor")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "classifier_weights.pth")


def download_and_convert():
    """
    1. Télécharge le modèle HuggingFace (~114 MB)
    2. Convertit + sauvegarde en .pth compatible avec le pipeline existant
    3. Sauvegarde aussi le modèle HuggingFace natif dans models/hf_brain_tumor/
    """
    print("=" * 60)
    print("  Téléchargement EfficientNet-B5 Brain Tumor (HuggingFace)")
    print("=" * 60)
    print(f"  Source : {HF_MODEL_ID}")
    print(f"  Dest   : {MODEL_DIR}")
    print()

    # ── Vérifier les dépendances ──────────────────────────────
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch
    except ImportError as e:
        print(f"❌ Dépendance manquante : {e}")
        print("   Installez avec : pip install transformers torch")
        sys.exit(1)

    # ── Téléchargement ────────────────────────────────────────
    print("📥 Téléchargement du modèle depuis HuggingFace (~114 MB)...")
    try:
        processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        model     = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
    except Exception as e:
        print(f"❌ Erreur de téléchargement : {e}")
        print()
        print("Vérifiez :")
        print("  1. Votre connexion Internet")
        print("  2. pip install -U transformers huggingface_hub")
        sys.exit(1)

    # ── Sauvegarde format HuggingFace natif ───────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    processor.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print(f"✅ Modèle HuggingFace sauvegardé dans : {MODEL_DIR}/")

    # ── Lecture des classes ───────────────────────────────────
    id2label = model.config.id2label   # {0: 'Glioma', 1: 'Meningioma', ...}
    label2id = model.config.label2id
    print(f"   Classes : {id2label}")

    # ── Conversion en .pth compatible pipeline ────────────────
    # Mapping HuggingFace → format interne du projet
    HF_TO_INTERNAL = {
        "Glioma":         "glioma",
        "Meningioma":     "meningioma",
        "No Tumor":       "no_tumor",
        "Pituitary Tumor":"pituitary_tumor",
        # Variantes possibles
        "glioma":         "glioma",
        "meningioma":     "meningioma",
        "notumor":        "no_tumor",
        "pituitary":      "pituitary_tumor",
    }

    class_names_internal = [
        HF_TO_INTERNAL.get(id2label[i], id2label[i].lower().replace(" ", "_"))
        for i in range(len(id2label))
    ]

    import torch
    torch.save({
        "model_state_dict":    model.state_dict(),
        "class_names":         class_names_internal,
        "hf_id2label":         id2label,
        "num_classes":         len(id2label),
        "architecture":        "efficientnet_b5_hf",
        "hf_model_id":         HF_MODEL_ID,
        "val_accuracy":        0.802,
        "trained_on":          "Kaggle Brain Tumor MRI Dataset (7023 images)",
        "note": (
            "EfficientNet-B5 fine-tuné sur Brain Tumor MRI Kaggle. "
            "Accuracy 80.2%. USAGE ACADÉMIQUE — non certifié médicalement."
        ),
    }, WEIGHTS_PATH)

    print(f"✅ Poids .pth sauvegardés : {WEIGHTS_PATH}")
    size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
    print(f"   Taille : {size_mb:.1f} MB")
    print()
    print("✅ Installation terminée !")
    print()
    print("Configurez .env :")
    print("   APP_MODE=PRODUCTION_READY")
    print("   LLM_PROVIDER=gemini  # ou mistral")
    print()
    print("Lancez : streamlit run app.py")


if __name__ == "__main__":
    download_and_convert()
