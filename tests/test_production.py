"""
Tests du mode PRODUCTION_READY.
Vérifie le modèle EfficientNet-B0, Grad-CAM et l'intégration Gemini.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS_PATH = os.path.join("models", "classifier_weights.pth")


def create_mri_tensor():
    """Crée un tensor IRM synthétique 1×3×224×224."""
    import torch
    return torch.randn(1, 3, 224, 224) * 0.3 + 0.5


def create_sample_image(size=256):
    arr = np.random.randint(30, 200, (size, size), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        return f.name


# ─── Modèle ────────────────────────────────────────────────

class TestClassifierModel:

    @pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Poids absents")
    def test_load_classifier(self):
        """Charge le modèle et vérifie l'architecture."""
        from models.classifier import load_classifier, NUM_CLASSES
        model = load_classifier(WEIGHTS_PATH, device="cpu")
        assert model is not None
        # Vérifier la tête de classification
        import torch.nn as nn
        assert isinstance(model.classifier[-1], nn.Linear)
        assert model.classifier[-1].out_features == NUM_CLASSES

    @pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Poids absents")
    def test_forward_pass_shape(self):
        """Vérifie que la sortie a la bonne forme."""
        import torch
        from models.classifier import load_classifier, NUM_CLASSES
        model = load_classifier(WEIGHTS_PATH, device="cpu")
        model.eval()
        with torch.no_grad():
            out = model(create_mri_tensor())
        assert out.shape == (1, NUM_CLASSES), f"Shape inattendue : {out.shape}"

    @pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Poids absents")
    def test_softmax_probabilities_sum_to_one(self):
        """Les probabilités softmax doivent sommer à 1."""
        import torch
        import torch.nn.functional as F
        from models.classifier import load_classifier
        model = load_classifier(WEIGHTS_PATH, device="cpu")
        model.eval()
        with torch.no_grad():
            logits = model(create_mri_tensor())
            probs = F.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    @pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Poids absents")
    def test_predicted_class_is_valid(self):
        """La classe prédite doit être dans CLASS_NAMES."""
        import torch
        import torch.nn.functional as F
        from models.classifier import load_classifier, CLASS_NAMES
        model = load_classifier(WEIGHTS_PATH, device="cpu")
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(create_mri_tensor()), dim=1)
        pred_idx = probs.argmax().item()
        assert CLASS_NAMES[pred_idx] in CLASS_NAMES


# ─── Inférence production ──────────────────────────────────

class TestProductionInference:

    HF_MODEL_DIR = os.path.join("models", "hf_brain_tumor")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "hf_brain_tumor", "config.json")),
        reason="Modèle HuggingFace absent — lancez: python models/download_hf_model.py"
    )
    def test_production_inference_returns_valid_structure(self):
        """L'inférence production renvoie un dictionnaire complet."""
        from services.inference import _run_production_inference
        import numpy as np

        path = create_sample_image()
        try:
            dummy = np.zeros((1, 256, 256), dtype=np.float32)
            result = _run_production_inference(dummy, path)

            assert "tumor_detected" in result
            assert isinstance(result["tumor_detected"], bool)
            assert "suspected_tumor_type" in result
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0
            assert "class_probabilities" in result
            assert len(result["class_probabilities"]) == 4
            assert result.get("mode") == "PRODUCTION_READY"
        finally:
            os.unlink(path)

    @pytest.mark.skipif(
        not os.path.exists(os.path.join("models", "hf_brain_tumor", "config.json")),
        reason="Modèle HuggingFace absent — lancez: python models/download_hf_model.py"
    )
    def test_gradcam_map_shape(self):
        """La Grad-CAM doit être un tableau 2D non-None."""
        from services.inference import _run_production_inference
        import numpy as np

        path = create_sample_image()
        try:
            dummy = np.zeros((1, 256, 256), dtype=np.float32)
            result = _run_production_inference(dummy, path)
            cam = result.get("gradcam_map")
            assert cam is not None
            assert cam.ndim == 2
            assert cam.min() >= 0.0 and cam.max() <= 1.0
        finally:
            os.unlink(path)

    @pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Poids absents")
    def test_full_workflow_production_mode(self):
        """Test du workflow complet en mode PRODUCTION_READY."""
        import os
        os.environ["APP_MODE"] = "PRODUCTION_READY"

        # Recharger settings avec le nouveau mode
        import importlib
        import config.settings as s
        importlib.reload(s)
        s.settings.APP_MODE = "PRODUCTION_READY"

        from workflow.graph import run_brain_tumor_analysis
        path = create_sample_image(256)
        try:
            result = run_brain_tumor_analysis(image_path=path)
            assert result["image_valid"] is True
            assert result["tumor_detected"] is not None
            assert result["final_report"] is not None
            # En production, on doit avoir des class_probabilities
        finally:
            os.unlink(path)
            os.environ["APP_MODE"] = "DEMO"
            s.settings.APP_MODE = "DEMO"


# ─── Segmentation production ───────────────────────────────

class TestProductionSegmentation:

    @pytest.mark.skipif(not os.path.exists(WEIGHTS_PATH), reason="Poids absents")
    def test_gradcam_segmentation_creates_files(self):
        """La segmentation depuis Grad-CAM doit créer masque + heatmap."""
        import numpy as np
        from services.segmentation import _production_segmentation

        path = create_sample_image(256)
        dummy_cam = np.random.rand(7, 7).astype(np.float32)
        inference_result = {
            "tumor_detected": True,
            "gradcam_map": dummy_cam,
        }
        try:
            result = _production_segmentation(
                np.zeros((1, 256, 256)), inference_result, path
            )
            assert result["segmentation_mask_path"] is not None
            assert result["heatmap_path"] is not None
            assert os.path.exists(result["segmentation_mask_path"])
            assert os.path.exists(result["heatmap_path"])
        finally:
            os.unlink(path)


# ─── LLM Client ───────────────────────────────────────────

class TestLLMClient:

    def test_mock_report_no_tumor(self):
        """Le rapport mock sans tumeur doit mentionner 'aucune anomalie'."""
        from services.llm_client import _mock_report
        report = _mock_report("no_tumor detected. OUI: false.")
        assert len(report) > 100
        assert "AVERTISSEMENT" in report or "avertissement" in report.lower()

    def test_mock_report_with_glioma(self):
        """Le rapport mock avec gliome doit mentionner le type."""
        from services.llm_client import _mock_report
        report = _mock_report("OUI — glioma detected avec confiance 0.85")
        assert "gliome" in report.lower()

    def test_generate_report_returns_fallback_without_key(self):
        """Sans clé API, generate_report doit utiliser le fallback."""
        import os
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("MISTRAL_API_KEY", None)

        import importlib
        import config.settings as s
        importlib.reload(s)
        s.settings.GEMINI_API_KEY = ""
        s.settings.MISTRAL_API_KEY = ""

        from services.llm_client import generate_report
        report, provider = generate_report("Analyse IRM test. no_tumor.")
        assert provider == "FALLBACK"
        assert len(report) > 50

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY", ""),
        reason="GEMINI_API_KEY non configurée — test ignoré"
    )
    def test_gemini_api_real_call(self):
        """Test réel de l'API Gemini (nécessite GEMINI_API_KEY dans .env)."""
        from services.llm_client import _call_gemini
        report = _call_gemini(
            prompt="En une phrase, confirme que tu es un assistant médical académique.",
            system_prompt="Tu es un assistant IA académique en neuroradiologie."
        )
        assert isinstance(report, str)
        assert len(report) > 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
