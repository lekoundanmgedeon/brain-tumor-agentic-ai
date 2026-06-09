"""
Tests du workflow LangGraph complet.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_sample_mri(width: int = 256, height: int = 256) -> str:
    """
    Crée une image IRM synthétique réaliste pour les tests.
    Simule un cerveau avec une zone centrale plus brillante.
    """
    # Créer un fond sombre
    arr = np.zeros((height, width), dtype=np.uint8)

    # Ajouter un cercle simulant le crâne
    y, x = np.ogrid[:height, :width]
    cx, cy = width // 2, height // 2
    skull = ((x - cx) / (width * 0.45)) ** 2 + ((y - cy) / (height * 0.45)) ** 2
    arr[skull <= 1.0] = 80  # Tissu cérébral

    # Ajouter une zone plus brillante (simulant une tumeur)
    tumor_x, tumor_y = width // 3, height // 3
    tumor = ((x - tumor_x) / 30) ** 2 + ((y - tumor_y) / 25) ** 2
    arr[tumor <= 1.0] = 200

    # Ajouter du bruit
    noise = np.random.normal(0, 10, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="L").convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        return f.name


class TestWorkflow:

    def test_workflow_with_valid_image(self):
        """Test du workflow complet avec une image valide."""
        from workflow.graph import run_brain_tumor_analysis

        path = create_sample_mri()
        try:
            result = run_brain_tumor_analysis(image_path=path, patient_context=None)

            # Vérifications de base
            assert result is not None
            assert "image_valid" in result
            assert result["image_valid"] is True

            # Le radiologue doit avoir produit un résultat
            assert "tumor_detected" in result
            assert result["tumor_detected"] is not None

            # La confiance doit être présente
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0

            # Un rapport doit être généré
            assert "final_report" in result
            assert result["final_report"] is not None
            assert len(result["final_report"]) > 100

            # Le statut final doit être présent
            assert "final_status" in result
            assert result["final_status"] in ["ANALYSE_COMPLETE", "IMAGE_INVALIDE"]

        finally:
            os.unlink(path)

    def test_workflow_with_invalid_image(self):
        """Test du workflow avec une image invalide (fichier inexistant)."""
        from workflow.graph import run_brain_tumor_analysis

        result = run_brain_tumor_analysis(image_path="/nexiste/pas.png", patient_context=None)

        assert result is not None
        assert result["image_valid"] is False
        assert len(result["image_errors"]) > 0
        # Un rapport d'erreur doit quand même être généré
        assert result["final_report"] is not None

    def test_workflow_with_patient_context(self):
        """Test du workflow avec un contexte patient."""
        from workflow.graph import run_brain_tumor_analysis

        path = create_sample_mri()
        patient_ctx = {
            "age": 45,
            "sexe": "Masculin",
            "symptomes": "Céphalées persistantes",
        }

        try:
            result = run_brain_tumor_analysis(image_path=path, patient_context=patient_ctx)
            assert result is not None
            assert result["patient_context"] == patient_ctx
        finally:
            os.unlink(path)

    def test_state_completeness(self):
        """Test que tous les champs de l'état sont présents dans le résultat final."""
        from workflow.graph import run_brain_tumor_analysis
        from workflow.state import BrainTumorState

        path = create_sample_mri()
        try:
            result = run_brain_tumor_analysis(image_path=path)
            expected_keys = [
                "image_path", "image_valid", "image_quality_score", "image_errors",
                "tumor_detected", "suspected_tumor_type", "confidence",
                "result_valid", "confidence_level", "validation_warnings",
                "retrieved_documents", "sources",
                "final_report", "final_status",
            ]
            for key in expected_keys:
                assert key in result, f"Clé manquante dans l'état final : '{key}'"
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
