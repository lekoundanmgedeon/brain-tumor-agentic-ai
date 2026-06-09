"""
Tests de validation de l'Agent Validation Image.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from PIL import Image

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.validation_image_agent import validate_image_agent


def create_test_image(width: int = 256, height: int = 256, mode: str = "L") -> str:
    """Crée une image de test temporaire et retourne son chemin."""
    arr = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    img = Image.fromarray(arr, mode=mode)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        return f.name


class TestValidateImageAgent:

    def test_valid_image(self):
        """Test avec une image valide."""
        path = create_test_image(256, 256)
        try:
            state = {
                "image_path": path,
                "image_valid": False,
                "image_quality_score": None,
                "image_errors": [],
            }
            result = validate_image_agent(state)
            assert result["image_valid"] is True
            assert result["image_quality_score"] > 0
            assert result["image_errors"] == []
        finally:
            os.unlink(path)

    def test_invalid_path(self):
        """Test avec un chemin inexistant."""
        state = {
            "image_path": "/path/inexistant/image.png",
            "image_valid": False,
            "image_quality_score": None,
            "image_errors": [],
        }
        result = validate_image_agent(state)
        assert result["image_valid"] is False
        assert len(result["image_errors"]) > 0

    def test_empty_path(self):
        """Test avec un chemin vide."""
        state = {
            "image_path": "",
            "image_valid": False,
            "image_quality_score": None,
            "image_errors": [],
        }
        result = validate_image_agent(state)
        assert result["image_valid"] is False

    def test_too_small_image(self):
        """Test avec une image trop petite."""
        path = create_test_image(10, 10)
        try:
            state = {
                "image_path": path,
                "image_valid": False,
                "image_quality_score": None,
                "image_errors": [],
            }
            result = validate_image_agent(state)
            assert result["image_valid"] is False
            assert any("petite" in err.lower() for err in result["image_errors"])
        finally:
            os.unlink(path)

    def test_quality_score_range(self):
        """Test que le score de qualité est dans [0, 1]."""
        path = create_test_image(256, 256)
        try:
            state = {
                "image_path": path,
                "image_valid": False,
                "image_quality_score": None,
                "image_errors": [],
            }
            result = validate_image_agent(state)
            score = result["image_quality_score"]
            assert 0.0 <= score <= 1.0
        finally:
            os.unlink(path)

    def test_jpg_format(self):
        """Test avec une image JPEG."""
        arr = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name, format="JPEG")
            path = f.name
        try:
            state = {
                "image_path": path,
                "image_valid": False,
                "image_quality_score": None,
                "image_errors": [],
            }
            result = validate_image_agent(state)
            assert result["image_valid"] is True
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
