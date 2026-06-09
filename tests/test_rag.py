"""
Tests du module RAG médical.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAG:

    def test_local_knowledge_glioma(self):
        """Test chargement des documents locaux pour le gliome."""
        from services.rag_nci import _load_local_knowledge

        docs = _load_local_knowledge("glioma")
        assert isinstance(docs, list)
        # Le fichier glioma.md doit exister et être chargé
        if docs:
            assert "content" in docs[0]
            assert len(docs[0]["content"]) > 0

    def test_local_knowledge_meningioma(self):
        """Test chargement des documents locaux pour le méningiome."""
        from services.rag_nci import _load_local_knowledge

        docs = _load_local_knowledge("meningioma")
        assert isinstance(docs, list)

    def test_local_knowledge_no_tumor(self):
        """Test pour no_tumor (pas de fichier spécifique attendu)."""
        from services.rag_nci import _load_local_knowledge

        docs = _load_local_knowledge("no_tumor")
        assert isinstance(docs, list)
        # Pour "no_tumor", pas de fichier Markdown → liste vide
        assert len(docs) == 0

    def test_generic_context_fallback(self):
        """Test du contexte générique de secours."""
        from services.rag_nci import _get_generic_context

        for tumor_type in ["glioma", "meningioma", "pituitary_tumor", "no_tumor", "unknown"]:
            doc = _get_generic_context(tumor_type)
            assert "title" in doc
            assert "content" in doc
            assert len(doc["content"]) > 50

    def test_rag_query_builder(self):
        """Test la construction de la requête RAG."""
        from agents.medical_rag_agent import _build_rag_query

        query = _build_rag_query("glioma")
        assert "gliome" in query.lower() or "glioma" in query.lower()
        assert len(query) > 20

    def test_medical_rag_agent(self):
        """Test l'agent RAG médical complet."""
        from agents.medical_rag_agent import medical_rag_agent

        state = {
            "image_path": "",
            "image_valid": True,
            "image_quality_score": 0.8,
            "image_errors": [],
            "tumor_detected": True,
            "suspected_tumor_type": "glioma",
            "confidence": 0.85,
            "segmentation_mask_path": None,
            "heatmap_path": None,
            "tumor_location": "région frontale",
            "tumor_area_mm2": 150.0,
            "tumor_volume_mm3": None,
            "technical_findings": ["zone suspecte"],
            "result_valid": True,
            "confidence_level": "high",
            "validation_warnings": [],
            "can_call_rag": True,
            "can_generate_report": True,
            "rag_query": None,
            "retrieved_documents": [],
            "medical_context_summary": None,
            "sources": [],
            "patient_context": None,
            "final_report": None,
            "final_status": "EN_COURS",
            "llm_provider_used": None,
        }

        result = medical_rag_agent(state)

        assert "rag_query" in result
        assert result["rag_query"] is not None
        assert "medical_context_summary" in result
        assert result["medical_context_summary"] is not None
        assert "retrieved_documents" in result
        assert isinstance(result["retrieved_documents"], list)

    def test_synthesis_no_documents(self):
        """Test la synthèse quand il n'y a pas de documents."""
        from agents.medical_rag_agent import _synthesize_context

        summary = _synthesize_context([], "glioma")
        assert isinstance(summary, str)
        assert len(summary) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
