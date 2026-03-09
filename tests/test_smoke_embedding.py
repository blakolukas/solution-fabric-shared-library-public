"""Smoke tests for tasks/embedding/ — verify importability and basic execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------
class TestCosineSimilaritySmoke:
    """Smoke tests for tasks.embedding.cosine_similarity."""

    def test_importable(self):
        from tasks.embedding.cosine_similarity import cosine_similarity

        assert hasattr(cosine_similarity, "__wrapped_function__")

    @pytest.mark.unit
    def test_identical_vectors(self):
        from tasks.embedding.cosine_similarity import cosine_similarity

        vec = [1.0, 0.0, 0.0]
        result = cosine_similarity.__wrapped_function__(vec, vec)
        assert result == pytest.approx(1.0)

    @pytest.mark.unit
    def test_orthogonal_vectors(self):
        from tasks.embedding.cosine_similarity import cosine_similarity

        result = cosine_similarity.__wrapped_function__([1.0, 0.0], [0.0, 1.0])
        assert result == pytest.approx(0.0)

    @pytest.mark.unit
    def test_empty_vectors_return_zero(self):
        from tasks.embedding.cosine_similarity import cosine_similarity

        result = cosine_similarity.__wrapped_function__([], [])
        assert result == 0.0


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------
class TestEmbedTextSmoke:
    """Smoke tests for tasks.embedding.embed_text."""

    def test_importable(self):
        from tasks.embedding.embed_text import embed_text

        assert hasattr(embed_text, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.embedding.embed_text import embed_text

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        result = embed_text.__wrapped_function__("hello world", model=mock_model)
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.unit
    def test_empty_text_returns_empty(self):
        from tasks.embedding.embed_text import embed_text

        mock_model = MagicMock()
        result = embed_text.__wrapped_function__("", model=mock_model)
        assert result == []


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------
class TestEmbedTextsSmoke:
    """Smoke tests for tasks.embedding.embed_texts."""

    def test_importable(self):
        from tasks.embedding.embed_texts import embed_texts

        assert hasattr(embed_texts, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.embedding.embed_texts import embed_texts

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        result = embed_texts.__wrapped_function__(["hello", "world"], model=mock_model)
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.unit
    def test_empty_list_returns_empty(self):
        from tasks.embedding.embed_texts import embed_texts

        mock_model = MagicMock()
        result = embed_texts.__wrapped_function__([], model=mock_model)
        assert result == []


# ---------------------------------------------------------------------------
# load_embedding_model
# ---------------------------------------------------------------------------
class TestLoadEmbeddingModelSmoke:
    """Smoke tests for tasks.embedding.load_embedding_model."""

    def test_importable(self):
        from tasks.embedding.load_embedding_model import load_embedding_model

        assert hasattr(load_embedding_model, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution_with_mock(self):
        import sys

        mock_model_instance = MagicMock()
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model_instance

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            from tasks.embedding.load_embedding_model import load_embedding_model

            result = load_embedding_model.__wrapped_function__(
                model_name="all-MiniLM-L6-v2", device="cpu"
            )
        assert result is mock_model_instance


# ---------------------------------------------------------------------------
# rank_by_similarity
# ---------------------------------------------------------------------------
class TestRankBySimilaritySmoke:
    """Smoke tests for tasks.embedding.rank_by_similarity."""

    def test_importable(self):
        from tasks.embedding.rank_by_similarity import rank_by_similarity

        assert hasattr(rank_by_similarity, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.embedding.rank_by_similarity import rank_by_similarity

        query = [1.0, 0.0]
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        scores, indices = rank_by_similarity.__wrapped_function__(
            query, embeddings, top_k=2
        )
        assert len(scores) == 2
        assert len(indices) == 2
        # The most similar vector (index 0 - identical) should be first
        assert indices[0] == 0

    @pytest.mark.unit
    def test_empty_inputs_return_empty(self):
        from tasks.embedding.rank_by_similarity import rank_by_similarity

        scores, indices = rank_by_similarity.__wrapped_function__([], [])
        assert scores == []
        assert indices == []
