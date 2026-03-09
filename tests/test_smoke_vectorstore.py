"""Smoke tests for tasks/vectorstore/ — verify importability and basic execution."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# create_chroma_client
# ---------------------------------------------------------------------------
class TestCreateChromaClientSmoke:
    """Smoke tests for tasks.vectorstore.create_chroma_client."""

    def test_importable(self):
        from tasks.vectorstore.create_chroma_client import create_chroma_client

        assert hasattr(create_chroma_client, "__wrapped_function__")

    @pytest.mark.unit
    def test_ephemeral_client(self):
        import sys

        mock_client = MagicMock()
        mock_chromadb = MagicMock()
        mock_chromadb.Client.return_value = mock_client

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            from tasks.vectorstore.create_chroma_client import create_chroma_client

            result = create_chroma_client.__wrapped_function__()
        assert result is mock_client

    @pytest.mark.unit
    def test_persistent_client(self, tmp_path):
        import sys

        mock_client = MagicMock()
        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            from tasks.vectorstore.create_chroma_client import create_chroma_client

            result = create_chroma_client.__wrapped_function__(
                persist_directory=str(tmp_path)
            )
        assert result is mock_client


# ---------------------------------------------------------------------------
# get_or_create_collection
# ---------------------------------------------------------------------------
class TestGetOrCreateCollectionSmoke:
    """Smoke tests for tasks.vectorstore.get_or_create_collection."""

    def test_importable(self):
        from tasks.vectorstore.get_or_create_collection import get_or_create_collection

        assert hasattr(get_or_create_collection, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vectorstore.get_or_create_collection import get_or_create_collection

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        result = get_or_create_collection.__wrapped_function__(
            client=mock_client, collection_name="test"
        )
        assert result is mock_collection
        mock_client.get_or_create_collection.assert_called_once_with(name="test")


# ---------------------------------------------------------------------------
# add_documents
# ---------------------------------------------------------------------------
class TestAddDocumentsSmoke:
    """Smoke tests for tasks.vectorstore.add_documents."""

    def test_importable(self):
        from tasks.vectorstore.add_documents import add_documents

        assert hasattr(add_documents, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vectorstore.add_documents import add_documents

        mock_collection = MagicMock()
        docs = ["doc1", "doc2"]

        success, ids = add_documents.__wrapped_function__(
            collection=mock_collection, documents=docs
        )
        assert success is True
        assert len(ids) == 2
        mock_collection.add.assert_called_once()

    @pytest.mark.unit
    def test_empty_documents(self):
        from tasks.vectorstore.add_documents import add_documents

        mock_collection = MagicMock()
        success, ids = add_documents.__wrapped_function__(
            collection=mock_collection, documents=[]
        )
        assert success is True
        assert ids == []
        mock_collection.add.assert_not_called()


# ---------------------------------------------------------------------------
# collection_count
# ---------------------------------------------------------------------------
class TestCollectionCountSmoke:
    """Smoke tests for tasks.vectorstore.collection_count."""

    def test_importable(self):
        from tasks.vectorstore.collection_count import collection_count

        assert hasattr(collection_count, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vectorstore.collection_count import collection_count

        mock_collection = MagicMock()
        mock_collection.count.return_value = 42

        result = collection_count.__wrapped_function__(collection=mock_collection)
        assert result == 42


# ---------------------------------------------------------------------------
# delete_documents
# ---------------------------------------------------------------------------
class TestDeleteDocumentsSmoke:
    """Smoke tests for tasks.vectorstore.delete_documents."""

    def test_importable(self):
        from tasks.vectorstore.delete_documents import delete_documents

        assert hasattr(delete_documents, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vectorstore.delete_documents import delete_documents

        mock_collection = MagicMock()
        result = delete_documents.__wrapped_function__(
            collection=mock_collection, ids=["id1", "id2"]
        )
        assert result is True
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])

    @pytest.mark.unit
    def test_empty_ids(self):
        from tasks.vectorstore.delete_documents import delete_documents

        mock_collection = MagicMock()
        result = delete_documents.__wrapped_function__(
            collection=mock_collection, ids=[]
        )
        assert result is True
        mock_collection.delete.assert_not_called()


# ---------------------------------------------------------------------------
# query_collection
# ---------------------------------------------------------------------------
class TestQueryCollectionSmoke:
    """Smoke tests for tasks.vectorstore.query_collection."""

    def test_importable(self):
        from tasks.vectorstore.query_collection import query_collection

        assert hasattr(query_collection, "__wrapped_function__")

    @pytest.mark.unit
    def test_with_query_embeddings(self):
        from tasks.vectorstore.query_collection import query_collection

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.1]],
            "ids": [["id1"]],
        }

        docs, metas, dists, ids = query_collection.__wrapped_function__(
            collection=mock_collection,
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=1,
        )
        assert docs == ["doc1"]
        assert ids == ["id1"]

    @pytest.mark.unit
    def test_no_query_returns_empty(self):
        from tasks.vectorstore.query_collection import query_collection

        mock_collection = MagicMock()
        docs, metas, dists, ids = query_collection.__wrapped_function__(
            collection=mock_collection
        )
        assert docs == []
        assert ids == []


# ---------------------------------------------------------------------------
# retrieve_similar
# ---------------------------------------------------------------------------
class TestRetrieveSimilarSmoke:
    """Smoke tests for tasks.vectorstore.retrieve_similar."""

    def test_importable(self):
        from tasks.vectorstore.retrieve_similar import retrieve_similar

        assert hasattr(retrieve_similar, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.vectorstore.retrieve_similar import retrieve_similar

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc_a", "doc_b"]],
            "distances": [[0.2, 0.5]],
        }

        docs, scores = retrieve_similar.__wrapped_function__(
            collection=mock_collection,
            query_embedding=[0.1, 0.2],
            top_k=2,
        )
        assert len(docs) == 2
        assert len(scores) == 2
        # Scores should be converted from distances (lower distance = higher score)
        assert scores[0] > scores[1]

    @pytest.mark.unit
    def test_score_threshold_filtering(self):
        from tasks.vectorstore.retrieve_similar import retrieve_similar

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["doc_a", "doc_b"]],
            "distances": [[0.01, 100.0]],  # second doc is very distant
        }

        docs, scores = retrieve_similar.__wrapped_function__(
            collection=mock_collection,
            query_embedding=[0.1],
            top_k=2,
            score_threshold=0.5,  # filter out low-score results
        )
        # Only the close doc should survive the threshold
        assert len(docs) == 1
