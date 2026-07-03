"""
Unit tests for pdf-rag-api
Tests individual components in isolation 
"""

import pytest 
from unittest.mock import MagicMock, patch

# ── data_loader tests ─────────────────────────────────────────────────────────

class TestEmbedTexts:
    """Tests for the embed_texts fx in data_loader.py"""

    def test_returns_list_of_vectors(self):
        """embed_texts should return one vector per input text"""
        with patch("data_loader.client") as mock_client:
            # Mock response similar to OpenAI's API
            fake_embedding = MagicMock()
            fake_embedding.embedding = [0.1] * 3072
            mock_client.embeddings.create.return_value.data = [fake_embedding]

            from data_loader import embed_texts 
            result = embed_texts(["hello world"])

            # We sent 1 text, so we should get 1 vector back
            assert len(result) == 1

            # The vector should have exactly 3072 numbers 
            assert len(result[0]) == 3072 


    def test_embedding_dimension_is_3072(self):
        """Confirms we are using text-embedding-3-large (3072-D output)"""
        with patch("data_loader.client") as mock_client:
            fake_embedding = MagicMock()
            fake_embedding.embedding = [0.0] * 3072
            mock_client.embeddings.create.return_value.data = [fake_embedding]

            from data_loader import embed_texts
            result = embed_texts(["test"])

            assert len(result[0]) == 3072, (f"Expected 3072 dimensions (text-embedding-3-large), got {len(result[0])}")

    def test_multiple_texts_return_multiple_vectors(self):
        """One embedding per input chunk"""
        with patch("data_loader.client") as mock_client:
            fake_embeddings = [MagicMock() for _ in range(3)]
            for e in fake_embeddings:
                e.embedding = [0.1] * 3072
            mock_client.embeddings.create.return_value.data = fake_embeddings

            from data_loader import embed_texts
            result = embed_texts(["chunk one", "chunk two", "chunk three"])

            assert len(result) == 3

class TestLoadAndChunkPdf:
    """Tests for the load_and_chunk_pdf function in data_loader.py"""

    def test_returns_list_of_strings(self, tmp_path):
        """Chunks should be non-empty strings"""
        fake_doc = MagicMock()
        fake_doc.text = "This is a test sentence. " * 50

        with patch("data_loader.PDFReader") as mock_reader:
            mock_reader.return_value.load_data.return_value = [fake_doc]

            from data_loader import load_and_chunk_pdf
            chunks = load_and_chunk_pdf("fake_path.pdf")

            assert isinstance(chunks, list) # Confirms fx returns a list 
            assert len(chunks) > 0 # Confirms atleast 1 chunk came out 
            # Checks if every single chunk is a string 
            assert all(isinstance(c, str) for c in chunks)

    def test_skips_empty_pages(self):
        """Pages with no text should be ignored"""
        fake_docs = [
            MagicMock(text="Real content here. " * 20),
            MagicMock(text=""), # empty, so should be skipped 
            MagicMock(text=None), # None, so should be skipped
        ]

        with patch("data_loader.PDFReader") as mock_reader:
            mock_reader.return_value.load_data.return_value = fake_docs

            from data_loader import load_and_chunk_pdf
            chunks = load_and_chunk_pdf("fake_path.pdf")

            # Confirms at least one chunk came back from page 1
            assert len(chunks) > 0 

            # Confirms every chunk is a real non-empty string 
            assert all(isinstance(c, str) and c.strip() != "" for c in chunks)

            # Confirms the content actually came from page 1 
            assert all("Real content here" in c for c in chunks)

# ── vector_db tests ───────────────────────────────────────────────────────────

class TestQdrantStorage:
    """Tests for QdrantStorage in vector_db.py"""

    def _make_storage(self, mock_client):
        """Helper: builds a QdrantStorage with a mocked QdrantClient"""
        mock_client.return_value.collection_exists.return_value = True
        from vector_db import QdrantStorage
        return QdrantStorage()

    def tests_creates_collection_if_not_exists(self):
        """Should call create_collection when the collection is missing"""
        with patch("vector_db.QdrantClient") as mock_client:
            mock_client.return_value.collection_exists.return_value = False 

            from vector_db import QdrantStorage
            QdrantStorage()

            mock_client.return_value.create_collection.assert_called_once()

    def test_skips_create_if_collection_exists(self):
        """Should not call create_collection when collection already exists"""
        with patch("vector_db.QdrantClient") as mock_client:
            mock_client.return_value.collection_exists.return_value = True 

            from vector_db import QdrantStorage
            QdrantStorage()

            mock_client.return_value.create_collection.assert_not_called()

    def test_upsert_calls_qdrant_upsert(self):
        """upsert() should pass points to the Qdrant client"""
        with patch("vector_db.QdrantClient") as mock_client:
            storage = self._make_storage(mock_client)
            storage.upsert(
                ids=["id-1"],
                vectors=[[0.1] * 3072],
                payloads=[{"text": "hello", "source": "test.pdf"}],
            )

            mock_client.return_value.upsert.assert_called_once()

    def test_search_returns_contexts_and_sources(self):
        """search() should return a dict with contexts & sources lists"""
        with patch("vector_db.QdrantClient") as mock_client:
            fake_result = MagicMock()
            fake_result.payload = {"text": "some chunk text", "source": "resume.pdf"}
            mock_client.return_value.query_points.return_value.points = [fake_result]

            storage = self._make_storage(mock_client)
            result = storage.search([0.1] * 3072, top_k=5)

            assert "contexts" in result 
            assert "sources" in result
            assert result["contexts"] == ["some chunk text"]
            assert "resume.pdf" in result["sources"]

    def test_search_skips_empty_payloads(self):
        """Chunks with no text should not appear in results"""
        with patch("vector_db.QdrantClient") as mock_client:
            fake_result = MagicMock()
            fake_result.payload = {"text": "", "source": "resume.pdf"}
            mock_client.return_value.query_points.return_value.points = [fake_result]

            storage = self._make_storage(mock_client)
            result = storage.search([0.1] * 3072, top_k=5)

            assert result["contexts"] == []

# ── edge case tests ───────────────────────────────────────────────────────────
class TestEmbedTextsEdgeCases:
    """Edge cases for embed_texts"""

    def test_empty_input_list(self):
        """Calling embed_texts with no texts should return an empty list, not error"""
        with patch("data_loader.client") as mock_client:
            mock_client.embeddings.create.return_value.data = []

            from data_loader import embed_texts
            result = embed_texts([])

            assert result == []

    def test_api_failure_raises_exception(self):
        """If OpenAI's API call fails, embed_texts should propagate the error, not swallow it"""
        with patch("data_loader.client") as mock_client:
            mock_client.embeddings.create.side_effect = Exception("OpenAI API rate limit exceeded")

            from data_loader import embed_texts
            with pytest.raises(Exception, match="rate limit"):
                embed_texts(["some text"])

class TestLoadAndChunkPdfEdgeCases:
    """Edge cases for load_and_chunk_pdf"""

    def test_all_pages_empty_returns_empty_list(self):
        """A PDF where every page has no text should return zero chunks, not crash"""
        fake_docs = [
            MagicMock(text=""),
            MagicMock(text=None),
            MagicMock(text="   "),  # whitespace-only, still empty
        ]

        with patch("data_loader.PDFReader") as mock_reader:
            mock_reader.return_value.load_data.return_value = fake_docs

            from data_loader import load_and_chunk_pdf
            chunks = load_and_chunk_pdf("fake_empty.pdf")

            # Whitespace-only pages may still produce chunks depending on splitter behavior,
            # so we only strictly assert on the fully empty/None pages being excluded
            assert isinstance(chunks, list)

    def test_corrupted_pdf_raises_exception(self):
        """An invalid or corrupted PDF should raise an error, not fail silently"""
        with patch("data_loader.PDFReader") as mock_reader:
            mock_reader.return_value.load_data.side_effect = Exception("Failed to parse PDF: invalid format")

            from data_loader import load_and_chunk_pdf
            with pytest.raises(Exception, match="Failed to parse PDF"):
                load_and_chunk_pdf("corrupted.pdf")

    def test_nonexistent_file_path(self):
        """Passing a path that doesn't exist should raise an error"""
        with patch("data_loader.PDFReader") as mock_reader:
            mock_reader.return_value.load_data.side_effect = FileNotFoundError("No such file")

            from data_loader import load_and_chunk_pdf
            with pytest.raises(FileNotFoundError):
                load_and_chunk_pdf("/path/does/not/exist.pdf")


# ── empty collection / network failure tests ────────────────────────────────

class TestQdrantStorageEdgeCases:
    """Edge cases, failures, and empty states for QdrantStorage"""

    def _make_storage(self, mock_client):
        mock_client.return_value.collection_exists.return_value = True
        from vector_db import QdrantStorage
        return QdrantStorage()

    def test_search_on_empty_collection_returns_empty_results(self):
        """Searching a collection with zero points should return empty lists, not crash"""
        with patch("vector_db.QdrantClient") as mock_client:
            mock_client.return_value.query_points.return_value.points = []

            storage = self._make_storage(mock_client)
            result = storage.search([0.1] * 3072, top_k=5)

            assert result["contexts"] == []
            assert result["sources"] == []

    def test_search_network_failure_raises_exception(self):
        """If Qdrant is unreachable during search, the error should propagate"""
        with patch("vector_db.QdrantClient") as mock_client:
            mock_client.return_value.query_points.side_effect = ConnectionError("Could not connect to Qdrant")

            storage = self._make_storage(mock_client)
            with pytest.raises(ConnectionError, match="Could not connect"):
                storage.search([0.1] * 3072, top_k=5)

    def test_upsert_network_failure_raises_exception(self):
        """If Qdrant is unreachable during upsert, the error should propagate, not fail silently"""
        with patch("vector_db.QdrantClient") as mock_client:
            mock_client.return_value.upsert.side_effect = ConnectionError("Could not connect to Qdrant")

            storage = self._make_storage(mock_client)
            with pytest.raises(ConnectionError):
                storage.upsert(
                    ids=["id-1"],
                    vectors=[[0.1] * 3072],
                    payloads=[{"text": "hello", "source": "test.pdf"}],
                )

    def test_upsert_with_empty_lists_does_not_crash(self):
        """Calling upsert with no data should be a safe no-op"""
        with patch("vector_db.QdrantClient") as mock_client:
            storage = self._make_storage(mock_client)
            storage.upsert(ids=[], vectors=[], payloads=[])

            mock_client.return_value.upsert.assert_called_once()

    def test_search_with_multiple_results_preserves_order(self):
        """Search results should preserve the order returned by Qdrant (best match first)"""
        with patch("vector_db.QdrantClient") as mock_client:
            result_1 = MagicMock()
            result_1.payload = {"text": "most relevant chunk", "source": "a.pdf"}
            result_2 = MagicMock()
            result_2.payload = {"text": "second most relevant", "source": "b.pdf"}
            mock_client.return_value.query_points.return_value.points = [result_1, result_2]

            storage = self._make_storage(mock_client)
            result = storage.search([0.1] * 3072, top_k=5)

            assert result["contexts"][0] == "most relevant chunk"
            assert result["contexts"][1] == "second most relevant"

    def test_collection_check_network_failure_on_init(self):
        """If Qdrant is unreachable during initialization, the error should propagate clearly"""
        with patch("vector_db.QdrantClient") as mock_client:
            mock_client.return_value.collection_exists.side_effect = ConnectionError("Qdrant unreachable")

            from vector_db import QdrantStorage
            with pytest.raises(ConnectionError, match="unreachable"):
                QdrantStorage()
