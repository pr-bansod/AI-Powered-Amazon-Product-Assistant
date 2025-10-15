"""
Unit tests for RAG pipeline functions.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


@pytest.mark.unit
class TestGetEmbedding:
    """Tests for get_embedding function"""

    @patch('api.rag.retrieval_generation.openai.embeddings.create')
    @patch('api.rag.retrieval_generation.get_current_run_tree')
    def test_get_embedding_success(self, mock_run_tree, mock_create, mock_openai_embedding_response):
        """Test successful embedding generation"""
        from api.rag.retrieval_generation import get_embedding

        mock_create.return_value = mock_openai_embedding_response
        mock_run_tree.return_value = None

        result = get_embedding("test query")

        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)
        mock_create.assert_called_once()

    @patch('api.rag.retrieval_generation.openai.embeddings.create')
    def test_get_embedding_with_custom_model(self, mock_create, mock_openai_embedding_response):
        """Test embedding with custom model"""
        from api.rag.retrieval_generation import get_embedding

        mock_create.return_value = mock_openai_embedding_response

        get_embedding("test query", model="text-embedding-ada-002")

        mock_create.assert_called_once_with(
            input="test query",
            model="text-embedding-ada-002"
        )


@pytest.mark.unit
class TestRetrieveData:
    """Tests for retrieve_data function"""

    @patch('api.rag.retrieval_generation.get_embedding')
    @patch('api.rag.retrieval_generation.cohere.ClientV2')
    def test_retrieve_data_success(
        self,
        mock_cohere,
        mock_get_embedding,
        mock_qdrant_client,
        mock_qdrant_query_results,
        sample_query
    ):
        """Test successful data retrieval"""
        from api.rag.retrieval_generation import retrieve_data

        mock_get_embedding.return_value = [0.1] * 1536
        mock_qdrant_client.query_points.return_value = mock_qdrant_query_results

        result = retrieve_data(sample_query, mock_qdrant_client, k=2)

        assert "retrieved_context_ids" in result
        assert "retrieved_context" in result
        assert "retrieved_context_ratings" in result
        assert "similarity_scores" in result
        assert len(result["retrieved_context_ids"]) == 2
        assert result["retrieved_context_ids"] == ["B001", "B002"]
        assert result["similarity_scores"] == [0.95, 0.85]

    @patch('api.rag.retrieval_generation.get_embedding')
    @patch('api.rag.retrieval_generation.cohere.ClientV2')
    def test_retrieve_data_custom_k(
        self,
        mock_cohere,
        mock_get_embedding,
        mock_qdrant_client,
        mock_qdrant_query_results
    ):
        """Test retrieve_data with custom k parameter"""
        from api.rag.retrieval_generation import retrieve_data

        mock_get_embedding.return_value = [0.1] * 1536
        mock_qdrant_client.query_points.return_value = mock_qdrant_query_results

        retrieve_data("test query", mock_qdrant_client, k=10)

        # Verify that query_points was called with limit=10
        call_args = mock_qdrant_client.query_points.call_args
        assert call_args.kwargs['limit'] == 10


@pytest.mark.unit
class TestProcessContext:
    """Tests for process_context function"""

    def test_process_context_formatting(self, sample_retrieved_context):
        """Test context formatting"""
        from api.rag.retrieval_generation import process_context

        result = process_context(sample_retrieved_context)

        assert isinstance(result, str)
        assert "B001" in result
        assert "B002" in result
        assert "4.5" in result
        assert "4.0" in result
        assert "Test product 1 description" in result
        assert "Test product 2 description" in result

    def test_process_context_empty(self):
        """Test processing empty context"""
        from api.rag.retrieval_generation import process_context

        empty_context = {
            "retrieved_context_ids": [],
            "retrieved_context": [],
            "retrieved_context_ratings": []
        }

        result = process_context(empty_context)
        assert result == ""


@pytest.mark.unit
class TestBuildPrompt:
    """Tests for build_prompt function"""

    @patch('api.rag.retrieval_generation.prompt_template_config')
    def test_build_prompt_success(self, mock_template_config, mock_prompt_template, sample_query):
        """Test successful prompt building"""
        from api.rag.retrieval_generation import build_prompt

        mock_template_config.return_value = mock_prompt_template
        context = "Test context"

        result = build_prompt(context, sample_query)

        assert isinstance(result, str)
        assert "Test context" in result
        assert sample_query in result
        mock_template_config.assert_called_once()


@pytest.mark.unit
class TestGenerateAnswer:
    """Tests for generate_answer function"""

    @patch('api.rag.retrieval_generation.instructor.from_openai')
    @patch('api.rag.retrieval_generation.openai.OpenAI')
    @patch('api.rag.retrieval_generation.get_current_run_tree')
    def test_generate_answer_success(
        self,
        mock_run_tree,
        mock_openai,
        mock_instructor,
        mock_instructor_response
    ):
        """Test successful answer generation"""
        from api.rag.retrieval_generation import generate_answer

        # Setup mocks
        mock_client = Mock()
        mock_completion = Mock()
        mock_completion.usage = Mock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )

        mock_client.chat.completions.create_with_completion.return_value = (
            mock_instructor_response,
            mock_completion
        )
        mock_instructor.return_value = mock_client
        mock_run_tree.return_value = None

        result = generate_answer("Test prompt")

        assert result.answer == "This is a test answer about the products."
        assert len(result.references) == 2
        assert result.references[0].id == "B001"


@pytest.mark.unit
class TestRAGPipeline:
    """Tests for complete RAG pipeline"""

    @patch('api.rag.retrieval_generation.generate_answer')
    @patch('api.rag.retrieval_generation.build_prompt')
    @patch('api.rag.retrieval_generation.process_context')
    @patch('api.rag.retrieval_generation.retrieve_data')
    def test_rag_pipeline_integration(
        self,
        mock_retrieve,
        mock_process,
        mock_build,
        mock_generate,
        mock_qdrant_client,
        sample_query,
        sample_retrieved_context,
        mock_instructor_response
    ):
        """Test complete RAG pipeline integration"""
        from api.rag.retrieval_generation import rag_pipeline

        # Setup mocks
        mock_retrieve.return_value = sample_retrieved_context
        mock_process.return_value = "Formatted context"
        mock_build.return_value = "Built prompt"
        mock_generate.return_value = mock_instructor_response

        result = rag_pipeline(sample_query, mock_qdrant_client, top_k=2)

        # Verify all steps were called
        mock_retrieve.assert_called_once_with(sample_query, mock_qdrant_client, 2)
        mock_process.assert_called_once_with(sample_retrieved_context)
        mock_build.assert_called_once_with("Formatted context", sample_query)
        mock_generate.assert_called_once_with("Built prompt")

        # Verify output structure
        assert "answer" in result
        assert "references" in result
        assert "question" in result
        assert "retrieved_context_ids" in result
        assert result["question"] == sample_query


@pytest.mark.integration
@pytest.mark.requires_api
class TestRAGPipelineWrapper:
    """Integration tests for RAG pipeline wrapper"""

    @patch('api.rag.retrieval_generation.QdrantClient')
    @patch('api.rag.retrieval_generation.rag_pipeline')
    def test_rag_pipeline_wrapper_basic(
        self,
        mock_rag_pipeline,
        mock_qdrant_class,
        sample_query,
        mock_qdrant_client,
        mock_qdrant_query_results
    ):
        """Test RAG pipeline wrapper basic functionality"""
        from api.rag.retrieval_generation import rag_pipeline_wrapper

        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_rag_pipeline.return_value = {
            "answer": "Test answer",
            "references": [
                Mock(id="B001", description="Product 1"),
                Mock(id="B002", description="Product 2")
            ],
            "question": sample_query,
            "retrieved_context_ids": ["B001", "B002"],
            "retrieved_context": ["Context 1", "Context 2"],
            "similarity_scores": [0.95, 0.85]
        }

        mock_qdrant_client.query_points.return_value = mock_qdrant_query_results

        result = rag_pipeline_wrapper(sample_query, top_k=2)

        assert "answer" in result
        assert "used_context" in result
        assert result["answer"] == "Test answer"
        mock_qdrant_class.assert_called_once_with(url="http://qdrant:6333")
