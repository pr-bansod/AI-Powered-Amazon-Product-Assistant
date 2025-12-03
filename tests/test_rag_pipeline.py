from unittest.mock import Mock, patch

import numpy as np
from src.api.rag.retrieval_generation import (
    RAGGenerationResponseWithReferences,
    build_prompt,
    generate_answer,
    get_embedding,
    process_context,
    rag_pipeline,
    rag_pipeline_wrapper,
    retrieve_data,
)


class TestGetEmbedding:
    """Test OpenAI embedding generation"""

    @patch("src.api.rag.retrieval_generation.openai")
    @patch("src.api.rag.retrieval_generation.get_current_run_tree")
    def test_get_embedding_success(self, mock_get_current_run_tree, mock_openai, mock_openai_embedding):
        """Test successful embedding generation with OpenAI"""
        mock_openai.embeddings.create.return_value = mock_openai_embedding

        # Mock the current run with proper metadata attribute
        mock_run = Mock()
        mock_run.metadata = {}
        mock_get_current_run_tree.return_value = mock_run

        result = get_embedding("test query")

        assert isinstance(result, list)
        assert len(result) == 1536  # OpenAI embedding dimension
        mock_openai.embeddings.create.assert_called_once_with(input="test query", model="text-embedding-3-small")

    @patch("src.api.rag.retrieval_generation.openai")
    @patch("src.api.rag.retrieval_generation.get_current_run_tree")
    def test_get_embedding_with_custom_model(self, mock_get_current_run_tree, mock_openai, mock_openai_embedding):
        """Test embedding generation with custom model"""
        mock_openai.embeddings.create.return_value = mock_openai_embedding
        mock_get_current_run_tree.return_value = None

        result = get_embedding("test query", model="text-embedding-3-large")

        mock_openai.embeddings.create.assert_called_once_with(input="test query", model="text-embedding-3-large")

    @patch("src.api.rag.retrieval_generation.openai")
    def test_get_embedding_langsmith_tracking(self, mock_openai, mock_openai_embedding):
        """Test LangSmith usage tracking"""
        mock_openai.embeddings.create.return_value = mock_openai_embedding

        with patch("src.api.rag.retrieval_generation.get_current_run_tree") as mock_get_current_run_tree:
            mock_run = Mock()
            mock_run.metadata = {}
            mock_get_current_run_tree.return_value = mock_run

            get_embedding("test query")

            assert "usage_metadata" in mock_run.metadata
            assert mock_run.metadata["usage_metadata"]["input_tokens"] == 10
            assert mock_run.metadata["usage_metadata"]["total_tokens"] == 10


class TestRetrieveData:
    """Test data retrieval from Qdrant vector database"""

    @patch("src.api.rag.retrieval_generation.get_embedding")
    def test_retrieve_data_success(self, mock_get_embedding, mock_qdrant_client, mock_qdrant_points):
        """Test successful data retrieval from Qdrant"""
        mock_get_embedding.return_value = np.random.rand(1536).tolist()
        mock_qdrant_client.query_points.return_value = mock_qdrant_points

        result = retrieve_data("wireless headphones", mock_qdrant_client, k=3)

        assert "retrieved_context_ids" in result
        assert "retrieved_context" in result
        assert "similarity_scores" in result
        assert len(result["retrieved_context_ids"]) == 5  # From mock fixture
        assert "B000EXAMPLE" in result["retrieved_context_ids"][0]

        mock_qdrant_client.query_points.assert_called_once()
        call_args = mock_qdrant_client.query_points.call_args
        assert call_args[1]["collection_name"] == "Amazon-items-collection-01-hybrid-search"
        assert call_args[1]["limit"] == 3

    @patch("src.api.rag.retrieval_generation.get_embedding")
    def test_retrieve_data_hybrid_search(self, mock_get_embedding, mock_qdrant_client, mock_qdrant_points):
        """Test hybrid search configuration with both vector and BM25"""
        mock_get_embedding.return_value = np.random.rand(1536).tolist()
        mock_qdrant_client.query_points.return_value = mock_qdrant_points

        retrieve_data("noise cancelling headphones", mock_qdrant_client)

        call_args = mock_qdrant_client.query_points.call_args
        prefetch = call_args[1]["prefetch"]

        # Check that both vector and BM25 prefetch are configured
        assert len(prefetch) == 2
        assert prefetch[0].using == "text-embedding-3-small"
        assert prefetch[1].using == "bm25"

    def test_retrieve_data_formats_correctly(self, mock_qdrant_client, mock_qdrant_points):
        """Test that retrieved data is formatted correctly"""
        with patch("src.api.rag.retrieval_generation.get_embedding") as mock_get_embedding:
            mock_get_embedding.return_value = np.random.rand(1536).tolist()
            mock_qdrant_client.query_points.return_value = mock_qdrant_points

            result = retrieve_data("test query", mock_qdrant_client)

            # Check data structure
            assert isinstance(result["retrieved_context_ids"], list)
            assert isinstance(result["retrieved_context"], list)
            assert isinstance(result["similarity_scores"], list)
            assert isinstance(result["retrieved_context_ratings"], list)

            # Check data consistency
            assert len(result["retrieved_context_ids"]) == len(result["retrieved_context"])
            assert len(result["retrieved_context_ids"]) == len(result["similarity_scores"])


class TestProcessContext:
    """Test context processing and formatting"""

    def test_process_context_formatting(self, sample_rag_context):
        """Test context formatting for LLM prompt"""
        formatted = process_context(sample_rag_context)

        assert isinstance(formatted, str)
        assert "B001EXAMPLE" in formatted
        assert "B002EXAMPLE" in formatted
        assert "Wireless Bluetooth headphones" in formatted
        assert "Portable Bluetooth speaker" in formatted
        assert "rating:" in formatted

        # Check format structure
        lines = formatted.strip().split("\n")
        assert len(lines) == 2  # Two products
        for line in lines:
            assert line.startswith("- ")
            assert ":," in line  # Product ID separator
            assert "rating:" in line
            assert "description:" in line

    def test_process_context_empty(self):
        """Test context processing with empty data"""
        empty_context = {
            "retrieved_context_ids": [],
            "retrieved_context": [],
            "retrieved_context_ratings": [],
            "similarity_scores": [],
        }

        formatted = process_context(empty_context)
        assert formatted == ""

    def test_process_context_with_ratings(self):
        """Test context processing includes ratings"""
        context_with_ratings = {
            "retrieved_context_ids": ["B001TEST"],
            "retrieved_context": ["Test product description"],
            "retrieved_context_ratings": [4.7],
            "similarity_scores": [0.95],
        }

        formatted = process_context(context_with_ratings)

        assert "rating:4.7" in formatted
        assert "Test product description" in formatted


class TestBuildPrompt:
    """Test prompt building functionality"""

    @patch("src.api.rag.retrieval_generation.prompt_template_config")
    def test_build_prompt_success(self, mock_prompt_template_config):
        """Test successful prompt building"""
        mock_template = Mock()
        mock_template.render.return_value = "Rendered prompt with context and question"
        mock_prompt_template_config.return_value = mock_template

        context = "- B001: Test product"
        question = "What products are available?"

        result = build_prompt(context, question)

        assert result == "Rendered prompt with context and question"
        mock_prompt_template_config.assert_called_once_with("prompts/retrieval_generation.yaml", "retrieval_generation")
        mock_template.render.assert_called_once_with(preprocessed_context=context, question=question)

    @patch("src.api.rag.retrieval_generation.prompt_template_config")
    def test_build_prompt_with_openai_context(self, mock_prompt_template_config):
        """Test prompt building with OpenAI-optimized context"""
        mock_template = Mock()
        mock_template.render.return_value = "OpenAI optimized prompt"
        mock_prompt_template_config.return_value = mock_template

        openai_context = "- B001EXAMPLE:, rating:4.5, description:Sony headphones with OpenAI integration"
        openai_question = "What are the best headphones for AI development?"

        result = build_prompt(openai_context, openai_question)

        mock_template.render.assert_called_once_with(preprocessed_context=openai_context, question=openai_question)


class TestGenerateAnswer:
    """Test answer generation with OpenAI"""

    @patch("src.api.rag.retrieval_generation.instructor.from_openai")
    @patch("src.api.rag.retrieval_generation.openai.OpenAI")
    @patch("src.api.rag.retrieval_generation.get_current_run_tree")
    def test_generate_answer_success(
        self,
        mock_get_current_run_tree,
        mock_openai,
        mock_instructor,
        sample_rag_response_with_references,
        mock_openai_chat_completion,
    ):
        """Test successful answer generation with OpenAI"""
        mock_client = Mock()
        mock_client.chat.completions.create_with_completion.return_value = (
            sample_rag_response_with_references,
            mock_openai_chat_completion,
        )
        mock_instructor.return_value = mock_client

        # Mock the current run with proper metadata attribute
        mock_run = Mock()
        mock_run.metadata = {}
        mock_get_current_run_tree.return_value = mock_run

        prompt = "Test prompt for OpenAI"
        result = generate_answer(prompt)

        assert result == sample_rag_response_with_references
        mock_client.chat.completions.create_with_completion.assert_called_once_with(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            response_model=RAGGenerationResponseWithReferences,
        )

    @patch("src.api.rag.retrieval_generation.instructor.from_openai")
    @patch("src.api.rag.retrieval_generation.openai.OpenAI")
    def test_generate_answer_langsmith_tracking(
        self,
        mock_openai,
        mock_instructor,
        sample_rag_response_with_references,
        mock_openai_chat_completion,
    ):
        """Test LangSmith usage tracking for OpenAI calls"""
        mock_client = Mock()
        mock_client.chat.completions.create_with_completion.return_value = (
            sample_rag_response_with_references,
            mock_openai_chat_completion,
        )
        mock_instructor.return_value = mock_client

        with patch("src.api.rag.retrieval_generation.get_current_run_tree") as mock_get_current_run_tree:
            mock_run = Mock()
            mock_run.metadata = {}
            mock_get_current_run_tree.return_value = mock_run

            generate_answer("test prompt")

            assert "usage_metadata" in mock_run.metadata
            expected_metadata = mock_run.metadata["usage_metadata"]
            assert expected_metadata["input_tokens"] == 100
            assert expected_metadata["output_tokens"] == 50
            assert expected_metadata["total_tokens"] == 150


class TestRAGPipeline:
    """Test end-to-end RAG pipeline"""

    @patch("src.api.rag.retrieval_generation.retrieve_data")
    @patch("src.api.rag.retrieval_generation.process_context")
    @patch("src.api.rag.retrieval_generation.build_prompt")
    @patch("src.api.rag.retrieval_generation.generate_answer")
    def test_rag_pipeline_success(
        self,
        mock_generate_answer,
        mock_build_prompt,
        mock_process_context,
        mock_retrieve_data,
        mock_qdrant_client,
        sample_rag_response_with_references,
    ):
        """Test complete RAG pipeline execution"""
        # Setup mocks
        mock_retrieve_data.return_value = {
            "retrieved_context_ids": ["B001", "B002"],
            "retrieved_context": ["Product 1", "Product 2"],
            "similarity_scores": [0.9, 0.8],
        }
        mock_process_context.return_value = "Formatted context"
        mock_build_prompt.return_value = "Complete prompt"
        mock_generate_answer.return_value = sample_rag_response_with_references

        result = rag_pipeline("What headphones do you recommend?", mock_qdrant_client, top_k=3)

        # Verify all pipeline steps were called
        mock_retrieve_data.assert_called_once_with("What headphones do you recommend?", mock_qdrant_client, 3)
        mock_process_context.assert_called_once()
        mock_build_prompt.assert_called_once_with("Formatted context", "What headphones do you recommend?")
        mock_generate_answer.assert_called_once_with("Complete prompt")

        # Verify result structure
        assert "answer" in result
        assert "references" in result
        assert "question" in result
        assert "retrieved_context_ids" in result
        assert result["question"] == "What headphones do you recommend?"


class TestRAGPipelineWrapper:
    """Test RAG pipeline wrapper with metadata enrichment"""

    @patch("src.api.rag.retrieval_generation.QdrantClient")
    @patch("src.api.rag.retrieval_generation.rag_pipeline")
    def test_rag_pipeline_wrapper_success(self, mock_rag_pipeline, mock_qdrant_client_class):
        """Test RAG pipeline wrapper with metadata enrichment"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_qdrant_client_class.return_value = mock_client_instance

        mock_rag_pipeline.return_value = {
            "answer": "I recommend the wireless headphones.",
            "references": [Mock(id="B001EXAMPLE", description="Wireless headphones")],
            "question": "What headphones do you recommend?",
            "retrieved_context_ids": ["B001EXAMPLE"],
            "retrieved_context": ["Wireless headphones"],
            "similarity_scores": [0.95],
        }

        # Mock Qdrant query for metadata enrichment
        mock_point = Mock()
        mock_point.payload = {"images": "https://example.com/image.jpg", "price": 99.99}
        mock_query_result = Mock()
        mock_query_result.points = [mock_point]
        mock_client_instance.query_points.return_value = mock_query_result

        result = rag_pipeline_wrapper("What headphones do you recommend?", top_k=3)

        # Verify Qdrant client initialization
        mock_qdrant_client_class.assert_called_once_with(url="http://qdrant:6333")

        # Verify pipeline execution
        mock_rag_pipeline.assert_called_once_with("What headphones do you recommend?", mock_client_instance, 3)

        # Verify result structure with enriched metadata
        assert "answer" in result
        assert "used_context" in result
        assert len(result["used_context"]) == 1

        context_item = result["used_context"][0]
        assert context_item["image_url"] == "https://example.com/image.jpg"
        assert context_item["price"] == 99.99
        assert context_item["description"] == "Wireless headphones"

    @patch("src.api.rag.retrieval_generation.QdrantClient")
    @patch("src.api.rag.retrieval_generation.rag_pipeline")
    def test_rag_pipeline_wrapper_openai_integration(self, mock_rag_pipeline, mock_qdrant_client_class):
        """Test wrapper specifically for OpenAI integration scenarios"""
        mock_client_instance = Mock()
        mock_qdrant_client_class.return_value = mock_client_instance

        # Simulate OpenAI-optimized response
        openai_reference = Mock()
        openai_reference.id = "B001OPENAI"
        openai_reference.description = "OpenAI-compatible wireless headphones"

        mock_rag_pipeline.return_value = {
            "answer": "Based on OpenAI compatibility, I recommend these headphones.",
            "references": [openai_reference],
            "question": "What headphones work best with OpenAI APIs?",
            "retrieved_context_ids": ["B001OPENAI"],
            "retrieved_context": ["OpenAI-compatible headphones"],
            "similarity_scores": [0.98],
        }

        # Mock enhanced metadata for OpenAI scenario
        mock_point = Mock()
        mock_point.payload = {
            "images": "https://openai-headphones.com/image.jpg",
            "price": 299.99,
        }
        mock_query_result = Mock()
        mock_query_result.points = [mock_point]
        mock_client_instance.query_points.return_value = mock_query_result

        result = rag_pipeline_wrapper("What headphones work best with OpenAI APIs?")

        # Verify OpenAI-specific result
        assert "OpenAI compatibility" in result["answer"]
        assert result["used_context"][0]["description"] == "OpenAI-compatible wireless headphones"
        assert result["used_context"][0]["price"] == 299.99
