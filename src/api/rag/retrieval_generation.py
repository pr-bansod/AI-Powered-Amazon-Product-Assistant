import openai
import instructor
import numpy as np
import cohere
import logging
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field
from langsmith import traceable, get_current_run_tree

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Document
from qdrant_client.http.exceptions import UnexpectedResponse

from api.rag.utils.prompt_management import prompt_template_config


logger = logging.getLogger(__name__)


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question.")
    description: str = Field(description="Short description of the item used to answer the question.")

class RAGGenerationResponseWithReferences(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"}
)
def get_embedding(text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    """Generate embedding vector for the given text using OpenAI's embedding model.

    :param text: Text to generate embedding for
    :type text: str
    :param model: OpenAI embedding model name
    :type model: str
    :returns: Embedding vector if successful, None if failed
    :rtype: Optional[List[float]]
    """
    try:
        logger.info(f"Generating embedding for text using model '{model}'")

        response = openai.embeddings.create(
            input=text,
            model=model,
        )

        current_run = get_current_run_tree()

        if current_run:
            current_run.metadata["usage_metadata"] = {
                "input_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }

        logger.info(f"Successfully generated embedding with {len(response.data[0].embedding)} dimensions")
        return response.data[0].embedding

    except openai.APIError as e:
        logger.error(f"OpenAI API error generating embedding: {e}")
        return None
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit exceeded: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating embedding: {e}")
        return None


@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_data(query: str, qdrant_client: QdrantClient, k: int = 5) -> Optional[Dict[str, List[Any]]]:
    """Retrieve relevant product data using hybrid search (semantic + BM25 with RRF fusion).

    :param query: Search query text
    :type query: str
    :param qdrant_client: Qdrant client instance
    :type qdrant_client: QdrantClient
    :param k: Number of results to retrieve
    :type k: int
    :returns: Dictionary with retrieved context data if successful, None if failed
    :rtype: Optional[Dict[str, List[Any]]]
    """
    try:
        logger.info(f"Retrieving data for query: '{query}' (top_k={k})")

        query_embedding = get_embedding(query)
        if query_embedding is None:
            logger.error("Failed to generate embedding for query")
            return None

        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="text-embedding-3-small",
                    limit=20
                ),
                Prefetch(
                    query=Document(
                        text=query,
                        model="qdrant/bm25"
                    ),
                    using="bm25",
                    limit=20
                )
            ],
            query=FusionQuery(fusion="rrf"),
            limit=k,
        )

        if not results or not results.points:
            logger.warning(f"No results found for query: '{query}'")
            return {
                "retrieved_context_ids": [],
                "retrieved_context": [],
                "retrieved_context_ratings": [],
                "similarity_scores": [],
            }

        retrieved_context_ids = []
        retrieved_context = []
        retrieved_context_ratings = []
        similarity_scores = []

        for result in results.points:
            retrieved_context_ids.append(result.payload.get("parent_asin", ""))
            retrieved_context.append(result.payload.get("description", ""))
            retrieved_context_ratings.append(result.payload.get("average_rating", 0.0))
            similarity_scores.append(result.score)

        logger.info(f"Successfully retrieved {len(retrieved_context_ids)} results")

        return {
            "retrieved_context_ids": retrieved_context_ids,
            "retrieved_context": retrieved_context,
            "retrieved_context_ratings": retrieved_context_ratings,
            "similarity_scores": similarity_scores,
        }

    except UnexpectedResponse as e:
        logger.error(f"Qdrant unexpected response error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error retrieving data: {e}")
        return None


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context: Dict[str, List[Any]]) -> Optional[str]:
    """Format retrieved context into a readable string for the LLM prompt.

    :param context: Dictionary containing retrieved context data
    :type context: Dict[str, List[Any]]
    :returns: Formatted context string if successful, None if failed
    :rtype: Optional[str]
    """
    try:
        logger.info("Formatting retrieved context for prompt")

        if not context or not context.get("retrieved_context_ids"):
            logger.warning("No context data to format")
            return ""

        formatted_context = ""

        for id, chunk, rating in zip(
            context["retrieved_context_ids"],
            context["retrieved_context"],
            context["retrieved_context_ratings"]
        ):
            formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

        logger.info(f"Successfully formatted {len(context['retrieved_context_ids'])} context items")
        return formatted_context

    except KeyError as e:
        logger.error(f"Missing required key in context data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error formatting context: {e}")
        return None


@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_context: str, question: str) -> Optional[str]:
    """Build the final prompt using the template and context.

    :param preprocessed_context: Formatted context string
    :type preprocessed_context: str
    :param question: User's question
    :type question: str
    :returns: Rendered prompt string if successful, None if failed
    :rtype: Optional[str]
    """
    try:
        logger.info("Building prompt from template")

        template = prompt_template_config("src/api/rag/prompts/retrieval_generation.yaml", "retrieval_generation")

        if template is None:
            logger.error("Failed to load prompt template")
            return None

        rendered_prompt = template.render(preprocessed_context=preprocessed_context, question=question)

        logger.info("Successfully built prompt")
        return rendered_prompt

    except Exception as e:
        logger.error(f"Unexpected error building prompt: {e}")
        return None


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"}
)
def generate_answer(prompt: str) -> Optional[RAGGenerationResponseWithReferences]:
    """Generate an answer using OpenAI's GPT model with structured output.

    :param prompt: The formatted prompt to send to the LLM
    :type prompt: str
    :returns: Structured response with answer and references if successful, None if failed
    :rtype: Optional[RAGGenerationResponseWithReferences]
    """
    try:
        logger.info("Generating answer using GPT-4.1-mini")

        client = instructor.from_openai(openai.OpenAI())

        response, raw_response = client.chat.completions.create_with_completion(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            response_model=RAGGenerationResponseWithReferences
        )

        current_run = get_current_run_tree()

        if current_run:
            current_run.metadata["usage_metadata"] = {
                "input_tokens": raw_response.usage.prompt_tokens,
                "output_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens
            }

        logger.info("Successfully generated answer")
        return response

    except openai.APIError as e:
        logger.error(f"OpenAI API error generating answer: {e}")
        return None
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit exceeded: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating answer: {e}")
        return None


@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question: str, qdrant_client: QdrantClient, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Execute the complete RAG pipeline from retrieval to answer generation.

    :param question: User's question
    :type question: str
    :param qdrant_client: Qdrant client instance
    :type qdrant_client: QdrantClient
    :param top_k: Number of results to retrieve
    :type top_k: int
    :returns: Dictionary with answer and context if successful, None if failed
    :rtype: Optional[Dict[str, Any]]
    """
    try:
        logger.info(f"Starting RAG pipeline for question: '{question}'")

        retrieved_context = retrieve_data(question, qdrant_client, top_k)
        if retrieved_context is None:
            logger.error("Failed to retrieve context")
            return None

        preprocessed_context = process_context(retrieved_context)
        if preprocessed_context is None:
            logger.error("Failed to process context")
            return None

        prompt = build_prompt(preprocessed_context, question)
        if prompt is None:
            logger.error("Failed to build prompt")
            return None

        answer = generate_answer(prompt)
        if answer is None:
            logger.error("Failed to generate answer")
            return None

        final_result = {
            "answer": answer.answer,
            "references": answer.references,
            "question": question,
            "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
            "retrieved_context": retrieved_context["retrieved_context"],
            "similarity_scores": retrieved_context["similarity_scores"]
        }

        logger.info("Successfully completed RAG pipeline")
        return final_result

    except Exception as e:
        logger.error(f"Unexpected error in RAG pipeline: {e}")
        return None


def rag_pipeline_wrapper(question: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Wrapper function for RAG pipeline that enriches results with product metadata.

    :param question: User's question
    :type question: str
    :param top_k: Number of results to retrieve
    :type top_k: int
    :returns: Dictionary with answer and enriched context if successful, None if failed
    :rtype: Optional[Dict[str, Any]]
    """
    try:
        logger.info(f"Starting RAG pipeline wrapper for question: '{question}'")

        qdrant_client = QdrantClient(url="http://qdrant:6333")

        result = rag_pipeline(question, qdrant_client, top_k)
        if result is None:
            logger.error("RAG pipeline failed")
            return None

        used_context = []
        dummy_vector = np.zeros(1536).tolist()

        for item in result.get("references", []):
            try:
                query_result = qdrant_client.query_points(
                    collection_name="Amazon-items-collection-01-hybrid-search",
                    query=dummy_vector,
                    using="text-embedding-3-small",
                    limit=1,
                    with_payload=True,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="parent_asin",
                                match=MatchValue(value=item.id))
                        ]
                    )
                )

                if query_result.points:
                    payload = query_result.points[0].payload
                    image_url = payload.get("image")
                    price = payload.get("price")
                    if image_url:
                        used_context.append({
                            "image_url": image_url,
                            "price": price,
                            "description": item.description
                        })
                else:
                    logger.warning(f"No payload found for item ID: {item.id}")

            except Exception as e:
                logger.error(f"Error fetching metadata for item {item.id}: {e}")
                continue

        logger.info(f"Successfully enriched {len(used_context)} context items with metadata")

        return {
            "answer": result["answer"],
            "used_context": used_context
        }

    except UnexpectedResponse as e:
        logger.error(f"Qdrant connection error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in RAG pipeline wrapper: {e}")
        return None