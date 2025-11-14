import instructor
import numpy as np
import openai
from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    Prefetch,
)

from .utils.prompt_management import prompt_template_config


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question.")
    description: str = Field(description="Short description of the item used to answer the question.")


class RAGGenerationResponseWithReferences(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question.")


@traceable(
    name="embed-query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
)
def get_embedding(text, model="text-embedding-3-small"):
    """
    Generate embeddings for the given text using OpenAI's embedding model.

    This function converts text into a high-dimensional vector representation
    that captures semantic meaning, enabling similarity-based search.

    Args:
        text (str): The input text to generate embeddings for
        model (str, optional): The OpenAI embedding model to use.
            Defaults to "text-embedding-3-small" (1536 dimensions).

    Returns:
        list[float]: A vector embedding representing the input text

    Raises:
        openai.APIError: If the API request fails

    Example:
        >>> embedding = get_embedding("wireless earphones")
        >>> len(embedding)
        1536
    """
    response = openai.embeddings.create(input=text, model=model)
    # to print llm usage on langsmit UI
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.data[0].embedding


@traceable(name="retrieve-data", run_type="retriever")
def retrieve_data(query, qdrant_client, k=5):
    """
    Retrieve relevant product data from Qdrant vector database based on semantic similarity.

    This function performs the retrieval step of the RAG pipeline by:
    1. Converting the query to embeddings
    2. Searching the Qdrant collection for similar product vectors
    3. Extracting product metadata and similarity scores

    Args:
        query (str): The user's search query or question
        qdrant_client (QdrantClient): Connected Qdrant client instance
        k (int, optional): Number of most similar products to retrieve. Defaults to 5.

    Returns:
        dict: A dictionary containing:
            - retrieved_context_ids (list[str]): Product ASINs
            - retrieved_context (list[str]): Product descriptions
            - similarity_scores (list[float]): Cosine similarity scores

    Example:
        >>> client = QdrantClient(url="http://localhost:6333")
        >>> results = retrieve_data("noise cancelling headphones", client, k=10)
        >>> print(len(results["retrieved_context_ids"]))
        10
    """
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(query=query_embedding, using="text-embedding-3-small", limit=20),
            Prefetch(query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    retrieved_context_ratings = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }


@traceable(name="format-retrieved-context", run_type="prompt")
def process_context(context):
    """
    Format retrieved product data into a structured string for the LLM prompt.

    Converts the raw retrieval results into a human-readable bulleted list
    that pairs product IDs with their descriptions.

    Args:
        context (dict): Dictionary containing:
            - retrieved_context_ids (list[str]): Product ASINs
            - retrieved_context (list[str]): Product descriptions
            - similarity_scores (list[float]): Similarity scores (not used in formatting)

    Returns:
        str: Formatted string with each product on a new line in the format:
            "- {product_id}: {description}\n"

    Example:
        >>> context = {
        ...     "retrieved_context_ids": ["B001", "B002"],
        ...     "retrieved_context": ["Wireless headphones", "Bluetooth speaker"],
        ...     "similarity_scores": [0.95, 0.89],
        ... }
        >>> print(process_context(context))
        - B001: Wireless headphones
        - B002: Bluetooth speaker
    """
    formatted_context = ""
    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- {id}:, rating:{rating}, description:{chunk}\n"

    return formatted_context


@traceable(name="build-prompt", run_type="prompt")
def build_prompt(preprocessed_context, question):
    """
    Construct a prompt for the LLM that includes product context and user question.

    Creates a structured prompt that instructs the LLM to act as a shopping assistant
    and answer questions based only on the provided product information.

    Args:
        preprocessed_context (str): Formatted product information from process_context()
        question (str): The user's question about products

    Returns:
        str: Complete prompt string ready to be sent to the LLM

    Example:
        >>> context = "- B001: Wireless headphones\\n- B002: Bluetooth speaker\\n"
        >>> question = "What audio products are available?"
        >>> prompt = build_prompt(context, question)
        >>> print("shopping assistant" in prompt)
        True
    """
    template = prompt_template_config("prompts/retrieval_generation.yaml", "retrieval_generation")
    rendered_prompt = template.render(preprocessed_context=preprocessed_context, question=question)

    return rendered_prompt


@traceable(
    name="generate-answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o-mini"},
)
def generate_answer(prompt):
    """
    Generate a natural language answer using OpenAI's chat completion API.

    Sends the complete prompt (including context and question) to the LLM
    to generate a response based on the retrieved product information.

    Args:
        prompt (str): Complete prompt including system instructions, context, and question

    Returns:
        str: The generated answer from the LLM

    Raises:
        openai.APIError: If the API request fails

    Example:
        >>> prompt = "You are a shopping assistant...\\n\\nContext: ...\\n\\nQuestion: ..."
        >>> answer = generate_answer(prompt)
        >>> print(type(answer))
        <class 'str'>

    """
    client = instructor.from_openai(openai.OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        response_model=RAGGenerationResponseWithReferences,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
    return response


@traceable(name="rag-pipeline")
def rag_pipeline(question, qdrant_client, top_k=5):
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline for answering questions about products.

    This function implements a full RAG pipeline that:
    1. Retrieves relevant product information from a Qdrant vector database
    2. Processes the retrieved context into a formatted string
    3. Builds a prompt with the context and question
    4. Generates an answer using OpenAI's GPT model

    Args:
        question (str): The user's question about products
        top_k (int, optional): Number of most similar products to retrieve. Defaults to 5.

    Returns:
        str: Generated answer based on retrieved product information
    """
    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    processed_context = process_context(retrieved_context)
    prompt = build_prompt(processed_context, question)
    answer = generate_answer(prompt)

    final_result = {
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"],
    }

    return final_result


def rag_pipeline_wrapper(question: str, top_k: int = 5) -> dict[str, any]:
    """Execute RAG pipeline and enrich results with product metadata for API response.

    This wrapper function extends the base RAG pipeline by:
    1. Initializing a connection to the Qdrant vector database
    2. Executing the RAG pipeline to get answer and references
    3. Enriching reference items with additional metadata (images, prices)
    4. Formatting the response for API consumption

    The function queries Qdrant for each referenced product to fetch complete
    metadata including product images and pricing information that wasn't
    included in the initial vector search results.

    Args:
        question: The user's question about products to be answered using RAG
        top_k: Number of most similar products to retrieve from vector database.
            Defaults to 5.

    Returns:
        A dictionary containing:
            - answer (str): Generated natural language answer to the question
            - used_context (list[dict]): List of product references used, each with:
                - image_url (str): URL to product image
                - price (float): Product price
                - description (str): Product description
    """
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result = rag_pipeline(question, qdrant_client, top_k)

    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in result.get("references", []):
        payload = (
            qdrant_client.query_points(
                collection_name="Amazon-items-collection-01-hybrid-search",
                query=dummy_vector,
                using="text-embedding-3-small",
                limit=1,
                with_payload=True,
                query_filter=Filter(must=[FieldCondition(key="parent_asin", match=MatchValue(value=item.id))]),
            )
            .points[0]
            .payload
        )
        image_url = payload.get("images", "")
        price = payload.get("price")
        used_context.append({"image_url": image_url, "price": price, "description": item.description})

    return {"answer": result["answer"], "used_context": used_context}
