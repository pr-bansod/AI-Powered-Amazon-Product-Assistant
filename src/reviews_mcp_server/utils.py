from typing import Dict, List

import openai
from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.http.models import Query
from qdrant_client.models import FieldCondition, Filter, FusionQuery, MatchAny, Prefetch


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate embedding vector for text using OpenAI embedding model."""
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    return response.data[0].embedding


def retrieve_reviews_data(query: str, item_list: List[str], k: int = 5) -> Dict[str, List]:
    """Retrieve product reviews using filtered hybrid search for specific items."""
    query_embedding = get_embedding(query)

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-reviews",
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=item_list
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["text"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }


@traceable(
    name="format_retrieved_reviews_context",
    run_type="prompt"
)
def process_reviews_context(context: Dict[str, List]) -> str:
    """Format retrieved review context into readable string format."""
    formatted_context = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- ID: {id}, review: {chunk}\n"

    return formatted_context