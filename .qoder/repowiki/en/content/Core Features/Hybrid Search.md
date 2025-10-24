# Hybrid Search

<cite>
**Referenced Files in This Document**   
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Hybrid Search Architecture](#hybrid-search-architecture)
3. [Implementation of Dual Retrieval](#implementation-of-dual-retrieval)
4. [RRF Fusion Mechanism](#rrf-fusion-mechanism)
5. [Performance Considerations](#performance-considerations)
6. [Error Handling and Debugging](#error-handling-and-debugging)
7. [Conclusion](#conclusion)

## Introduction
The Hybrid Search feature implements a dual-retrieval strategy combining semantic (vector) and keyword (BM25) search methods to improve product retrieval relevance. This approach leverages both intent-based understanding through embeddings and exact keyword matching through BM25, fused using Reciprocal Rank Fusion (RRF) to produce optimal results. The system is designed to handle Amazon product queries with high precision and recall.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L78-L153)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L194-L244)

## Hybrid Search Architecture

```mermaid
graph TD
UserQuery["User Query: 'wireless headphones'"] --> EmbeddingAPI
UserQuery --> BM25Search
EmbeddingAPI["OpenAI API<br/>Embedding"] --> SemanticSearch
BM25Search["BM25 Token Matching"] --> BM25Index
SemanticSearch["Semantic Search<br/>(Vector Index)"] --> QdrantEngine
BM25Index["BM25 Search<br/>(Sparse Index)"] --> QdrantEngine
QdrantEngine["Qdrant Query Engine"] --> RRFusion["RRF Fusion<br/>(Re-ranking)"]
RRFusion --> TopResults["Top-k Results<br/>(Default: 5)"]
```

**Diagram sources**
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L588-L631)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L99-L118)

## Implementation of Dual Retrieval

The `retrieve_data` function in retrieval_generation.py performs dual prefetch queries against Qdrant using both text embeddings and BM25 token matching. The implementation uses Qdrant's Prefetch functionality to execute both retrieval methods in parallel.

```mermaid
flowchart TD
Start["retrieve_data(query, qdrant_client, k=5)"] --> EmbeddingStep["Generate embedding<br/>via OpenAI API"]
EmbeddingStep --> PrefetchSetup["Configure Prefetch queries"]
PrefetchSetup --> SemanticPrefetch["Prefetch 1:<br/>- query: embedding vector<br/>- using: text-embedding-3-small<br/>- limit: 20"]
PrefetchSetup --> BM25Prefetch["Prefetch 2:<br/>- query: Document(text=query)<br/>- using: bm25<br/>- limit: 20"]
SemanticPrefetch --> QdrantCall
BM25Prefetch --> QdrantCall
QdrantCall["qdrant_client.query_points()<br/>with FusionQuery(fusion='rrf')"] --> ResultProcessing
ResultProcessing["Process results and extract<br/>context IDs, descriptions,<br/>ratings, and scores"] --> ReturnResults
ReturnResults["Return structured results<br/>or None on failure"]
```

**Diagram sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L81-L117)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L37-L79)

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L78-L153)

## RRF Fusion Mechanism

Reciprocal Rank Fusion (RRF) combines results from both retrieval methods to improve relevance by aggregating rankings from semantic and keyword searches. The RRF score is calculated using the formula: `RRF_score(p) = 1/(60 + semantic_rank(p)) + 1/(60 + bm25_rank(p))`.

```mermaid
classDiagram
class FusionQuery {
+fusion : str
+params : Optional[Dict]
}
class Prefetch {
+query : Union[Vector, Document]
+using : str
+limit : int
+prefetch : List[Prefetch]
}
class QdrantClient {
+query_points(collection_name, prefetch, query, limit)
}
FusionQuery <|-- RRFQuery : "specialization"
Prefetch <|-- SemanticPrefetch : "vector-based"
Prefetch <|-- KeywordPrefetch : "BM25-based"
QdrantClient --> FusionQuery : "uses"
QdrantClient --> Prefetch : "uses multiple"
```

**Diagram sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L115)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L268-L297)

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L104-L112)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L588-L631)

## Performance Considerations

The hybrid search implementation balances query latency, result diversity, and retrieval quality through careful configuration of k-values and limit parameters. The system retrieves top-20 results from each method before applying RRF fusion to produce the final top-k results (default: 5).

**Key Performance Parameters:**
- **Prefetch limit**: 20 results per method (semantic and BM25)
- **Final limit (k)**: Configurable, default 5
- **Embedding model**: text-embedding-3-small (1536 dimensions)
- **Fusion method**: RRF with k=60 parameter

The dual retrieval approach increases query complexity but significantly improves result relevance by combining the strengths of both semantic understanding and keyword matching. The system is optimized to minimize the performance impact through parallel execution of prefetch queries.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L104-L112)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L588-L631)

## Error Handling and Debugging

The system implements comprehensive error handling for various failure scenarios, including embedding generation failures and empty result sets. All pipeline steps are instrumented with LangSmith tracing for debugging and performance analysis.

**Common Issues and Solutions:**
- **Empty results**: Handled by returning empty context lists with appropriate logging
- **Embedding failures**: Detected and logged, with function returning None
- **Qdrant connection issues**: Caught and logged as UnexpectedResponse
- **Rate limiting**: OpenAI rate limit errors are specifically handled

```mermaid
sequenceDiagram
participant User
participant API as FastAPI Backend
participant Qdrant
participant OpenAI
User->>API : Submit query
API->>OpenAI : Request embedding
alt Embedding Success
OpenAI-->>API : Return embedding vector
API->>Qdrant : Dual prefetch query
Qdrant-->>API : Return fused results
API->>User : Return top-k products
else Embedding Failure
OpenAI-->>API : Return error
API->>User : Return error response
end
alt No Results Found
Qdrant-->>API : Empty results
API->>User : Return empty context
end
```

**Diagram sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L81-L153)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L37-L79)

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L81-L153)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L37-L79)

## Conclusion
The Hybrid Search implementation effectively combines semantic and keyword retrieval methods using RRF fusion to deliver high-quality product search results. By leveraging both vector embeddings and BM25 matching, the system captures both intent-based relevance and exact keyword matches, producing more comprehensive and accurate results than either method alone. The architecture is robust, observable through LangSmith tracing, and handles edge cases gracefully, making it suitable for production deployment in e-commerce search scenarios.