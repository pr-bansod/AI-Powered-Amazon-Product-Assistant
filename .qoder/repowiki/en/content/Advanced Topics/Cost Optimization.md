# Cost Optimization

<cite>
**Referenced Files in This Document**   
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py)
- [prompt_management.py](file://src/api/rag/utils/prompt_management.py)
- [retrieval_generation.yaml](file://src/api/rag/prompts/retrieval_generation.yaml)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md)
</cite>

## Table of Contents
1. [Token Usage Tracking with LangSmith](#token-usage-tracking-with-langsmith)
2. [Model Selection Trade-offs](#model-selection-trade-offs)
3. [Caching Mechanisms](#caching-mechanisms)
4. [Prompt Engineering for Token Efficiency](#prompt-engineering-for-token-efficiency)
5. [Retrieval Parameters and Downstream Costs](#retrieval-parameters-and-downstream-costs)
6. [Optimizing Retrieval-Generation Balance](#optimizing-retrieval-generation-balance)
7. [Rate Limit and API Error Handling](#rate-limit-and-api-error-handling)
8. [Cost Monitoring and Budget Alerts](#cost-monitoring-and-budget-alerts)

## Token Usage Tracking with LangSmith

The system implements comprehensive token usage tracking through LangSmith's metadata system across both embedding and LLM generation steps. For embedding operations using `text-embedding-3-small`, the system captures `input_tokens` and `total_tokens` from the OpenAI API response and logs them to LangSmith metadata within the traceable span. During LLM generation with `gpt-4.1-mini`, the system captures detailed token metrics including `input_tokens`, `output_tokens`, and `total_tokens` from the raw response usage data. These metrics are stored in the `usage_metadata` field of the current run's metadata, enabling granular cost analysis per request. The tracing hierarchy in LangSmith provides a complete breakdown of token consumption across all pipeline steps, allowing for precise attribution of costs to specific components such as embedding, retrieval, and generation.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L34-L71)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L233-L273)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L492-L525)

## Model Selection Trade-offs

The system currently utilizes `gpt-4.1-mini` for LLM generation, which offers a favorable cost-performance balance at $0.150 per million input tokens and $0.600 per million output tokens. This model provides strong reasoning capabilities while maintaining lower costs compared to larger models like GPT-4. The embedding model `text-embedding-3-small` is used at $0.00002 per thousand tokens, providing efficient vectorization of queries. The architecture supports alternative LLM providers through configuration, including Groq and Google APIs, allowing for cost-based model switching. The trade-off between model size and cost is particularly evident in the generation step, where larger models would provide potentially better quality but at significantly higher token costs, especially for output tokens which are typically more expensive.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L233-L273)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L453-L492)

## Caching Mechanisms

While the current implementation does not include response caching, the architecture supports the implementation of caching mechanisms to avoid redundant API calls. The system could implement Redis-based caching of answers for identical queries, particularly for frequently asked questions about popular products. The `RequestIDMiddleware` provides a foundation for request identification that could be extended to support cache lookups. Additionally, the structured output format with `RAGResponse` and consistent request IDs enables deterministic caching strategies. Future implementation could cache both the final responses and intermediate results such as embeddings, with appropriate TTL settings to ensure freshness of product information.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L279-L328)
- [endpoints.py](file://src/api/api/endpoints.py#L47-L73)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L1290-L1335)

## Prompt Engineering for Token Efficiency

The system employs YAML-based prompt management with Jinja2 templates to enable systematic prompt engineering for token efficiency. The `retrieval_generation.yaml` template is designed to be concise while providing clear instructions to the LLM. The prompt structure minimizes unnecessary text by using bullet points and direct instructions, reducing input token consumption. The template separates context formatting from prompt construction, allowing optimization of each component independently. By using structured outputs with Instructor and Pydantic, the system reduces the need for extensive post-processing prompts, as the LLM directly generates validated JSON output. This approach minimizes both input tokens (by avoiding complex parsing instructions) and output tokens (by eliminating the need for reformatting).

**Section sources**
- [retrieval_generation.yaml](file://src/api/rag/prompts/retrieval_generation.yaml)
- [prompt_management.py](file://src/api/rag/utils/prompt_management.py)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L199-L225)

## Retrieval Parameters and Downstream Costs

Retrieval parameters significantly impact downstream token costs through the `top_k` parameter that controls the number of results retrieved. The system uses hybrid search with Reciprocal Rank Fusion (RRF), retrieving top-20 results from both semantic and BM25 searches before fusing and limiting to the final `k` results (default: 5). The `k` parameter directly affects context size and thus input token count for the LLM generation step. Each retrieved product contributes to the context string format "- ID: {id}, rating: {rating}, description: {chunk}\n", meaning that larger `k` values linearly increase input tokens. The current default of 5 represents a balance between providing sufficient context for accurate responses while minimizing token costs. Adjusting `k` provides a direct mechanism for cost optimization based on use case requirements.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L78-L153)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L160-L192)

## Optimizing Retrieval-Generation Balance

The system optimizes the retrieval-generation balance by carefully structuring the context passed to the LLM. The retrieval step fetches product data including descriptions, ratings, and IDs, but only essential fields are included in the formatted context to minimize token usage. The context formatting process strips unnecessary metadata and presents information in a compact format. The prompt template explicitly instructs the LLM to answer based only on the provided context, preventing hallucination and reducing the need for extensive context. The structured output requirement further optimizes this balance by ensuring the LLM focuses on extracting and formatting information rather than generating creative content. This balanced approach ensures sufficient context for accurate responses while avoiding unnecessary token consumption from excessive or redundant information.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L160-L192)
- [retrieval_generation.yaml](file://src/api/rag/prompts/retrieval_generation.yaml)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L199-L225)

## Rate Limit and API Error Handling

The system currently implements fail-fast error handling for rate limits and API errors without retry mechanisms. When encountering `openai.RateLimitError`, the system logs the error and returns `None`, which propagates up to return a 500 Internal Server Error to the client. Similarly, `openai.APIError` is caught and handled by logging and returning `None`. This approach avoids wasted calls by immediately failing rather than attempting retries that might exacerbate rate limit issues. However, this strategy could be enhanced with exponential backoff retry logic to handle transient rate limit conditions more gracefully. The current implementation prioritizes system stability over availability during high load periods, ensuring that rate limit errors are surfaced immediately rather than consuming additional tokens through repeated failed attempts.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L34-L71)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L233-L273)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L492-L525)

## Cost Monitoring and Budget Alerts

The system leverages LangSmith for comprehensive cost monitoring by tracking token usage across all pipeline steps. The tracing system captures detailed metrics including input, output, and total tokens for both embedding and generation operations, enabling precise cost attribution. The hierarchical trace structure allows for breakdown of costs by component, facilitating identification of cost hotspots. While the current implementation does not include automated budget alerts, the LangSmith integration provides the foundation for setting up such monitoring. Cost metrics can be tracked over time through LangSmith's analytics capabilities, allowing teams to monitor spending patterns and identify anomalies. The system's structured logging and request tracing with unique request IDs enable detailed cost analysis and reporting, supporting the implementation of budget alerting systems based on historical usage patterns.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L34-L71)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L233-L273)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L492-L525)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L633-L674)