# Performance Tracking

<cite>
**Referenced Files in This Document**   
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md)
</cite>

## Table of Contents
1. [Token Usage Monitoring](#token-usage-monitoring)
2. [Latency and Retrieval Efficiency](#latency-and-retrieval-efficiency)
3. [Cost Calculation and Optimization](#cost-calculation-and-optimization)
4. [Logging and Failure Tracking](#logging-and-failure-tracking)
5. [Performance Optimization Strategies](#performance-optimization-strategies)
6. [Cost-Performance Trade-off Analysis](#cost-performance-trade-off-analysis)

## Token Usage Monitoring

The RAG pipeline captures detailed token usage metrics through the `@traceable` decorator integrated with LangSmith. Each function that interacts with OpenAI APIs logs token consumption directly to the trace metadata for comprehensive observability.

In the `get_embedding` function, when generating embeddings using the `text-embedding-3-small` model, the system extracts token usage from the OpenAI API response and attaches it to the LangSmith trace:

```python
current_run.metadata["usage_metadata"] = {
    "input_tokens": response.usage.prompt_tokens,
    "total_tokens": response.usage.total_tokens
}
```

Similarly, in the `generate_answer` function, which uses the `gpt-4.1-mini` model, both input and output tokens are captured from the `raw_response.usage` object:

```python
current_run.metadata["usage_metadata"] = {
    "input_tokens": raw_response.usage.prompt_tokens,
    "output_tokens": raw_response.usage.completion_tokens,
    "total_tokens": raw_response.usage.total_tokens
}
```

This metadata is automatically captured under the `"usage_metadata"` key in LangSmith traces, enabling detailed analysis of token consumption across all pipeline stages. The trace hierarchy shows how embedding and LLM calls are nested within the overall `rag_pipeline`, allowing granular cost attribution.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L34-L71)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L233-L273)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L400-L425)

## Latency and Retrieval Efficiency

The system monitors retrieval efficiency through similarity scores and latency metrics captured by LangSmith traces. The `retrieve_data` function performs hybrid search using both semantic and BM25 methods with Reciprocal Rank Fusion (RRF), returning similarity scores for each retrieved result.

These similarity scores are stored in the pipeline output under the `"similarity_scores"` key and can be used to assess the quality of vector search results. Higher similarity scores indicate better alignment between the query embedding and stored product vectors, serving as a proxy for retrieval relevance.

Latency is automatically tracked by LangSmith for each `@traceable` function. The trace hierarchy breaks down execution time across pipeline stages:
- Embedding generation (`embed_query`)
- Data retrieval (`retrieve_data`)
- Context formatting (`format_retrieved_context`)
- Prompt building (`build_prompt`)
- Answer generation (`generate_answer`)

This breakdown allows identification of performance bottlenecks, such as slow embedding calls or high-latency LLM generations. The `retrieve_data` function also logs retrieval duration, which can be correlated with the number of results and query complexity.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L78-L153)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L279-L328)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L268-L297)

## Cost Calculation and Optimization

Cost calculation is based on token consumption for both the `text-embedding-3-small` and `gpt-4.1-mini` models. The system captures token usage in LangSmith traces, enabling post-execution cost analysis.

For the `text-embedding-3-small` model, the cost is $0.00002 per 1K tokens, with only input tokens being charged. For `gpt-4.1-mini`, input tokens cost $0.150 per million and output tokens cost $0.600 per million. These rates can be applied to the captured `input_tokens`, `output_tokens`, and `total_tokens` values to calculate per-request costs.

The trace metadata includes provider and model information:
```python
metadata={
    "ls_provider": "openai",
    "ls_model_name": "text-embedding-3-small"
}
```

This enables filtering and aggregation of costs by model type in LangSmith. By analyzing historical traces, teams can estimate monthly costs based on query volume and average token usage, and identify high-cost queries for optimization.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L34-L71)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L233-L273)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L453-L492)

## Logging and Failure Tracking

The system uses Python's `logging` module to track pipeline execution and potential failures. Each major function includes structured logging statements that capture entry, success, and error conditions.

The `logger` instance is used consistently across the pipeline:
```python
logger.info("Generating answer using GPT-4.1-mini")
```

Error conditions are logged with appropriate context:
```python
logger.error(f"OpenAI API error generating answer: {e}")
```

The logging format includes timestamps, logger names, and log levels, with output directed to stdout for containerized logging. The `RequestIDMiddleware` adds a unique `X-Request-ID` header to each request, enabling correlation of log entries across the system.

Critical failure points are monitored:
- OpenAI API errors and rate limits
- Qdrant connection issues
- Empty retrieval results
- Prompt building failures
- Answer generation failures

These logs are captured in LangSmith traces, providing a complete audit trail for debugging and performance analysis.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L17)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L233-L273)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L78-L153)

## Performance Optimization Strategies

Several strategies can be employed to optimize RAG pipeline performance and reduce costs:

**Adjusting top_k parameter**: The `top_k` parameter in `rag_pipeline` controls the number of retrieved results. Reducing `top_k` decreases context size and subsequent LLM input tokens, lowering costs and potentially improving response quality by reducing noise.

**Caching embeddings**: Frequently asked queries could have their embeddings cached to avoid repeated calls to the `text-embedding-3-small` API, reducing both latency and token costs.

**Tuning prompt length**: The context formatting in `process_context` directly impacts prompt size. Optimizing the format string to include only essential information reduces input tokens to the LLM. The YAML-based prompt template in `retrieval_generation.yaml` can be modified to be more concise while maintaining instruction clarity.

**Hybrid search configuration**: The RRF fusion parameters and individual search limits (currently 20 for both semantic and BM25) can be tuned to balance retrieval quality and performance.

These optimizations can be tested and compared using LangSmith's A/B testing capabilities, measuring their impact on both cost metrics and evaluation scores.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L279-L328)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L160-L192)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L199-L225)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L322-L363)

## Cost-Performance Trade-off Analysis

Evaluation results and trace data enable informed decisions about cost-performance trade-offs. By analyzing LangSmith traces, teams can correlate token usage and latency with RAGAS evaluation metrics such as faithfulness, relevancy, and context precision.

High-cost queries can be identified and analyzed for optimization opportunities. For example, queries with very long context inputs might be simplified through better retrieval filtering or more concise context formatting.

The system's observability stack allows comparison of different configurations:
- Different `top_k` values and their impact on answer quality vs. cost
- Prompt template variations and their effect on output token usage
- Model parameter tuning (temperature, etc.) and its influence on response length

This data-driven approach enables systematic optimization of the RAG pipeline, balancing user experience requirements with operational costs. The trace data serves as a foundation for capacity planning and budget forecasting based on expected query volumes and average token consumption.

**Section sources**
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L279-L328)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py#L233-L273)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L492-L525)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L995-L1012)