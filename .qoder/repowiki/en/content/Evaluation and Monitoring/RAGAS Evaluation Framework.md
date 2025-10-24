# RAGAS Evaluation Framework

<cite>
**Referenced Files in This Document**   
- [eval_retriever.py](file://evals/eval_retriever.py)
- [retrieval_generation.py](file://src/api/rag/retrieval_generation.py)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md)
- [04-RAG-Evals.ipynb](file://notebooks/phase_2/04-RAG-Evals.ipynb)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [RAGAS Metrics Implementation](#ragas-metrics-implementation)
3. [Model Integration with Langchain Wrappers](#model-integration-with-langchain-wrappers)
4. [SingleTurnSample Construction](#singleturnsample-construction)
5. [Asynchronous Scoring Functions](#asynchronous-scoring-functions)
6. [Metric Interpretation and Troubleshooting](#metric-interpretation-and-troubleshooting)
7. [Conclusion](#conclusion)

## Introduction
The RAGAS Evaluation Framework is designed to assess the quality of the Retrieval-Augmented Generation (RAG) system's retrieval and generation components within the AI-Powered Amazon Product Assistant. This framework leverages the RAGAS library to compute key metrics such as IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, and ResponseRelevancy. These metrics are integrated into the evaluation pipeline via Langsmith, using the `eval_retriever.py` script to systematically evaluate the performance of the RAG pipeline. The framework ensures that both the relevance of retrieved context and the accuracy of generated responses are rigorously measured, providing actionable insights for system improvement.

**Section sources**
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L1050-L1087)

## RAGAS Metrics Implementation
The RAGAS metrics are implemented in `eval_retriever.py` to evaluate the quality of the RAG system's outputs. The metrics are categorized into retrieval and generation types. Retrieval metrics, such as IDBasedContextPrecision and IDBasedContextRecall, assess the relevance of the retrieved context by comparing the retrieved context IDs with the reference context IDs from the evaluation dataset. Generation metrics, including Faithfulness and ResponseRelevancy, evaluate the generated answer's accuracy and relevance to the user's question. Faithfulness ensures that the generated answer is grounded in the retrieved context, preventing hallucinations, while ResponseRelevancy measures how well the answer addresses the user's query. These metrics are computed using the `SingleTurnSample` class, which encapsulates the necessary data for evaluation.

**Section sources**
- [eval_retriever.py](file://evals/eval_retriever.py#L0-L79)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L1050-L1087)

## Model Integration with Langchain Wrappers
The integration of LangchainLLMWrapper and LangchainEmbeddingsWrapper with the GPT-4.1-mini and text-embedding-3-small models is crucial for the computation of RAGAS metrics. In `eval_retriever.py`, the `ragas_llm` variable is initialized with `LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))`, enabling the use of the GPT-4.1-mini model for generating responses and evaluating faithfulness. Similarly, `ragas_embeddings` is set up with `LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))`, which provides the necessary embeddings for the ResponseRelevancy metric. This integration allows the RAGAS framework to leverage state-of-the-art language models and embedding techniques to ensure accurate and reliable metric computation.

**Section sources**
- [eval_retriever.py](file://evals/eval_retriever.py#L0-L79)

## SingleTurnSample Construction
The `SingleTurnSample` is constructed from run outputs and evaluation examples to facilitate the computation of RAGAS metrics. In the asynchronous scoring functions, such as `ragas_faithfulness` and `ragas_responce_relevancy`, the `SingleTurnSample` is created using the user input (question), the generated response (answer), and the retrieved context. For retrieval metrics like `ragas_context_precision_id_based` and `ragas_context_recall_id_based`, the sample is constructed using the retrieved context IDs and the reference context IDs from the evaluation example. This structured approach ensures that the evaluation process has access to all necessary data points, enabling a comprehensive assessment of the RAG system's performance.

**Section sources**
- [eval_retriever.py](file://evals/eval_retriever.py#L0-L79)
- [04-RAG-Evals.ipynb](file://notebooks/phase_2/04-RAG-Evals.ipynb#L0-L449)

## Asynchronous Scoring Functions
The asynchronous scoring functions, including `ragas_faithfulness`, `ragas_responce_relevancy`, `ragas_context_precision_id_based`, and `ragas_context_recall_id_based`, play a critical role in computing per-query metrics. These functions are designed to be non-blocking, allowing for efficient evaluation of multiple queries in parallel. Each function initializes a scorer with the appropriate model or embeddings, constructs a `SingleTurnSample` from the run and example data, and then computes the score using the `single_turn_ascore` method. The results are aggregated by the `ls_client.evaluate` function, which orchestrates the evaluation process and reports the outcomes to Langsmith. This asynchronous design ensures that the evaluation pipeline can handle large datasets efficiently, providing timely feedback on the RAG system's performance.

**Section sources**
- [eval_retriever.py](file://evals/eval_retriever.py#L0-L79)

## Metric Interpretation and Troubleshooting
Interpreting the RAGAS metric scores is essential for identifying retrieval weaknesses or generation inaccuracies. High values for IDBasedContextPrecision and IDBasedContextRecall indicate that the retrieved context is highly relevant and comprehensive, while low values suggest issues with the retrieval mechanism. Similarly, high Faithfulness and ResponseRelevancy scores reflect accurate and relevant generated answers. Common issues, such as missing context IDs or mismatched reference sets, can lead to failed evaluations. To troubleshoot these issues, it is recommended to verify the integrity of the evaluation dataset, ensure that the context IDs are correctly mapped, and validate the alignment between the retrieved and reference contexts. Regular monitoring and iterative refinement of the RAG pipeline based on these metrics can significantly enhance the system's overall performance.

**Section sources**
- [eval_retriever.py](file://evals/eval_retriever.py#L0-L79)
- [ARCHITECTURE.md](file://documentation/ARCHITECTURE.md#L1087-L1104)

## Conclusion
The RAGAS Evaluation Framework provides a robust and systematic approach to assessing the quality of the RAG system's retrieval and generation components. By leveraging key metrics such as IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, and ResponseRelevancy, the framework ensures that both the relevance of retrieved context and the accuracy of generated responses are rigorously evaluated. The integration of LangchainLLMWrapper and LangchainEmbeddingsWrapper with advanced models like GPT-4.1-mini and text-embedding-3-small further enhances the reliability of the evaluation process. Through the use of asynchronous scoring functions and the construction of `SingleTurnSample` from run outputs and evaluation examples, the framework delivers efficient and comprehensive performance assessments. Addressing common issues and interpreting metric scores effectively can lead to continuous improvement of the RAG system, ultimately enhancing the user experience.