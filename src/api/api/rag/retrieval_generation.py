import openai
from openai.types.shared import responses_model
from qdrant_client import QdrantClient 
from langsmith import traceable, get_current_run_tree


@traceable(
    name="embed-query",
    run_type="embedding",
    metadata={"ls_provider":"openai", "ls_model_name":"text-embedding-3-small"}
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
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    # to print llm usage on langsmit UI
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens":response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens 
        }
    return response.data[0].embedding

@traceable(
    name="retrieve-data",
    run_type="retriever"
)
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
    query_embeddings = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embeddings,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores  = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }


@traceable(
    name="format-retrieved-context",
    run_type="prompt"
)
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
        ...     "similarity_scores": [0.95, 0.89]
        ... }
        >>> print(process_context(context))
        - B001: Wireless headphones
        - B002: Bluetooth speaker
    """
    formatted_context = ""
    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- {id}: {chunk} \n"

    return formatted_context


@traceable(
    name="build-prompt",
    run_type="prompt"
)
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
    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products.

Context:
{preprocessed_context}

Question:
{question}
"""

    return prompt


@traceable(
    name="generate-answer",
    run_type="llm",
    metadata={"ls_provider":"openai", "ls_model_name":"gpt-4o-mini"}
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

    Note:
        - Uses gpt-4o-mini model for cost-effective inference
        - Temperature set to 0.5 for balanced creativity and consistency
        - Prompt is sent as a system message for stronger instruction-following
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens":response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens 
        }
    return response.choices[0].message.content
    
@traceable(
    name="rag-pipeline"
)
def rag_pipeline(question, top_k=5):
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
        
    Example:
        >>> answer = rag_pipeline("What kind of earphones can I get?", top_k=10)
        >>> print(answer)
    """
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    retrieved_context = retrieve_data(question, qdrant_client, top_k)
    processed_context = process_context(retrieved_context)
    prompt = build_prompt(processed_context, question)
    answer = generate_answer(prompt)

    final_result = {

        "answer" : answer,
        "question":question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"]
    } 
    
    return final_result 
   

