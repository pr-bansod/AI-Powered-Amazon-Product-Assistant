from fastapi import APIRouter, Request, HTTPException, status
import logging

from api.api.models import RAGRequest, RAGResponse, RAGUsedContext

from api.rag.retrieval_generation import rag_pipeline_wrapper


logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:
    """Process RAG query and return answer with product context.

    :param request: FastAPI Request object with request_id in state
    :type request: Request
    :param payload: RAG request containing the user query
    :type payload: RAGRequest
    :returns: RAG response with answer and used context
    :rtype: RAGResponse
    :raises HTTPException: If pipeline fails or returns no results
    """
    try:
        logger.info(f"Processing RAG request (request_id: {request.state.request_id})")

        if not payload.query or not payload.query.strip():
            logger.error("Empty query received")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )

        answer = rag_pipeline_wrapper(payload.query)

        if answer is None:
            logger.error("RAG pipeline returned None")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process query. Please try again later."
            )

        if not answer.get("answer"):
            logger.error("RAG pipeline returned empty answer")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate answer. Please try again."
            )

        logger.info(f"Successfully processed RAG request (request_id: {request.state.request_id})")

        return RAGResponse(
            request_id=request.state.request_id,
            answer=answer["answer"],
            used_context=[RAGUsedContext(**used_context) for used_context in answer.get("used_context", [])]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in RAG endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])