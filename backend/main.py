"""
Zuora Help Agent - FastAPI Application
A RAG-based help agent for Zuora product documentation and support.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Zuora Help Agent",
    description="RAG-based AI assistant for Zuora product documentation and support",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class AskRequest(BaseModel):
    """Request model for /ask endpoint"""

    question: str
    conversation_id: Optional[str] = None
    context: Optional[dict] = None


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""

    answer: str
    sources: list[str] = []
    conversation_id: str
    confidence: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""

    status: str
    version: str
    environment: str


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Zuora Help Agent",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns service status and version information.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.ENVIRONMENT,
    )


@app.post("/ask", response_model=AskResponse, tags=["Agent"])
async def ask_question(request: AskRequest):
    """
    Ask a question to the Zuora Help Agent.

    This endpoint will use RAG (Retrieval-Augmented Generation) to:
    1. Retrieve relevant documentation from the vector store
    2. Generate a contextual answer using LLM
    3. Return answer with source citations

    Args:
        request: AskRequest containing the question and optional context

    Returns:
        AskResponse with answer, sources, and conversation tracking

    Raises:
        HTTPException: If the question is empty or processing fails
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # TODO: Implement RAG pipeline
        # 1. Generate embeddings for the question
        # 2. Retrieve relevant documents from vector store
        # 3. Generate answer using LLM with retrieved context
        # 4. Track conversation history

        logger.info(f"Processing question: {request.question[:100]}...")

        # Placeholder response
        return AskResponse(
            answer="This endpoint will be implemented with RAG capabilities. Your question was: "
            + request.question,
            sources=["placeholder_source.pdf"],
            conversation_id=request.conversation_id or "new_conversation_id",
            confidence=0.0,
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Zuora Help Agent...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    # TODO: Initialize vector store, LLM client, etc.


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Zuora Help Agent...")
    # TODO: Cleanup connections, save state, etc.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
