"""
Zuora Help Agent - FastAPI Backend
Production-ready RAG-based AI assistant for Zuora documentation.
"""

from typing import List, Optional, Dict, Any
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from config import settings
from embeddings import EmbeddingGenerator, EmbeddingError
from faiss_loader import FAISSIndexLoader
from prompts import ZUORA_EXPERT_PROMPT, create_rag_prompt

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances (initialized at startup)
embedder: Optional[EmbeddingGenerator] = None
index_loader: Optional[FAISSIndexLoader] = None
llm_client: Optional[Any] = None


# Pydantic Models
class AskRequest(BaseModel):
    """Request model for /ask endpoint"""

    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for tracking")
    context: Optional[Dict] = Field(default_factory=dict, description="Additional context")

    @validator("question")
    def validate_question(cls, v):
        """Validate question is not empty or whitespace"""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or whitespace")
        return v.strip()


class Source(BaseModel):
    """Source document model"""

    title: str = Field(..., description="Document title")
    source: str = Field(..., description="Source file or URL")
    url: Optional[str] = Field(None, description="URL to source document")
    chunk_index: Optional[int] = Field(None, description="Chunk index in source")
    similarity: Optional[float] = Field(None, description="Similarity score")


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""

    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="Source documents")
    conversation_id: str = Field(..., description="Conversation ID")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment name")
    rag_enabled: bool = Field(..., description="Whether RAG pipeline is initialized")


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("Starting Zuora Help Agent...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    global embedder, index_loader, llm_client

    try:
        # Initialize embedding generator
        logger.info("Initializing embedding generator...")
        embedder = EmbeddingGenerator(
            provider=settings.EMBEDDING_PROVIDER,
            model=settings.EMBEDDING_MODEL,
        )
        logger.info(f"✅ Embeddings: {embedder.get_provider_info()}")

        # Initialize FAISS index loader
        logger.info("Loading FAISS index...")
        try:
            index_loader = FAISSIndexLoader(
                index_path="../data/faiss.index",
                metadata_path="../data/metadata.json",
            )
            stats = index_loader.get_stats()
            logger.info(f"✅ FAISS Index: {stats['num_chunks']} chunks, {stats['dimension']}d")
        except FileNotFoundError as e:
            logger.warning(f"⚠️  FAISS index not found: {e}")
            logger.warning("RAG will operate in limited mode without vector search")
            index_loader = None

        # Initialize LLM client
        logger.info("Initializing LLM client...")
        if settings.LLM_PROVIDER == "anthropic":
            import anthropic
            llm_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info(f"✅ LLM: Anthropic Claude ({settings.LLM_MODEL})")
        elif settings.LLM_PROVIDER == "openai":
            import openai
            llm_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(f"✅ LLM: OpenAI ({settings.LLM_MODEL})")
        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")

        logger.info("🚀 Zuora Help Agent started successfully!")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        logger.warning("Service will start but RAG features may be limited")

    yield

    # Shutdown
    logger.info("Shutting down Zuora Help Agent...")
    # Cleanup if needed


# Create FastAPI app
app = FastAPI(
    title="Zuora Help Agent API",
    description="RAG-based AI assistant for Zuora product documentation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(",") if settings.ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Zuora Help Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns service status and component availability.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.ENVIRONMENT,
        rag_enabled=all([embedder is not None, llm_client is not None]),
    )


@app.post("/ask", response_model=AskResponse, tags=["Agent"])
async def ask_question(request: AskRequest):
    """
    Ask a question to the Zuora Help Agent.

    This endpoint implements RAG (Retrieval-Augmented Generation):
    1. Generates embedding for the question
    2. Retrieves relevant documentation from FAISS index
    3. Generates contextual answer using LLM (Claude or GPT)
    4. Returns answer with source citations

    Args:
        request: AskRequest with question and optional context

    Returns:
        AskResponse with answer, sources, and metadata

    Raises:
        HTTPException: If question is invalid or processing fails
    """
    # Validate question
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Check if RAG components are initialized
    if not llm_client:
        raise HTTPException(
            status_code=503,
            detail="LLM service not initialized. Check API keys and configuration.",
        )

    if not embedder:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized. Check configuration.",
        )

    try:
        logger.info(f"Processing question: {request.question[:100]}...")

        # Step 1: Generate query embedding
        try:
            query_embedding = await embedder.generate_query_embedding(request.question)
            logger.debug(f"Generated query embedding: shape={query_embedding.shape}")
        except EmbeddingError as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

        # Step 2: Retrieve relevant documents
        relevant_docs = []
        sources = []

        if index_loader:
            try:
                results = index_loader.search(
                    query_embedding=query_embedding,
                    top_k=settings.TOP_K_RESULTS,
                    threshold=settings.SIMILARITY_THRESHOLD,
                )

                logger.info(f"Retrieved {len(results)} relevant documents")

                # Format results for context
                for result in results:
                    relevant_docs.append({
                        "content": result["chunk_text"],
                        "metadata": {
                            "source": result["source"],
                            "title": result["title"],
                        },
                    })

                    # Create source objects
                    sources.append(Source(
                        title=result["title"],
                        source=result["source"],
                        url=result.get("url"),
                        chunk_index=result.get("chunk_index"),
                        similarity=result.get("similarity"),
                    ))

            except Exception as e:
                logger.error(f"Document retrieval failed: {e}")
                # Continue without context if retrieval fails
                logger.warning("Proceeding without retrieved context")

        # Step 3: Generate answer using LLM
        try:
            if relevant_docs:
                # Use RAG with retrieved context
                user_prompt = create_rag_prompt(
                    question=request.question,
                    context_docs=relevant_docs,
                )
            else:
                # Fallback: answer without context
                user_prompt = f"Question: {request.question}\n\nPlease answer based on your knowledge of Zuora products."
                logger.warning("No context available, using LLM knowledge only")

            # Call LLM based on provider
            if settings.LLM_PROVIDER == "anthropic":
                response = llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                    system=ZUORA_EXPERT_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                answer = response.content[0].text

            elif settings.LLM_PROVIDER == "openai":
                response = llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                    messages=[
                        {"role": "system", "content": ZUORA_EXPERT_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                answer = response.choices[0].message.content

            else:
                raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")

            logger.info(f"Generated answer ({len(answer)} chars)")

        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Answer generation failed: {str(e)}",
            )

        # Step 4: Calculate confidence (simple heuristic)
        confidence = min(1.0, len(relevant_docs) / settings.TOP_K_RESULTS) if relevant_docs else 0.0

        # Step 5: Return response
        return AskResponse(
            answer=answer,
            sources=sources,
            conversation_id=request.conversation_id or f"conv_{hash(request.question)}",
            confidence=confidence,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
