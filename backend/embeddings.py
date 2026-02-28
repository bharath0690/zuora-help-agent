"""
Embedding Generation and Vector Store Management

Handles document chunking, embedding generation, and vector store operations.
Supports multiple embedding providers: Voyage AI (Anthropic recommended), OpenAI, local models.

Note: Claude (Anthropic) does not provide an embedding API. This implementation uses:
- Voyage AI: Anthropic's recommended embedding partner
- OpenAI: Alternative high-quality embeddings
- Local: sentence-transformers for offline use
"""

from typing import List, Dict, Optional, Union
import logging
import asyncio
import time
from pathlib import Path
from datetime import datetime
import os

import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


class EmbeddingGenerator:
    """
    Generates embeddings for text using configured embedding model.
    Supports Voyage AI (Anthropic recommended), OpenAI, and local models.

    Environment Variables:
        VOYAGE_API_KEY: For Voyage AI embeddings (Anthropic recommended)
        OPENAI_API_KEY: For OpenAI embeddings
        EMBEDDING_PROVIDER: Provider to use (voyage/openai/local)
        EMBEDDING_MODEL: Model name (provider-specific)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        """
        Initialize embedding generator.

        Args:
            provider: Embedding provider (voyage/openai/local)
            model: Model name (provider-specific)
            api_key: API key for provider
            batch_size: Max texts per batch request
            max_retries: Max retry attempts for API failures
        """
        # Set provider
        self.provider = (provider or os.getenv("EMBEDDING_PROVIDER") or "voyage").lower()

        # Set API key
        self.api_key = api_key or os.getenv(f"{self.provider.upper()}_API_KEY")

        # Set batch size and retries
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Initialize provider-specific client
        self._init_provider(model)

        logger.info(
            f"Initialized EmbeddingGenerator: provider={self.provider}, "
            f"model={self.model}, batch_size={self.batch_size}"
        )

    def _init_provider(self, model: Optional[str] = None):
        """Initialize the embedding provider client."""
        if self.provider == "voyage":
            self._init_voyage(model)
        elif self.provider == "openai":
            self._init_openai(model)
        elif self.provider == "local":
            self._init_local(model)
        else:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                "Choose from: voyage, openai, local"
            )

    def _init_voyage(self, model: Optional[str] = None):
        """Initialize Voyage AI client (Anthropic's recommended partner)."""
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai package required for Voyage AI embeddings. "
                "Install: pip install voyageai"
            )

        if not self.api_key:
            raise ValueError(
                "VOYAGE_API_KEY environment variable required for Voyage AI. "
                "Get your key at: https://www.voyageai.com/"
            )

        self.client = voyageai.Client(api_key=self.api_key)
        self.model = model or "voyage-2"  # Default: voyage-2 (1024 dims)
        self.dimension = 1024  # voyage-2 dimension

        logger.info(f"Initialized Voyage AI client with model: {self.model}")

    def _init_openai(self, model: Optional[str] = None):
        """Initialize OpenAI client."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI embeddings. "
                "Install: pip install openai"
            )

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model or "text-embedding-3-small"  # Default: small (1536 dims)

        # Set dimension based on model
        if "text-embedding-3-large" in self.model:
            self.dimension = 3072
        else:
            self.dimension = 1536

        logger.info(f"Initialized OpenAI client with model: {self.model}")

    def _init_local(self, model: Optional[str] = None):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required for local embeddings. "
                "Install: pip install sentence-transformers"
            )

        self.model = model or "all-MiniLM-L6-v2"  # Default: fast, small (384 dims)
        self.client = SentenceTransformer(self.model)
        self.dimension = self.client.get_sentence_embedding_dimension()

        logger.info(f"Initialized local model: {self.model} (dim={self.dimension})")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((EmbeddingError, Exception)),
        reraise=True,
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with retry logic.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails after retries
        """
        try:
            if self.provider == "voyage":
                response = self.client.embed(
                    texts=texts,
                    model=self.model,
                    input_type="document",  # or "query" for search queries
                )
                return response.embeddings

            elif self.provider == "openai":
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                return [item.embedding for item in response.data]

            elif self.provider == "local":
                embeddings = self.client.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return embeddings.tolist()

        except Exception as e:
            logger.error(f"Embedding batch failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    async def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts with batching and error handling.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings (num_texts, embedding_dim)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts...")
        start_time = time.time()

        all_embeddings = []
        failed_batches = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            try:
                logger.debug(f"Processing batch {batch_num}/{total_batches}")

                # For local models, run synchronously
                if self.provider == "local":
                    embeddings = self._embed_batch(batch)
                else:
                    # For API calls, run in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None, self._embed_batch, batch
                    )

                all_embeddings.extend(embeddings)

                # Rate limiting for API providers
                if self.provider in ["voyage", "openai"] and i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)  # Small delay between batches

            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {str(e)}")
                failed_batches.append((batch_num, batch))

                # If too many failures, raise error
                if len(failed_batches) > total_batches * 0.1:  # >10% failure rate
                    raise EmbeddingError(
                        f"Too many failed batches ({len(failed_batches)}/{total_batches})"
                    )

        # Retry failed batches
        if failed_batches:
            logger.warning(f"Retrying {len(failed_batches)} failed batches...")
            for batch_num, batch in failed_batches:
                try:
                    embeddings = self._embed_batch(batch)
                    all_embeddings.extend(embeddings)
                except Exception as e:
                    logger.error(f"Retry failed for batch {batch_num}: {str(e)}")
                    # Fill with zero vectors for failed texts
                    all_embeddings.extend([[0.0] * self.dimension] * len(batch))

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        elapsed = time.time() - start_time
        logger.info(
            f"Generated {len(embeddings_array)} embeddings in {elapsed:.2f}s "
            f"({len(texts)/elapsed:.1f} texts/sec)"
        )

        return embeddings_array

    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # For Voyage AI, use "query" input type
        if self.provider == "voyage":
            try:
                response = self.client.embed(
                    texts=[query],
                    model=self.model,
                    input_type="query",  # Optimized for search queries
                )
                return np.array(response.embeddings[0], dtype=np.float32)
            except Exception as e:
                raise EmbeddingError(f"Failed to generate query embedding: {str(e)}")

        # For other providers, use regular embedding
        embeddings = await self.generate_embeddings([query])
        return embeddings[0]

    def get_dimension(self) -> int:
        """Get the embedding dimension for this model."""
        return self.dimension

    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current provider."""
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
        }


class DocumentChunker:
    """
    Splits documents into chunks for embedding.
    Implements token-based chunking with overlap.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunking_strategy: str = "recursive",
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            chunking_strategy: Strategy to use (recursive, semantic, fixed)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.chunking_strategy = chunking_strategy

        # Initialize token counter
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
        except ImportError:
            logger.warning("tiktoken not available, using approximate token counting")
            self.use_tiktoken = False

        logger.info(
            f"Initialized DocumentChunker: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, strategy={self.chunking_strategy}"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.use_tiktoken:
            return len(self.encoding.encode(text))
        else:
            # Approximate: ~4 characters per token
            return len(text) // 4

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunks with metadata
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        chunks = []

        # Split by paragraphs
        paragraphs = text.split("\n\n")

        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            # If paragraph is too long, split by sentences
            if para_tokens > self.chunk_size:
                sentences = para.split(". ")
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    sentence_tokens = self.count_tokens(sentence)

                    if current_tokens + sentence_tokens > self.chunk_size:
                        if current_chunk:
                            # Save current chunk
                            chunk_text = ". ".join(current_chunk)
                            chunks.append({
                                "content": chunk_text,
                                "metadata": {
                                    **metadata,
                                    "chunk_index": len(chunks),
                                    "token_count": self.count_tokens(chunk_text),
                                    "chunked_at": datetime.now().isoformat(),
                                },
                            })

                        # Start new chunk with overlap
                        if chunks and self.chunk_overlap > 0:
                            overlap_text = ". ".join(current_chunk[-2:]) if len(current_chunk) > 1 else ""
                            current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                            current_tokens = self.count_tokens(". ".join(current_chunk))
                        else:
                            current_chunk = [sentence]
                            current_tokens = sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sentence_tokens

            # Normal paragraph fits in chunk
            elif current_tokens + para_tokens > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": len(chunks),
                            "token_count": self.count_tokens(chunk_text),
                            "chunked_at": datetime.now().isoformat(),
                        },
                    })

                # Start new chunk with overlap
                if chunks and self.chunk_overlap > 0:
                    overlap_text = "\n\n".join(current_chunk[-1:])
                    current_chunk = [overlap_text, para] if overlap_text else [para]
                    current_tokens = self.count_tokens("\n\n".join(current_chunk))
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "token_count": self.count_tokens(chunk_text),
                    "chunked_at": datetime.now().isoformat(),
                },
            })

        logger.info(f"Chunked text into {len(chunks)} chunks (avg {np.mean([c['metadata']['token_count'] for c in chunks]):.0f} tokens)")
        return chunks


# Export main classes
__all__ = [
    "EmbeddingGenerator",
    "DocumentChunker",
    "EmbeddingError",
]
