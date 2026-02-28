#!/usr/bin/env python3
"""
Document Ingestion Pipeline
Chunks scraped documentation, generates embeddings, and stores in FAISS.

Usage:
    python ingest_docs.py --input ../data/zuora_docs.json --output ../data/vector_store
    python ingest_docs.py --provider voyage --chunk-size 800
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import os
from datetime import datetime

import numpy as np
import faiss
from tqdm import tqdm

# Token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not installed. Using approximate token counting.")

# Embedding providers
try:
    import openai
except ImportError:
    openai = None

try:
    import voyageai
except ImportError:
    voyageai = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    chunk_id: str
    content: str
    metadata: Dict
    token_count: int
    embedding: Optional[np.ndarray] = None

    def to_dict(self):
        """Convert to dictionary (excluding embedding for JSON serialization)."""
        data = asdict(self)
        data.pop('embedding', None)  # Remove embedding - stored separately
        return data


class TokenCounter:
    """Token counting utility."""

    def __init__(self, model: str = "gpt-4"):
        """Initialize token counter."""
        self.model = model
        if tiktoken:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                logger.warning(f"Model {model} not found, using cl100k_base encoding")
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None
            logger.warning("Using approximate token counting (4 chars ≈ 1 token)")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Approximate: ~4 characters per token
            return len(text) // 4


class DocumentChunker:
    """Chunks documents into token-sized sections."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        model: str = "gpt-4",
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target size in tokens
            chunk_overlap: Overlap between chunks in tokens
            model: Model for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = TokenCounter(model)

    def chunk_document(self, doc: Dict, doc_index: int) -> List[DocumentChunk]:
        """
        Chunk a single document.

        Args:
            doc: Document dict with 'content' and 'metadata'
            doc_index: Index of document in source

        Returns:
            List of DocumentChunk objects
        """
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})

        if not content:
            logger.warning(f"Empty content for document: {metadata.get('url', 'unknown')}")
            return []

        # Split into paragraphs/sentences for better chunking
        paragraphs = content.split("\n\n")

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.token_counter.count_tokens(para)

            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        metadata,
                        doc_index,
                        chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long paragraph by sentences
                sentences = para.split(". ")
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    sentence_tokens = self.token_counter.count_tokens(sentence)

                    if current_tokens + sentence_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(
                                current_chunk,
                                metadata,
                                doc_index,
                                chunk_index
                            ))
                            chunk_index += 1

                        # Start new chunk with overlap
                        if chunks and self.chunk_overlap > 0:
                            # Take last few sentences for overlap
                            overlap_text = ". ".join(current_chunk[-2:]) if len(current_chunk) > 1 else ""
                            current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                            current_tokens = self.token_counter.count_tokens(". ".join(current_chunk))
                        else:
                            current_chunk = [sentence]
                            current_tokens = sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sentence_tokens

            # Normal paragraph - fits in chunk
            elif current_tokens + para_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        metadata,
                        doc_index,
                        chunk_index
                    ))
                    chunk_index += 1

                # Start new chunk with overlap
                if chunks and self.chunk_overlap > 0:
                    overlap_text = "\n\n".join(current_chunk[-1:])
                    current_chunk = [overlap_text, para] if overlap_text else [para]
                    current_tokens = self.token_counter.count_tokens("\n\n".join(current_chunk))
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                metadata,
                doc_index,
                chunk_index
            ))

        return chunks

    def _create_chunk(
        self,
        text_parts: List[str],
        source_metadata: Dict,
        doc_index: int,
        chunk_index: int,
    ) -> DocumentChunk:
        """Create a DocumentChunk from text parts."""
        content = "\n\n".join(text_parts)
        token_count = self.token_counter.count_tokens(content)

        # Create chunk metadata
        chunk_metadata = {
            **source_metadata,
            "chunk_index": chunk_index,
            "doc_index": doc_index,
            "chunked_at": datetime.now().isoformat(),
        }

        chunk_id = f"doc_{doc_index}_chunk_{chunk_index}"

        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=chunk_metadata,
            token_count=token_count,
        )


class EmbeddingGenerator:
    """Generate embeddings using various providers."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize embedding generator.

        Args:
            provider: 'openai', 'voyage', or 'local'
            model: Model name (provider-specific)
            api_key: API key for provider
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")

        # Set default models
        if model is None:
            if self.provider == "openai":
                model = "text-embedding-3-small"  # 1536 dims, cheap
            elif self.provider == "voyage":
                model = "voyage-2"  # Anthropic's recommended
            else:
                model = "all-MiniLM-L6-v2"  # Local model

        self.model = model
        self.dimension = None  # Will be set after first embedding

        # Initialize provider
        if self.provider == "openai":
            if not openai:
                raise ImportError("openai package required. Install: pip install openai")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY required")
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI embeddings with model: {model}")

        elif self.provider == "voyage":
            if not voyageai:
                raise ImportError("voyageai package required. Install: pip install voyageai")
            if not self.api_key:
                raise ValueError("VOYAGE_API_KEY required")
            self.client = voyageai.Client(api_key=self.api_key)
            logger.info(f"Initialized Voyage AI embeddings with model: {model}")

        elif self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self.client = SentenceTransformer(model)
                logger.info(f"Initialized local embeddings with model: {model}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for local embeddings. "
                    "Install: pip install sentence-transformers"
                )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for API calls

        Returns:
            Numpy array of embeddings (num_texts, embedding_dim)
        """
        embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]

            if self.provider == "openai":
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                )
                batch_embeddings = [item.embedding for item in response.data]

            elif self.provider == "voyage":
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                )
                batch_embeddings = response.embeddings

            elif self.provider == "local":
                batch_embeddings = self.client.encode(batch, convert_to_numpy=True)

            embeddings.extend(batch_embeddings)

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Set dimension on first run
        if self.dimension is None:
            self.dimension = embeddings_array.shape[1]
            logger.info(f"Embedding dimension: {self.dimension}")

        return embeddings_array


class FAISSVectorStore:
    """FAISS vector store for efficient similarity search."""

    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize FAISS index.

        Args:
            dimension: Embedding dimension
            index_type: 'flat' for exact search, 'ivf' for approximate
        """
        self.dimension = dimension
        self.index_type = index_type

        if index_type == "flat":
            # Exact search (best for < 1M vectors)
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # Approximate search (faster for large datasets)
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.chunks: List[DocumentChunk] = []
        logger.info(f"Initialized FAISS {index_type} index with dimension {dimension}")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the index."""
        # Train IVF index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Add embeddings
        self.index.add(embeddings)

        # Store chunks (for metadata retrieval)
        self.chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    def save(self, output_dir: Path):
        """Save index and metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = output_dir / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Save chunk metadata (without embeddings)
        metadata_path = output_dir / "chunks_metadata.pkl"
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(metadata_path, "wb") as f:
            pickle.dump(chunks_data, f)
        logger.info(f"Saved chunk metadata to {metadata_path}")

        # Save index config
        config_path = output_dir / "index_config.json"
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "num_chunks": len(self.chunks),
            "created_at": datetime.now().isoformat(),
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved index config to {config_path}")


def main():
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest scraped docs into FAISS vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI embeddings (default)
  python ingest_docs.py --input ../data/zuora_docs.json

  # Voyage AI embeddings (Anthropic recommended)
  python ingest_docs.py --provider voyage --input ../data/zuora_docs.json

  # Custom chunk size
  python ingest_docs.py --chunk-size 800 --chunk-overlap 100

  # Specify output directory
  python ingest_docs.py --output ../data/vector_store_v2
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../data/zuora_docs.json",
        help="Input JSON file with scraped docs",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/vector_store",
        help="Output directory for FAISS index",
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "voyage", "local"],
        default="openai",
        help="Embedding provider (default: openai)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Embedding model name (provider-specific)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size in tokens (default: 800)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap between chunks in tokens (default: 100)",
    )

    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf"],
        default="flat",
        help="FAISS index type (default: flat)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation (default: 100)",
    )

    args = parser.parse_args()

    # Load scraped documents
    logger.info(f"Loading documents from {args.input}")
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    with open(input_path) as f:
        data = json.load(f)

    documents = data.get("documents", [])
    logger.info(f"Loaded {len(documents)} documents")

    if not documents:
        logger.error("No documents found in input file")
        return

    # Chunk documents
    logger.info(f"Chunking documents (size: {args.chunk_size}, overlap: {args.chunk_overlap})")
    chunker = DocumentChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    all_chunks = []
    for i, doc in enumerate(tqdm(documents, desc="Chunking")):
        chunks = chunker.chunk_document(doc, i)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    logger.info(f"Average tokens per chunk: {np.mean([c.token_count for c in all_chunks]):.1f}")

    # Generate embeddings
    logger.info(f"Generating embeddings using {args.provider}")
    embedding_generator = EmbeddingGenerator(
        provider=args.provider,
        model=args.model,
    )

    chunk_texts = [chunk.content for chunk in all_chunks]
    embeddings = embedding_generator.embed_texts(chunk_texts, batch_size=args.batch_size)

    # Add embeddings to chunks
    for chunk, embedding in zip(all_chunks, embeddings):
        chunk.embedding = embedding

    # Create FAISS index
    logger.info("Building FAISS index")
    vector_store = FAISSVectorStore(
        dimension=embedding_generator.dimension,
        index_type=args.index_type,
    )
    vector_store.add_chunks(all_chunks, embeddings)

    # Save
    output_dir = Path(args.output)
    logger.info(f"Saving to {output_dir}")
    vector_store.save(output_dir)

    # Save ingestion summary
    summary_path = output_dir / "ingestion_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Document Ingestion Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {args.input}\n")
        f.write(f"Documents: {len(documents)}\n")
        f.write(f"Chunks: {len(all_chunks)}\n")
        f.write(f"Chunk Size: {args.chunk_size} tokens\n")
        f.write(f"Chunk Overlap: {args.chunk_overlap} tokens\n")
        f.write(f"Embedding Provider: {args.provider}\n")
        f.write(f"Embedding Model: {embedding_generator.model}\n")
        f.write(f"Embedding Dimension: {embedding_generator.dimension}\n")
        f.write(f"Index Type: {args.index_type}\n")
        f.write(f"Created: {datetime.now().isoformat()}\n\n")

        # Token statistics
        token_counts = [c.token_count for c in all_chunks]
        f.write(f"Token Statistics:\n")
        f.write(f"  Min: {min(token_counts)}\n")
        f.write(f"  Max: {max(token_counts)}\n")
        f.write(f"  Mean: {np.mean(token_counts):.1f}\n")
        f.write(f"  Median: {np.median(token_counts):.1f}\n")

    logger.info(f"Saved summary to {summary_path}")
    logger.info("✅ Ingestion complete!")


if __name__ == "__main__":
    main()
