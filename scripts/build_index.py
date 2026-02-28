#!/usr/bin/env python3
"""
FAISS Index Builder

Reads processed documents, generates embeddings, and builds FAISS index for runtime use.

Input:  data/processed_docs.json
Output: data/faiss.index + data/metadata.json

Usage:
    python build_index.py
    python build_index.py --input ../data/processed_docs.json --provider voyage
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import numpy as np
import faiss

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from embeddings import EmbeddingGenerator, EmbeddingError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_processed_docs(input_path: Path) -> List[Dict]:
    """
    Load processed documents from JSON file.

    Args:
        input_path: Path to processed_docs.json

    Returns:
        List of document chunks

    Expected format:
        {
            "metadata": {...},
            "documents": [
                {
                    "content": "chunk text",
                    "metadata": {
                        "source": "file.md",
                        "title": "Document Title",
                        ...
                    }
                },
                ...
            ]
        }
    """
    logger.info(f"Loading processed documents from {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, dict):
        documents = data.get("documents", [])
        if not documents and "content" in data:
            # Single document format
            documents = [data]
    elif isinstance(data, list):
        documents = data
    else:
        raise ValueError(f"Unexpected data format in {input_path}")

    logger.info(f"Loaded {len(documents)} document chunks")
    return documents


def prepare_metadata(documents: List[Dict]) -> List[Dict]:
    """
    Extract and prepare metadata for storage.

    Args:
        documents: List of document chunks

    Returns:
        List of metadata dicts with: text, source, title

    Required fields:
        - chunk_text: The actual text content
        - source: Source file/URL
        - title: Document title
    """
    metadata_list = []

    for i, doc in enumerate(documents):
        # Extract content
        text = doc.get("content", "")
        if not text:
            logger.warning(f"Document {i} has no content, skipping")
            continue

        # Extract metadata
        doc_metadata = doc.get("metadata", {})

        # Build metadata dict with required fields
        metadata = {
            "chunk_text": text,
            "source": doc_metadata.get("source") or doc_metadata.get("url", "unknown"),
            "title": doc_metadata.get("title", "Untitled"),
        }

        # Add optional fields if present
        optional_fields = [
            "url",
            "product",
            "chunk_index",
            "doc_index",
            "word_count",
            "token_count",
            "description",
            "scraped_at",
            "chunked_at",
        ]

        for field in optional_fields:
            if field in doc_metadata:
                metadata[field] = doc_metadata[field]

        metadata_list.append(metadata)

    logger.info(f"Prepared metadata for {len(metadata_list)} chunks")
    return metadata_list


async def generate_embeddings_batch(
    texts: List[str],
    provider: str = "voyage",
    model: str = None,
    batch_size: int = 100,
) -> np.ndarray:
    """
    Generate embeddings for all texts.

    Args:
        texts: List of text chunks
        provider: Embedding provider (voyage/openai/local)
        model: Model name (provider-specific)
        batch_size: Batch size for API calls

    Returns:
        Numpy array of embeddings (num_texts, embedding_dim)
    """
    logger.info(f"Generating embeddings using {provider}")

    embedder = EmbeddingGenerator(
        provider=provider,
        model=model,
        batch_size=batch_size,
    )

    # Get provider info
    info = embedder.get_provider_info()
    logger.info(f"Using {info['model']} ({info['dimension']} dimensions)")

    # Generate embeddings
    start_time = time.time()
    embeddings = await embedder.generate_embeddings(texts)
    elapsed = time.time() - start_time

    logger.info(
        f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s "
        f"({len(texts)/elapsed:.1f} texts/sec)"
    )

    return embeddings


def build_faiss_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    Build FAISS index from embeddings.

    Args:
        embeddings: Numpy array of embeddings
        index_type: Index type (flat/ivf)

    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    logger.info(f"Building FAISS {index_type} index")
    logger.info(f"Vectors: {num_vectors}, Dimension: {dimension}")

    if index_type == "flat":
        # Flat index - exact search (best for < 1M vectors)
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

    elif index_type == "ivf":
        # IVF index - approximate search (faster for large datasets)
        nlist = min(100, num_vectors // 10)  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        # Train index
        logger.info("Training IVF index...")
        index.train(embeddings)
        index.add(embeddings)

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    logger.info(f"Index built with {index.ntotal} vectors")
    return index


def save_index(
    index: faiss.Index,
    metadata: List[Dict],
    index_path: Path,
    metadata_path: Path,
) -> None:
    """
    Save FAISS index and metadata to disk.

    Args:
        index: FAISS index
        metadata: List of metadata dicts
        index_path: Path to save index (faiss.index)
        metadata_path: Path to save metadata (metadata.json)
    """
    # Create output directory
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    logger.info(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, str(index_path))

    # Save metadata
    logger.info(f"Saving metadata to {metadata_path}")
    metadata_output = {
        "metadata": {
            "num_chunks": len(metadata),
            "dimension": index.d,
            "index_type": "flat" if isinstance(index, faiss.IndexFlat) else "ivf",
            "created_at": datetime.now().isoformat(),
        },
        "chunks": metadata,
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_output, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Index and metadata saved successfully")


def print_summary(
    num_chunks: int,
    dimension: int,
    index_path: Path,
    metadata_path: Path,
):
    """Print build summary."""
    print("\n" + "=" * 60)
    print("FAISS Index Build Summary")
    print("=" * 60)
    print(f"\n✅ Successfully built FAISS index")
    print(f"\nStatistics:")
    print(f"  Chunks:     {num_chunks:,}")
    print(f"  Dimension:  {dimension}")
    print(f"\nOutput Files:")
    print(f"  Index:      {index_path}")
    print(f"  Metadata:   {metadata_path}")
    print(f"\nIndex is ready for runtime use!")
    print(f"\nExample usage:")
    print(f"  from pathlib import Path")
    print(f"  import faiss")
    print(f"  import json")
    print(f"")
    print(f"  # Load index")
    print(f"  index = faiss.read_index('{index_path}')")
    print(f"")
    print(f"  # Load metadata")
    print(f"  with open('{metadata_path}') as f:")
    print(f"      metadata = json.load(f)")
    print(f"")
    print(f"  # Search")
    print(f"  distances, indices = index.search(query_embedding, k=5)")
    print(f"  results = [metadata['chunks'][i] for i in indices[0]]")
    print("=" * 60)


async def main():
    """Main build pipeline."""
    parser = argparse.ArgumentParser(
        description="Build FAISS index from processed documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (Voyage AI embeddings)
  python build_index.py

  # OpenAI embeddings
  python build_index.py --provider openai

  # Custom paths
  python build_index.py --input ../data/my_docs.json --output ../data/my_index

  # IVF index (for large datasets)
  python build_index.py --index-type ivf

  # Custom batch size
  python build_index.py --batch-size 50
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../data/processed_docs.json",
        help="Input JSON file with processed documents (default: ../data/processed_docs.json)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data",
        help="Output directory for index and metadata (default: ../data)",
    )

    parser.add_argument(
        "--provider",
        choices=["voyage", "openai", "local"],
        default="voyage",
        help="Embedding provider (default: voyage)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Embedding model name (provider-specific)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation (default: 100)",
    )

    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf"],
        default="flat",
        help="FAISS index type (default: flat)",
    )

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    index_path = output_dir / "faiss.index"
    metadata_path = output_dir / "metadata.json"

    try:
        # 1. Load processed documents
        documents = load_processed_docs(input_path)

        if not documents:
            logger.error("No documents found in input file")
            return 1

        # 2. Prepare metadata
        metadata = prepare_metadata(documents)

        if not metadata:
            logger.error("No valid chunks found with required metadata")
            return 1

        # 3. Extract texts for embedding
        texts = [m["chunk_text"] for m in metadata]

        # 4. Generate embeddings
        embeddings = await generate_embeddings_batch(
            texts,
            provider=args.provider,
            model=args.model,
            batch_size=args.batch_size,
        )

        if len(embeddings) != len(metadata):
            logger.error(
                f"Mismatch: {len(embeddings)} embeddings for {len(metadata)} chunks"
            )
            return 1

        # 5. Build FAISS index
        index = build_faiss_index(embeddings, index_type=args.index_type)

        # 6. Save index and metadata
        save_index(index, metadata, index_path, metadata_path)

        # 7. Print summary
        print_summary(len(metadata), embeddings.shape[1], index_path, metadata_path)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1

    except EmbeddingError as e:
        logger.error(f"Embedding generation failed: {e}")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
