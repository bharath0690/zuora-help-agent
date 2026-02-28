"""
FAISS Index Loader

Runtime utilities for loading and querying the FAISS index.
Use this in production to retrieve relevant chunks for RAG.

Example:
    from faiss_loader import FAISSIndexLoader

    # Initialize loader
    loader = FAISSIndexLoader(
        index_path="../data/faiss.index",
        metadata_path="../data/metadata.json"
    )

    # Search
    results = loader.search(query_embedding, top_k=5)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSIndexLoader:
    """
    Load and query FAISS index at runtime.

    Attributes:
        index: FAISS index
        metadata: Document chunk metadata
        dimension: Embedding dimension
        num_chunks: Number of indexed chunks
    """

    def __init__(
        self,
        index_path: str = "../data/faiss.index",
        metadata_path: str = "../data/metadata.json",
    ):
        """
        Initialize FAISS index loader.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file

        Raises:
            FileNotFoundError: If index or metadata file not found
            ValueError: If metadata format is invalid
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        # Load index
        self._load_index()

        # Load metadata
        self._load_metadata()

        logger.info(
            f"Loaded FAISS index: {self.num_chunks} chunks, "
            f"{self.dimension} dimensions"
        )

    def _load_index(self):
        """Load FAISS index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        self.dimension = self.index.d
        self.num_chunks = self.index.ntotal

        logger.info(f"Index loaded: {self.num_chunks} vectors, {self.dimension}d")

    def _load_metadata(self):
        """Load metadata from disk."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        logger.info(f"Loading metadata from {self.metadata_path}")

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract chunks
        if isinstance(data, dict):
            self.chunks = data.get("chunks", [])
            self.metadata_info = data.get("metadata", {})
        elif isinstance(data, list):
            self.chunks = data
            self.metadata_info = {}
        else:
            raise ValueError(f"Invalid metadata format in {self.metadata_path}")

        # Validate
        if len(self.chunks) != self.num_chunks:
            logger.warning(
                f"Metadata mismatch: {len(self.chunks)} chunks in metadata, "
                f"{self.num_chunks} in index"
            )

        logger.info(f"Metadata loaded: {len(self.chunks)} chunks")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Search for most similar chunks.

        Args:
            query_embedding: Query embedding vector (1D or 2D array)
            top_k: Number of results to return
            threshold: Optional similarity threshold (L2 distance)

        Returns:
            List of result dicts with: chunk_text, source, title, score, rank

        Example:
            results = loader.search(query_embedding, top_k=5)
            for result in results:
                print(f"Score: {result['score']:.3f}")
                print(f"Source: {result['source']}")
                print(f"Text: {result['chunk_text'][:100]}...")
        """
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Validate dimension
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension ({query_embedding.shape[1]}) "
                f"does not match index dimension ({self.dimension})"
            )

        # Search
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        # Build results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            # Skip if beyond threshold
            if threshold is not None and distance > threshold:
                continue

            # Get chunk metadata
            if idx < len(self.chunks):
                chunk = self.chunks[idx]

                result = {
                    "rank": rank + 1,
                    "score": float(distance),  # L2 distance (lower is better)
                    "similarity": 1 / (1 + distance),  # Convert to similarity (higher is better)
                    "chunk_text": chunk.get("chunk_text", ""),
                    "source": chunk.get("source", "unknown"),
                    "title": chunk.get("title", "Untitled"),
                }

                # Add optional fields if present
                optional_fields = ["url", "product", "chunk_index", "description"]
                for field in optional_fields:
                    if field in chunk:
                        result[field] = chunk[field]

                results.append(result)

        return results

    def search_with_metadata(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search with metadata filtering.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"product": "billing"})

        Returns:
            Filtered list of results
        """
        # Get more results than needed for filtering
        search_k = min(top_k * 3, self.num_chunks)
        results = self.search(query_embedding, top_k=search_k)

        # Apply filters
        if filters:
            filtered_results = []
            for result in results:
                match = True
                for key, value in filters.items():
                    chunk_value = result.get(key)
                    if chunk_value != value:
                        match = False
                        break

                if match:
                    filtered_results.append(result)

            results = filtered_results[:top_k]
        else:
            results = results[:top_k]

        # Update ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1

        return results

    def get_chunk_by_index(self, index: int) -> Optional[Dict]:
        """
        Get chunk metadata by index.

        Args:
            index: Chunk index

        Returns:
            Chunk metadata dict or None if not found
        """
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None

    def get_stats(self) -> Dict:
        """
        Get index statistics.

        Returns:
            Dict with index stats
        """
        return {
            "num_chunks": self.num_chunks,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "metadata_info": self.metadata_info,
            "has_metadata": len(self.chunks) > 0,
        }


# Convenience function for quick usage
def search_index(
    query_embedding: np.ndarray,
    top_k: int = 5,
    index_path: str = "../data/faiss.index",
    metadata_path: str = "../data/metadata.json",
) -> List[Dict]:
    """
    Quick search function - loads index and searches in one call.

    Args:
        query_embedding: Query embedding vector
        top_k: Number of results
        index_path: Path to FAISS index
        metadata_path: Path to metadata JSON

    Returns:
        List of search results

    Example:
        results = search_index(query_embedding, top_k=5)
    """
    loader = FAISSIndexLoader(index_path, metadata_path)
    return loader.search(query_embedding, top_k)


__all__ = ["FAISSIndexLoader", "search_index"]
