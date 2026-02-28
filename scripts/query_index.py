#!/usr/bin/env python3
"""
Query FAISS Index
Test script to search the vector store and verify retrieval works.

Usage:
    python query_index.py --query "What is Zuora CPQ?"
    python query_index.py --query "How do I configure SSO?" --top-k 5
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import os

import numpy as np
import faiss

# Embedding providers (same as ingest_docs.py)
try:
    import openai
except ImportError:
    openai = None

try:
    import voyageai
except ImportError:
    voyageai = None


class VectorStoreQuery:
    """Query a FAISS vector store."""

    def __init__(
        self,
        vector_store_path: str,
        embedding_provider: str = "openai",
        embedding_model: str = None,
    ):
        """
        Initialize vector store query.

        Args:
            vector_store_path: Path to vector store directory
            embedding_provider: Provider for query embeddings
            embedding_model: Model name
        """
        self.vector_store_path = Path(vector_store_path)

        # Load index config
        config_path = self.vector_store_path / "index_config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        print(f"📚 Loaded index config:")
        print(f"   Dimension: {self.config['dimension']}")
        print(f"   Chunks: {self.config['num_chunks']}")
        print(f"   Created: {self.config['created_at']}")

        # Load FAISS index
        index_path = self.vector_store_path / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")

        # Load chunk metadata
        metadata_path = self.vector_store_path / "chunks_metadata.pkl"
        with open(metadata_path, "rb") as f:
            self.chunks_metadata = pickle.load(f)
        print(f"✅ Loaded {len(self.chunks_metadata)} chunk metadata entries\n")

        # Initialize embedding generator
        self.provider = embedding_provider
        api_key = os.getenv(f"{embedding_provider.upper()}_API_KEY")

        if embedding_model is None:
            embedding_model = "text-embedding-3-small" if embedding_provider == "openai" else "voyage-2"

        self.model = embedding_model

        if self.provider == "openai":
            if not openai:
                raise ImportError("openai package required")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required")
            self.client = openai.OpenAI(api_key=api_key)

        elif self.provider == "voyage":
            if not voyageai:
                raise ImportError("voyageai package required")
            if not api_key:
                raise ValueError("VOYAGE_API_KEY required")
            self.client = voyageai.Client(api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {embedding_provider}")

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=[query],
                model=self.model,
            )
            embedding = response.data[0].embedding

        elif self.provider == "voyage":
            response = self.client.embed(
                texts=[query],
                model=self.model,
            )
            embedding = response.embeddings[0]

        return np.array([embedding], dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (chunk_metadata, distance) tuples
        """
        # Generate query embedding
        query_embedding = self.embed_query(query)

        # Search index
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve chunks
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks_metadata):
                chunk = self.chunks_metadata[idx]
                results.append((chunk, float(distance)))

        return results

    def print_results(self, query: str, results: List[Tuple[Dict, float]]):
        """Pretty print search results."""
        print(f"🔍 Query: \"{query}\"\n")
        print(f"📊 Found {len(results)} results:\n")
        print("=" * 80)

        for i, (chunk, distance) in enumerate(results, 1):
            print(f"\n{i}. Score: {distance:.4f}")
            print(f"   Source: {chunk['metadata'].get('title', 'Unknown')}")
            print(f"   URL: {chunk['metadata'].get('url', 'N/A')}")
            print(f"   Product: {chunk['metadata'].get('product', 'N/A')}")
            print(f"   Tokens: {chunk['token_count']}")
            print(f"\n   Content Preview:")

            # Print first 300 chars of content
            content = chunk['content'][:300].replace("\n", " ")
            print(f"   {content}...")
            print("-" * 80)


def main():
    """Main query interface."""
    parser = argparse.ArgumentParser(
        description="Query FAISS vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python query_index.py --query "What is Zuora CPQ?"

  # More results
  python query_index.py --query "How to configure SSO?" --top-k 10

  # Voyage AI embeddings
  python query_index.py --provider voyage --query "Payment methods"
        """,
    )

    parser.add_argument(
        "--index",
        type=str,
        default="../data/vector_store",
        help="Path to vector store directory",
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "voyage"],
        default="openai",
        help="Embedding provider (default: openai)",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Embedding model name",
    )

    args = parser.parse_args()

    # Initialize query
    query_engine = VectorStoreQuery(
        vector_store_path=args.index,
        embedding_provider=args.provider,
        embedding_model=args.model,
    )

    # Search
    results = query_engine.search(args.query, top_k=args.top_k)

    # Display results
    query_engine.print_results(args.query, results)


if __name__ == "__main__":
    main()
